"""Potok treningowy: ekstrakcja cech, porownanie algorytmow, ewaluacja.

Porownywane sa Random Forest, SVM, Naive Bayes, regresja logistyczna oraz siec
neuronowa (MLP). Kazdy model jest oceniany tymi samymi miarami: accuracy,
precision, recall i F1-score (w wariancie wazonym i makro), a dodatkowo
balanced accuracy, top-3 accuracy oraz czasy treningu i predykcji.
"""

import os
import shutil
import sys
import time
import warnings
from collections import Counter

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    top_k_accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize

from config import (
    CHAR_NGRAM_RANGE,
    LANGUAGE_INFO,
    MAX_CHAR_FEATURES,
    MAX_WORD_FEATURES,
    MODELS_DIR,
    OUTPUT_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    TEST_SIZE,
    VECTORIZER_FIT_SAMPLES,
    WORD_NGRAM_RANGE,
    language_name,
    language_script,
    script_name,
)
from models import build_model, get_spec, resolve_model_names
from preprocess import WORD_TOKEN_PATTERN

warnings.filterwarnings('ignore')

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CANDIDATES_DIR = os.path.join(MODELS_DIR, 'candidates')

for _directory in (OUTPUT_DIR, MODELS_DIR, REPORTS_DIR, CANDIDATES_DIR):
    os.makedirs(_directory, exist_ok=True)

# Jezyki pokazywane na krzywych ROC - reprezentanci roznych pism i rodzin.
ROC_PRIORITY = ('en', 'de', 'pl', 'ru', 'zh', 'ja', 'ko', 'hi', 'ar', 'th', 'el', 'he')

SCRIPT_COLORS = {
    'latin': '#3498db',
    'cyrillic': '#e74c3c',
    'greek': '#9b59b6',
    'arabic': '#16a085',
    'hebrew': '#2980b9',
    'devanagari': '#e67e22',
    'bengali': '#d35400',
    'han': '#c0392b',
    'japanese': '#8e44ad',
    'hangul': '#27ae60',
    'thai': '#f39c12',
}
_DEFAULT_SCRIPT_COLOR = '#7f8c8d'


def _script_color(script):
    return SCRIPT_COLORS.get(script, _DEFAULT_SCRIPT_COLOR)


def _header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# --- Ekstrakcja cech ---------------------------------------------------------

def _stratified_subsample(texts, labels, max_total, random_state=RANDOM_STATE):
    """Warstwowa podprobka o zadanym rozmiarze (proporcjonalna do klas)."""
    if labels is None or len(texts) <= max_total:
        if len(texts) <= max_total:
            return texts
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(texts), size=max_total, replace=False)
        return [texts[i] for i in idx]

    by_label = {}
    for i, label in enumerate(labels):
        by_label.setdefault(label, []).append(i)

    per_class = max(1, max_total // len(by_label))
    rng = np.random.default_rng(random_state)
    selected = []
    for indices in by_label.values():
        if len(indices) > per_class:
            indices = rng.choice(indices, size=per_class, replace=False)
        selected.extend(int(i) for i in indices)
    return [texts[i] for i in selected]


def extract_features(texts, labels=None):
    """Buduje macierz cech: TF-IDF na n-gramach znakowych i slownych.

    N-gramy znakowe zaczynaja sie od unigramow - pojedynczy znak niesie w tym
    zadaniu duzo informacji, bo od razu identyfikuje pismo (np. hiragana, kana,
    dewanagari). Tokenizacja slowna uzywa wzorca rozbijajacego pisma bez spacji
    (chinski, japonski, tajski) na osobne znaki.
    """
    _header("EKSTRAKCJA CECH")
    print(f"  Liczba tekstow: {len(texts)}")

    fit_texts = _stratified_subsample(texts, labels, VECTORIZER_FIT_SAMPLES)
    if len(fit_texts) < len(texts):
        print(f"  Slownik n-gramow budowany na podprobce: {len(fit_texts)} tekstow")

    # Odsiewanie n-gramow z jednego dokumentu ma sens tylko na duzym korpusie;
    # na malej probce (testy, szybki przebieg) usunelo by cale slownictwo.
    min_df = 3 if len(fit_texts) >= 500 else 1

    print(f"  N-gramy znakowe {CHAR_NGRAM_RANGE}...")
    start = time.time()
    char_vectorizer = TfidfVectorizer(
        analyzer='char',
        ngram_range=CHAR_NGRAM_RANGE,
        max_features=MAX_CHAR_FEATURES,
        sublinear_tf=True,
        lowercase=True,
        min_df=min_df,
        dtype=np.float32,
    )
    char_vectorizer.fit(fit_texts)
    X_char = char_vectorizer.transform(texts)
    print(f"    Cechy znakowe: {X_char.shape[1]} (czas: {time.time() - start:.1f}s)")

    print(f"  N-gramy slowne {WORD_NGRAM_RANGE}...")
    start = time.time()
    word_vectorizer = TfidfVectorizer(
        analyzer='word',
        token_pattern=WORD_TOKEN_PATTERN,
        ngram_range=WORD_NGRAM_RANGE,
        max_features=MAX_WORD_FEATURES,
        sublinear_tf=True,
        lowercase=True,
        min_df=min_df,
        dtype=np.float32,
    )
    word_vectorizer.fit(fit_texts)
    X_word = word_vectorizer.transform(texts)
    print(f"    Cechy slowne: {X_word.shape[1]} (czas: {time.time() - start:.1f}s)")

    X = csr_matrix(hstack([X_char, X_word]))
    print(f"    Lacznie cech: {X.shape[1]}, gestosc: {X.nnz / (X.shape[0] * X.shape[1]):.5f}")
    return X, (char_vectorizer, word_vectorizer)


# --- Analiza eksploracyjna ---------------------------------------------------

def perform_eda(texts, labels, scripts, output_dir=OUTPUT_DIR):
    _header("ANALIZA EKSPLORACYJNA DANYCH (EDA)")
    print(f"  Liczba tekstow: {len(texts)}")
    print(f"  Liczba jezykow: {len(set(labels))}")
    print(f"  Liczba pism:    {len(set(scripts))}")

    text_lengths = [len(t) for t in texts]
    lang_counts = Counter(labels)
    script_counts = Counter(scripts)
    family_counts = Counter(
        LANGUAGE_INFO.get(label, {}).get('family', 'Inne') for label in labels
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Analiza eksploracyjna danych - detekcja jezyka i pisma',
                 fontsize=16, fontweight='bold')

    axes[0, 0].hist(text_lengths, bins=50, color='steelblue', edgecolor='black', alpha=0.75)
    axes[0, 0].axvline(np.mean(text_lengths), color='red', linestyle='--', linewidth=2,
                       label=f'Srednia: {np.mean(text_lengths):.0f}')
    axes[0, 0].axvline(np.median(text_lengths), color='green', linestyle='--', linewidth=2,
                       label=f'Mediana: {np.median(text_lengths):.0f}')
    axes[0, 0].set_xlabel('Dlugosc tekstu (znaki)')
    axes[0, 0].set_ylabel('Liczba tekstow')
    axes[0, 0].set_title('Rozklad dlugosci tekstow', fontweight='bold')
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)

    top_langs = lang_counts.most_common(25)
    names = [f"{code} ({language_name(code)})" for code, _ in top_langs]
    counts = [count for _, count in top_langs]
    colors = [_script_color(language_script(code)) for code, _ in top_langs]
    axes[0, 1].barh(range(len(names)), counts, color=colors)
    axes[0, 1].set_yticks(range(len(names)))
    axes[0, 1].set_yticklabels(names, fontsize=7)
    axes[0, 1].set_xlabel('Liczba probek')
    axes[0, 1].set_title('Top 25 jezykow (kolor = pismo)', fontweight='bold')
    axes[0, 1].invert_yaxis()
    axes[0, 1].grid(True, alpha=0.3, axis='x')

    families = [f for f, _ in family_counts.most_common()]
    family_values = [v for _, v in family_counts.most_common()]
    axes[1, 0].pie(family_values, labels=None, autopct=None,
                   colors=plt.cm.tab20(np.linspace(0, 1, len(families))))
    axes[1, 0].set_title('Rozklad rodzin jezykowych', fontweight='bold')
    axes[1, 0].legend(families, loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=7)

    script_items = script_counts.most_common()
    script_labels = [script_name(s) for s, _ in script_items]
    script_values = [v for _, v in script_items]
    axes[1, 1].bar(range(len(script_labels)), script_values,
                   color=[_script_color(s) for s, _ in script_items], edgecolor='black')
    axes[1, 1].set_xticks(range(len(script_labels)))
    axes[1, 1].set_xticklabels(script_labels, rotation=75, ha='right', fontsize=7)
    axes[1, 1].set_ylabel('Liczba tekstow')
    axes[1, 1].set_title(f'Rozklad pism ({len(script_items)} klas)', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(output_dir, 'eda_analysis.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Wykresy EDA: {path}")

    return text_lengths, lang_counts, family_counts


# --- Miary jakosci -----------------------------------------------------------

def evaluate_predictions(y_true, y_pred, y_proba=None, n_classes=None):
    """Liczy pelny zestaw miar wymaganych w analizie jakosci modeli."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
    }

    if y_proba is not None and n_classes and n_classes > 3:
        metrics['top3_accuracy'] = top_k_accuracy_score(
            y_true, y_proba, k=3, labels=list(range(n_classes))
        )
    else:
        metrics['top3_accuracy'] = metrics['accuracy']
    return metrics


def _drop_rare_classes(X, y, min_count=5):
    """Usuwa klasy o zbyt malej liczbie probek na warstwowy podzial zbioru."""
    counts = Counter(y)
    rare = {label for label, count in counts.items() if count < min_count}
    if not rare:
        return X, y, []
    mask = np.array([label not in rare for label in y])
    print(f"  Pominieto {len(rare)} klas o mniej niz {min_count} probkach: {sorted(rare)}")
    return X[mask], list(np.asarray(y, dtype=object)[mask]), sorted(rare)


# --- Wykresy -----------------------------------------------------------------

def _plot_confusion_matrix(y_true, y_pred, classes, title, path, cmap='Blues'):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    n = len(classes)
    annotate = n <= 30
    size = max(10, min(30, n * 0.28))

    plt.figure(figsize=(size, size * 0.92))
    sns.heatmap(
        cm,
        cmap=cmap,
        xticklabels=classes,
        yticklabels=classes,
        annot=annotate,
        fmt='d' if annotate else '',
        annot_kws={'size': 8} if annotate else None,
        cbar_kws={'label': 'Liczba probek'},
        linewidths=0.2 if n <= 60 else 0,
        linecolor='white',
    )
    plt.xlabel('Klasa przewidziana', fontsize=13, fontweight='bold')
    plt.ylabel('Klasa rzeczywista', fontsize=13, fontweight='bold')
    plt.title(title, fontsize=15, fontweight='bold')
    tick_size = 8 if n <= 40 else 6
    plt.xticks(rotation=90, fontsize=tick_size)
    plt.yticks(rotation=0, fontsize=tick_size)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"  Macierz pomylek: {path}")
    return cm


def _select_roc_languages(classes, limit=12):
    present = list(classes)
    ordered = [code for code in ROC_PRIORITY if code in present]

    by_script = {}
    for code in present:
        by_script.setdefault(language_script(code), []).append(code)
    for script in sorted(by_script):
        candidate = by_script[script][0]
        if candidate not in ordered:
            ordered.append(candidate)

    for code in present:
        if code not in ordered:
            ordered.append(code)
    return ordered[:limit]


def _plot_roc_curves(y_test, y_proba, classes, output_dir=OUTPUT_DIR):
    n_classes = len(classes)
    selected = _select_roc_languages(classes, limit=12)
    if not selected:
        return {}

    y_test_bin = label_binarize(y_test, classes=range(n_classes))
    class_list = list(classes)
    auc_scores = {}

    n_cols = 4
    n_rows = (len(selected) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4.6), squeeze=False)
    fig.suptitle('Krzywe ROC (One-vs-Rest) - powiekszony zakres',
                 fontsize=17, fontweight='bold')

    for i, code in enumerate(selected):
        idx = class_list.index(code)
        fpr, tpr, _ = roc_curve(y_test_bin[:, idx], y_proba[:, idx])
        roc_auc = auc(fpr, tpr)
        auc_scores[code] = roc_auc

        ax = axes[i // n_cols][i % n_cols]
        color = _script_color(language_script(code))
        ax.plot(fpr, tpr, color=color, lw=2.5, label=f'AUC = {roc_auc:.4f}')
        ax.plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.4)
        ax.fill_between(fpr, tpr, alpha=0.18, color=color)
        ax.set_xlim(0.0, 0.1)
        ax.set_ylim(0.9, 1.0)
        ax.set_xlabel('False Positive Rate', fontsize=9)
        ax.set_ylabel('True Positive Rate', fontsize=9)
        ax.set_title(f"{code.upper()} - {language_name(code)}\n[{script_name(language_script(code))}]",
                     fontsize=11, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')

    for i in range(len(selected), n_rows * n_cols):
        axes[i // n_cols][i % n_cols].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, 'roc_curves.png')
    plt.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"  Krzywe ROC: {path}")

    plt.figure(figsize=(14, 10))
    colors = plt.cm.tab20(np.linspace(0, 1, len(selected)))
    for i, code in enumerate(selected):
        idx = class_list.index(code)
        fpr, tpr, _ = roc_curve(y_test_bin[:, idx], y_proba[:, idx])
        plt.plot(fpr, tpr, color=colors[i], lw=2.2,
                 label=f'{code.upper()} ({language_name(code)}) - AUC = {auc_scores[code]:.4f}')
    plt.plot([0, 1], [0, 1], 'k:', lw=1.5, alpha=0.6, label='Klasyfikator losowy')
    plt.xlim(0.0, 0.1)
    plt.ylim(0.9, 1.0)
    plt.xlabel('False Positive Rate (1 - specyficznosc)', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate (czulosc)', fontsize=12, fontweight='bold')
    plt.title('Porownanie krzywych ROC dla reprezentantow roznych pism',
              fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=8, ncol=2)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    path = os.path.join(output_dir, 'roc_curves_combined.png')
    plt.savefig(path, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"  Polaczone krzywe ROC: {path}")

    return auc_scores


def _plot_per_language_f1(y_test, y_pred, classes, output_dir=OUTPUT_DIR):
    """F1 dla kazdego jezyka - pokazuje, ktore klasy sa najtrudniejsze."""
    scores = f1_score(y_test, y_pred, average=None, labels=range(len(classes)), zero_division=0)
    order = np.argsort(scores)
    codes = [classes[i] for i in order]
    values = scores[order]

    height = max(6, len(codes) * 0.22)
    plt.figure(figsize=(11, height))
    colors = [_script_color(language_script(code)) for code in codes]
    plt.barh(range(len(codes)), values, color=colors, edgecolor='black', linewidth=0.3)
    plt.yticks(range(len(codes)),
               [f"{code} ({language_name(code)})" for code in codes], fontsize=7)
    plt.xlabel('F1-score', fontsize=12, fontweight='bold')
    plt.title('F1-score per jezyk (rosnaco, kolor = pismo)', fontsize=14, fontweight='bold')
    plt.axvline(float(np.mean(values)), color='red', linestyle='--', linewidth=1.5,
                label=f'Srednia (macro): {np.mean(values):.3f}')
    plt.xlim(0, 1.02)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    path = os.path.join(output_dir, 'f1_per_language.png')
    plt.savefig(path, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"  F1 per jezyk: {path}")
    return dict(zip(classes, scores))


def _plot_model_comparison(results, output_dir=OUTPUT_DIR):
    labels = [row['label'] for row in results]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(17, 11))
    fig.suptitle('Porownanie algorytmow uczenia maszynowego - detekcja jezyka',
                 fontsize=16, fontweight='bold')

    metric_keys = [
        ('accuracy', 'Accuracy', '#3498db'),
        ('precision_weighted', 'Precision', '#2ecc71'),
        ('recall_weighted', 'Recall', '#f39c12'),
        ('f1_weighted', 'F1-score', '#e74c3c'),
    ]
    width = 0.2
    for i, (key, name, color) in enumerate(metric_keys):
        values = [row[key] for row in results]
        bars = axes[0, 0].bar(x + (i - 1.5) * width, values, width, label=name, color=color)
        for bar, value in zip(bars, values):
            axes[0, 0].text(bar.get_x() + bar.get_width() / 2, value + 0.008,
                            f'{value:.3f}', ha='center', fontsize=6, rotation=90)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels, rotation=25, ha='right', fontsize=8)
    axes[0, 0].set_ylabel('Wartosc miary')
    axes[0, 0].set_ylim(0, 1.12)
    axes[0, 0].set_title('Miary wazone (weighted average)', fontweight='bold')
    axes[0, 0].legend(fontsize=9, ncol=4, loc='lower right')
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    weighted = [row['f1_weighted'] for row in results]
    macro = [row['f1_macro'] for row in results]
    axes[0, 1].bar(x - 0.2, weighted, 0.4, label='F1 wazone', color='#e74c3c')
    axes[0, 1].bar(x + 0.2, macro, 0.4, label='F1 makro', color='#8e44ad')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(labels, rotation=25, ha='right', fontsize=8)
    axes[0, 1].set_ylabel('F1-score')
    axes[0, 1].set_ylim(0, 1.08)
    axes[0, 1].set_title('F1 wazone vs makro (wplyw klas rzadkich)', fontweight='bold')
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    train_times = [row['train_time'] for row in results]
    predict_times = [row['predict_time'] for row in results]
    axes[1, 0].bar(x - 0.2, train_times, 0.4, label='Trening', color='#34495e')
    axes[1, 0].bar(x + 0.2, predict_times, 0.4, label='Predykcja', color='#95a5a6')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels, rotation=25, ha='right', fontsize=8)
    axes[1, 0].set_ylabel('Czas [s], skala logarytmiczna')
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_title('Koszt obliczeniowy', fontweight='bold')
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    accuracy = [row['accuracy'] for row in results]
    top3 = [row['top3_accuracy'] for row in results]
    balanced = [row['balanced_accuracy'] for row in results]
    axes[1, 1].plot(x, accuracy, 'o-', lw=2, label='Accuracy', color='#3498db')
    axes[1, 1].plot(x, top3, 's-', lw=2, label='Top-3 accuracy', color='#27ae60')
    axes[1, 1].plot(x, balanced, '^-', lw=2, label='Balanced accuracy', color='#f39c12')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels, rotation=25, ha='right', fontsize=8)
    axes[1, 1].set_ylabel('Wartosc miary')
    axes[1, 1].set_title('Accuracy w trzech wariantach', fontweight='bold')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"  Wykres porownania modeli: {path}")


# --- Porownanie algorytmow ---------------------------------------------------

def compare_models(X, y, model_names=None, test_size=TEST_SIZE, keep_candidates=False):
    """Trenuje i ocenia kolejne algorytmy na tym samym podziale danych."""
    model_names = resolve_model_names(model_names)

    _header("POROWNANIE ALGORYTMOW UCZENIA MASZYNOWEGO")

    X, y, _ = _drop_rare_classes(X, y)
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)
    classes = list(label_encoder.classes_)
    n_classes = len(classes)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=test_size, random_state=RANDOM_STATE, stratify=y_enc
    )

    print(f"  Klasy:             {n_classes}")
    print(f"  Zbior treningowy:  {X_train.shape[0]} probek")
    print(f"  Zbior testowy:     {X_test.shape[0]} probek")
    print(f"  Cechy:             {X_train.shape[1]}")
    print(f"  Algorytmy:         {', '.join(model_names)}")

    results = []
    best = None

    for position, name in enumerate(model_names, 1):
        spec = get_spec(name)
        print(f"\n  [{position}/{len(model_names)}] {spec.label}")
        print(f"      {spec.notes}")

        model = build_model(name)
        try:
            start = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start

            start = time.time()
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            predict_time = time.time() - start
        except (MemoryError, ValueError) as exc:
            print(f"      POMINIETO - trening nieudany: {str(exc)[:120]}")
            continue

        metrics = evaluate_predictions(y_test, y_pred, y_proba, n_classes)
        row = {
            'model': name,
            'label': spec.label,
            'family': spec.family,
            'train_time': train_time,
            'predict_time': predict_time,
            **metrics,
        }
        results.append(row)

        print(f"      Accuracy: {metrics['accuracy']:.4f} | "
              f"Precision: {metrics['precision_weighted']:.4f} | "
              f"Recall: {metrics['recall_weighted']:.4f} | "
              f"F1: {metrics['f1_weighted']:.4f}")
        print(f"      F1 makro: {metrics['f1_macro']:.4f} | "
              f"Top-3: {metrics['top3_accuracy']:.4f} | "
              f"trening: {train_time:.1f}s, predykcja: {predict_time:.1f}s")

        report_path = os.path.join(REPORTS_DIR, f'classification_report_{name}.txt')
        with open(report_path, 'w', encoding='utf-8') as handle:
            handle.write(f"{spec.label} - raport klasyfikacji per jezyk\n")
            handle.write("=" * 70 + "\n\n")
            handle.write(classification_report(
                y_test, y_pred, labels=range(n_classes),
                target_names=classes, zero_division=0
            ))

        candidate_path = os.path.join(CANDIDATES_DIR, f'{name}.pkl')
        joblib.dump(model, candidate_path, compress=3)

        if best is None or metrics['f1_weighted'] > best['metrics']['f1_weighted']:
            best = {
                'name': name,
                'label': spec.label,
                'metrics': metrics,
                'y_pred': y_pred,
                'y_proba': y_proba,
                'path': candidate_path,
            }

        # Model jest juz na dysku - zwalniamy pamiec przed kolejnym algorytmem.
        del model

    if not results:
        raise RuntimeError("Zaden z wybranych algorytmow nie zakonczyl treningu.")

    results.sort(key=lambda row: row['f1_weighted'], reverse=True)
    _plot_model_comparison(results)
    _write_comparison_report(results, n_classes, X_train.shape, X_test.shape[0])

    best['model'] = joblib.load(best['path'])
    if not keep_candidates:
        for row in results:
            if row['model'] != best['name']:
                candidate = os.path.join(CANDIDATES_DIR, f"{row['model']}.pkl")
                if os.path.exists(candidate):
                    os.remove(candidate)

    return {
        'results': results,
        'best': best,
        'label_encoder': label_encoder,
        'classes': classes,
        'y_test': y_test,
        'n_train': X_train.shape[0],
        'n_test': X_test.shape[0],
        'n_features': X_train.shape[1],
    }


def _write_comparison_report(results, n_classes, train_shape, n_test):
    frame = pd.DataFrame(results)
    csv_path = os.path.join(REPORTS_DIR, 'model_comparison.csv')
    frame.to_csv(csv_path, index=False, encoding='utf-8')

    columns = [
        ('label', 'Model', 28),
        ('accuracy', 'Accuracy', 10),
        ('precision_weighted', 'Precision', 10),
        ('recall_weighted', 'Recall', 10),
        ('f1_weighted', 'F1', 10),
        ('f1_macro', 'F1 makro', 10),
        ('top3_accuracy', 'Top-3', 10),
        ('train_time', 'Trening[s]', 11),
    ]

    lines = [
        "=" * 70,
        "POROWNANIE ALGORYTMOW - DETEKCJA JEZYKA",
        "=" * 70,
        "",
        f"Data:              {pd.Timestamp.now():%Y-%m-%d %H:%M:%S}",
        f"Liczba klas:       {n_classes}",
        f"Zbior treningowy:  {train_shape[0]} probek",
        f"Zbior testowy:     {n_test} probek",
        f"Liczba cech:       {train_shape[1]}",
        "",
        "-" * 70,
    ]
    lines.append("".join(title.ljust(width) for _, title, width in columns))
    lines.append("-" * 70)
    for row in results:
        cells = []
        for key, _, width in columns:
            value = row[key]
            text = value if isinstance(value, str) else f"{value:.4f}"
            if key == 'train_time':
                text = f"{value:.1f}"
            cells.append(str(text).ljust(width))
        lines.append("".join(cells))
    lines.append("-" * 70)
    lines.append("")
    lines.append(f"Najlepszy model (po F1 wazonym): {results[0]['label']}")
    lines.append("")
    lines.append("Miary wazone uwzgledniaja liczebnosc klas, makro traktuje kazdy jezyk")
    lines.append("jednakowo - rozbieznosc miedzy nimi wskazuje na problemy z jezykami")
    lines.append("o mniejszej liczbie probek.")

    txt_path = os.path.join(REPORTS_DIR, 'model_comparison.txt')
    with open(txt_path, 'w', encoding='utf-8') as handle:
        handle.write("\n".join(lines))

    print(f"\n  Tabela porownania: {txt_path}")
    print(f"  Dane CSV:          {csv_path}")
    print("\n" + "\n".join(lines[10:]))


# --- Model jezyka ------------------------------------------------------------

def train_language_model(X, y, model_names=None, keep_candidates=False):
    """Porownuje algorytmy i zapisuje najlepszy jako model produkcyjny."""
    comparison = compare_models(X, y, model_names, keep_candidates=keep_candidates)

    best = comparison['best']
    classes = comparison['classes']
    y_test = comparison['y_test']

    _header(f"EWALUACJA NAJLEPSZEGO MODELU: {best['label']}")
    metrics = best['metrics']
    print(f"  Accuracy:           {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print(f"  Balanced accuracy:  {metrics['balanced_accuracy']:.4f}")
    print(f"  Precision (wazone): {metrics['precision_weighted']:.4f}")
    print(f"  Recall (wazone):    {metrics['recall_weighted']:.4f}")
    print(f"  F1-score (wazone):  {metrics['f1_weighted']:.4f}")
    print(f"  F1-score (makro):   {metrics['f1_macro']:.4f}")
    print(f"  Top-3 accuracy:     {metrics['top3_accuracy']:.4f}")

    report = classification_report(
        y_test, best['y_pred'], labels=range(len(classes)),
        target_names=classes, zero_division=0
    )
    report_path = os.path.join(OUTPUT_DIR, 'classification_report.txt')
    with open(report_path, 'w', encoding='utf-8') as handle:
        handle.write("=" * 70 + "\n")
        handle.write("RAPORT KLASYFIKACJI - MODEL DETEKCJI JEZYKA\n")
        handle.write("=" * 70 + "\n\n")
        handle.write(f"Data utworzenia:   {pd.Timestamp.now():%Y-%m-%d %H:%M:%S}\n")
        handle.write(f"Wybrany algorytm:  {best['label']}\n\n")
        handle.write("OGOLNE METRYKI:\n")
        handle.write(f"  Liczba klas:        {len(classes)}\n")
        for key, value in metrics.items():
            handle.write(f"  {key:<20}{value:.4f}\n")
        handle.write(f"\n  Zbior treningowy:   {comparison['n_train']} probek\n")
        handle.write(f"  Zbior testowy:      {comparison['n_test']} probek\n")
        handle.write(f"  Liczba cech:        {comparison['n_features']}\n\n")
        handle.write("-" * 70 + "\n")
        handle.write("SZCZEGOLOWY RAPORT PER JEZYK:\n")
        handle.write("-" * 70 + "\n\n")
        handle.write(report)
    print(f"  Raport klasyfikacji: {report_path}")

    _plot_confusion_matrix(
        y_test, best['y_pred'], classes,
        f'Macierz pomylek - detekcja jezyka ({best["label"]})',
        os.path.join(OUTPUT_DIR, 'confusion_matrix_language.png'),
    )
    _plot_per_language_f1(y_test, best['y_pred'], classes)
    _plot_roc_curves(y_test, best['y_proba'], classes)

    joblib.dump(best['model'], os.path.join(MODELS_DIR, 'language_model.pkl'), compress=3)
    joblib.dump(comparison['label_encoder'], os.path.join(MODELS_DIR, 'language_encoder.pkl'))
    joblib.dump(
        {
            'algorithm': best['name'],
            'label': best['label'],
            'metrics': metrics,
            'comparison': comparison['results'],
            'n_classes': len(classes),
            'trained_at': pd.Timestamp.now().isoformat(),
        },
        os.path.join(MODELS_DIR, 'model_metadata.pkl'),
    )
    print(f"  Model zapisany w: {MODELS_DIR}/language_model.pkl")

    return best['model'], comparison['label_encoder'], metrics, comparison['results']


# --- Model pisma -------------------------------------------------------------

def train_script_model(X, y, model_name='logistic_regression'):
    """Osobny klasyfikator rozpoznajacy pismo (alfabet) tekstu.

    Domyslnie regresja logistyczna, a nie SVM: rozpoznanie pisma jest zadaniem
    znacznie latwiejszym od rozpoznania jezyka, a softmax regresji daje
    prawdopodobienstwa nadajace sie do pokazania uzytkownikowi. Softmax na
    marginesach SVM byłby dla 27 klas rozmyty i mylacy w interfejsie.
    """
    _header("TRENOWANIE MODELU PISMA (ALFABETU)")

    X, y, _ = _drop_rare_classes(X, y)
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)
    classes = list(label_encoder.classes_)
    print(f"  Klasy pism ({len(classes)}): {', '.join(classes)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_enc
    )

    spec = get_spec(model_name)
    print(f"  Algorytm: {spec.label}")
    start = time.time()
    model = build_model(model_name)
    model.fit(X_train, y_train)
    print(f"  Wytrenowano w {time.time() - start:.1f}s")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    metrics = evaluate_predictions(y_test, y_pred, y_proba, len(classes))

    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print(f"  Precision: {metrics['precision_weighted']:.4f}")
    print(f"  Recall:    {metrics['recall_weighted']:.4f}")
    print(f"  F1-score:  {metrics['f1_weighted']:.4f}")

    _plot_confusion_matrix(
        y_test, y_pred, classes,
        'Macierz pomylek - rozpoznawanie pisma',
        os.path.join(OUTPUT_DIR, 'confusion_matrix_script.png'),
        cmap='Reds',
    )

    report_path = os.path.join(REPORTS_DIR, 'classification_report_script.txt')
    with open(report_path, 'w', encoding='utf-8') as handle:
        handle.write("RAPORT KLASYFIKACJI - MODEL PISMA\n")
        handle.write("=" * 70 + "\n\n")
        handle.write(classification_report(
            y_test, y_pred, labels=range(len(classes)),
            target_names=classes, zero_division=0
        ))

    joblib.dump(model, os.path.join(MODELS_DIR, 'script_model.pkl'), compress=3)
    joblib.dump(label_encoder, os.path.join(MODELS_DIR, 'script_encoder.pkl'))
    print(f"  Model pisma zapisany w: {MODELS_DIR}/script_model.pkl")

    return model, label_encoder, metrics


# --- Pelny potok -------------------------------------------------------------

def train_complete_model(texts, labels, scripts, model_names=None,
                         keep_candidates=False, run_eda=True):
    _header("PELNY TRENING SYSTEMU DETEKCJI JEZYKA I PISMA")
    start_time = pd.Timestamp.now()

    if run_eda:
        perform_eda(texts, labels, scripts)

    X, vectorizers = extract_features(texts, labels)

    lang_model, lang_encoder, lang_metrics, comparison = train_language_model(
        X, labels, model_names=model_names, keep_candidates=keep_candidates
    )
    script_model, script_encoder, script_metrics = train_script_model(X, scripts)

    joblib.dump(vectorizers, os.path.join(MODELS_DIR, 'vectorizers.pkl'), compress=3)
    joblib.dump(LANGUAGE_INFO, os.path.join(MODELS_DIR, 'language_info.pkl'))

    if not keep_candidates and os.path.isdir(CANDIDATES_DIR) and not os.listdir(CANDIDATES_DIR):
        shutil.rmtree(CANDIDATES_DIR, ignore_errors=True)

    training_time = (pd.Timestamp.now() - start_time).total_seconds()

    _header("PODSUMOWANIE TRENINGU")
    print(f"  Czas treningu: {training_time:.0f}s ({training_time / 60:.1f} min)")
    print("\n  MODEL JEZYKA:")
    print(f"    Wybrany algorytm: {comparison[0]['label']}")
    print(f"    Liczba jezykow:   {len(lang_encoder.classes_)}")
    print(f"    Accuracy:         {lang_metrics['accuracy']:.2%}")
    print(f"    Precision:        {lang_metrics['precision_weighted']:.2%}")
    print(f"    Recall:           {lang_metrics['recall_weighted']:.2%}")
    print(f"    F1-score:         {lang_metrics['f1_weighted']:.2%}")
    print("\n  MODEL PISMA:")
    print(f"    Liczba pism:      {len(script_encoder.classes_)}")
    print(f"    Accuracy:         {script_metrics['accuracy']:.2%}")
    print(f"    F1-score:         {script_metrics['f1_weighted']:.2%}")

    print("\n  PLIKI WYNIKOWE:")
    for name in ('eda_analysis.png', 'model_comparison.png', 'confusion_matrix_language.png',
                 'confusion_matrix_script.png', 'f1_per_language.png', 'roc_curves.png',
                 'roc_curves_combined.png', 'classification_report.txt'):
        print(f"    {os.path.join(OUTPUT_DIR, name)}")
    print(f"    {REPORTS_DIR}/model_comparison.csv (+ raporty per algorytm)")
    print(f"    {MODELS_DIR}/language_model.pkl, script_model.pkl, vectorizers.pkl")

    print("\n  >> Trening zakonczony pomyslnie!")

    return {
        'best_algorithm': comparison[0]['model'],
        'best_algorithm_label': comparison[0]['label'],
        'language_accuracy': lang_metrics['accuracy'],
        'language_precision': lang_metrics['precision_weighted'],
        'language_recall': lang_metrics['recall_weighted'],
        'language_f1': lang_metrics['f1_weighted'],
        'language_f1_macro': lang_metrics['f1_macro'],
        'script_accuracy': script_metrics['accuracy'],
        'script_f1': script_metrics['f1_weighted'],
        'comparison': comparison,
        'n_languages': len(lang_encoder.classes_),
        'n_scripts': len(script_encoder.classes_),
        'training_time_seconds': training_time,
    }


if __name__ == "__main__":
    print("Modul treningowy. Uruchom 'python main.py train' albo 'python main_train.py'.")
