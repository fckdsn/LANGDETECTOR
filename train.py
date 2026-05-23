import os
import sys
import time
import warnings
from collections import Counter

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize

from config import LANGUAGE_INFO, MODELS_DIR, OUTPUT_DIR

warnings.filterwarnings('ignore')

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def extract_features(texts):
    print("\n" + "="*60)
    print("EKSTRAKCJA CECH")
    print("="*60)
    print(f"  Liczba tekstow do przetworzenia: {len(texts)}")
    
    print("  Ekstrakcja cech znakowych (2-6 n-gramy)...")
    start = time.time()
    char_vectorizer = TfidfVectorizer(
        analyzer='char',
        ngram_range=(2, 6),
        max_features=20000,
        sublinear_tf=True,
        lowercase=True,
        dtype=np.float32
    )
    X_char = char_vectorizer.fit_transform(texts)
    print(f"    Cechy znakowe: {X_char.shape[1]} (czas: {time.time()-start:.1f}s)")
    
    print("  Ekstrakcja cech slownych (1-3 n-gramy)...")
    start = time.time()
    word_vectorizer = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 3),
        max_features=6000,
        sublinear_tf=True,
        lowercase=True,
        min_df=2,
        dtype=np.float32
    )
    X_word = word_vectorizer.fit_transform(texts)
    print(f"    Cechy slowne: {X_word.shape[1]} (czas: {time.time()-start:.1f}s)")
    
    X_combined = hstack([X_char, X_word])
    print(f"    Lacznie cech: {X_combined.shape[1]}")
    
    return X_combined, (char_vectorizer, word_vectorizer)


def perform_eda(texts, labels, alphabets, output_dir='pliki'):
    print("\n" + "="*60)
    print("ANALIZA EKSPLORACYJNA DANYCH (EDA)")
    print("="*60)
    print(f"  Liczba tekstow: {len(texts)}")
    
    text_lengths = [len(t.split()) for t in texts]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Analiza Eksploracyjna Danych - Language Detection', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    axes[0, 0].hist(text_lengths, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(np.mean(text_lengths), color='red', linestyle='--', 
                        linewidth=2, label=f'Srednia: {np.mean(text_lengths):.0f}')
    axes[0, 0].axvline(np.median(text_lengths), color='green', linestyle='--', 
                        linewidth=2, label=f'Mediana: {np.median(text_lengths):.0f}')
    axes[0, 0].set_xlabel('Liczba slow', fontsize=10)
    axes[0, 0].set_ylabel('Liczba tekstow', fontsize=10)
    axes[0, 0].set_title('Rozklad dlugosci tekstow', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)
    
    lang_counts = Counter(labels)
    top_langs = lang_counts.most_common(20)
    lang_names = []
    counts = []
    colors_lang = []
    
    for lang_code, count in top_langs:
        lang_info = LANGUAGE_INFO.get(lang_code, {})
        lang_name = lang_info.get('name', lang_code)
        alphabet = lang_info.get('alphabet', 'unknown')
        lang_names.append(f"{lang_code} ({lang_name})")
        counts.append(count)
        
        if alphabet == 'cyrillic':
            colors_lang.append('#e74c3c')
        else:
            colors_lang.append('#3498db')
    
    bars = axes[0, 1].barh(range(len(lang_names)), counts, color=colors_lang)
    axes[0, 1].set_yticks(range(len(lang_names)))
    axes[0, 1].set_yticklabels(lang_names, fontsize=7)
    axes[0, 1].set_xlabel('Liczba probek', fontsize=10)
    axes[0, 1].set_title('Top 20 jezykow w zbiorze', fontsize=12, fontweight='bold')
    axes[0, 1].invert_yaxis()
    
    for bar, count in zip(bars, counts):
        axes[0, 1].text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
                        str(count), va='center', fontsize=7)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', label='Alfabet lacinski'),
        Patch(facecolor='#e74c3c', label='Alfabet cyrylica')
    ]
    axes[0, 1].legend(handles=legend_elements, loc='lower right', fontsize=8)
    
    family_counts = Counter()
    for label in labels:
        family = LANGUAGE_INFO.get(label, {}).get('family', 'Inne')
        family_counts[family] += 1
    
    families = list(family_counts.keys())
    family_values = list(family_counts.values())
    colors_family = plt.cm.Set3(np.linspace(0, 1, len(families)))
    
    wedges, texts, autotexts = axes[1, 0].pie(
        family_values, 
        labels=None, 
        autopct='%1.1f%%',
        colors=colors_family, 
        textprops={'fontsize': 8},
        pctdistance=0.85
    )
    
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontsize(9)
        autotext.set_fontweight('bold')
    
    axes[1, 0].set_title('Rozklad rodzin jezykowych', fontsize=12, fontweight='bold')
    axes[1, 0].legend(wedges, families, title="Rodziny jezykowe", 
                      loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=8)
    
    alpha_counts = Counter(alphabets)
    alpha_names = list(alpha_counts.keys())
    alpha_values = list(alpha_counts.values())
    
    alpha_colors = []
    for a in alpha_names:
        if a == 'latin':
            alpha_colors.append('#3498db')
        elif a == 'cyrillic':
            alpha_colors.append('#e74c3c')
        else:
            alpha_colors.append('#95a5a6')
    
    bars_alpha = axes[1, 1].bar(alpha_names, alpha_values, color=alpha_colors, edgecolor='black')
    axes[1, 1].set_ylabel('Liczba tekstow', fontsize=10)
    axes[1, 1].set_title('Rozklad typow alfabetu', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    total = sum(alpha_values)
    for bar, val in zip(bars_alpha, alpha_values):
        height = bar.get_height()
        percentage = (val / total) * 100
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 50,
                        f'{int(val)}\n({percentage:.1f}%)', 
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/eda_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Wykresy EDA zapisane w: {output_dir}/eda_analysis.png")
    
    return text_lengths, lang_counts, family_counts


def train_language_model(X, y):
    print("\n" + "="*60)
    print("TRENOWANIE MODELU JEZYKOWEGO")
    print("="*60)
    
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)
    n_classes = len(label_encoder.classes_)
    
    print(f"  Liczba klas (jezykow): {n_classes}")
    print(f"  Jezyki: {', '.join(label_encoder.classes_[:10])}...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    
    print(f"\n  Zbior treningowy: {X_train.shape[0]} probek")
    print(f"  Zbior testowy:     {X_test.shape[0]} probek")
    print(f"  Liczba cech:       {X_train.shape[1]}")
    
    print("\n  Trenowanie RandomForest...")
    start = time.time()
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=30,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced_subsample',
        verbose=0
    )
    
    model.fit(X_train, y_train)
    print(f"  Model wytrenowany! (czas: {time.time()-start:.1f}s)")
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print("\n  WYNIKI EWALUACJI:")
    print(f"  {'='*50}")
    print(f"  Accuracy:          {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision (wazone): {precision_weighted:.4f}")
    print(f"  Recall (wazone):    {recall_weighted:.4f}")
    print(f"  F1-score (wazone):  {f1_weighted:.4f}")
    
    report = classification_report(
        y_test, y_pred, 
        target_names=label_encoder.classes_,
        zero_division=0
    )
    
    with open(f'{OUTPUT_DIR}/classification_report.txt', 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("RAPORT KLASYFIKACJI - MODEL DETEKCJI JEZYKA\n")
        f.write("="*70 + "\n\n")
        f.write(f"Data utworzenia: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("OGOLNE METRYKI:\n")
        f.write(f"  Liczba klas:        {n_classes}\n")
        f.write(f"  Accuracy:           {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"  Precision (wazone): {precision_weighted:.4f}\n")
        f.write(f"  Recall (wazone):    {recall_weighted:.4f}\n")
        f.write(f"  F1-score (wazone):  {f1_weighted:.4f}\n")
        f.write(f"\n  Zbior treningowy:   {X_train.shape[0]} probek\n")
        f.write(f"  Zbior testowy:      {X_test.shape[0]} probek\n")
        f.write(f"  Liczba cech:        {X_train.shape[1]}\n\n")
        f.write("-"*70 + "\n")
        f.write("SZCZEGOLOWY RAPORT PER KLASA:\n")
        f.write("-"*70 + "\n\n")
        f.write(report)
    
    print(f"  Raport zapisany w: {OUTPUT_DIR}/classification_report.txt")
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(22, 20))
    sns.heatmap(
        cm, 
        cmap='Blues',
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
        annot=True, 
        fmt='d', 
        annot_kws={'size': 5},
        cbar_kws={'label': 'Liczba probek'},
        linewidths=0.3,
        linecolor='white'
    )
    plt.xlabel('Przewidywany jezyk', fontsize=14, fontweight='bold')
    plt.ylabel('Rzeczywisty jezyk', fontsize=14, fontweight='bold')
    plt.title('Macierz pomylek - Model detekcji jezyka', fontsize=16, fontweight='bold')
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/confusion_matrix_language.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Macierz pomylek zapisana w: {OUTPUT_DIR}/confusion_matrix_language.png")
    
    priority_langs = ['en', 'de', 'pl', 'ru', 'fr', 'es', 'it', 'uk', 'bg', 'sr', 'kk', 'ky']
    selected_langs = [lang for lang in priority_langs if lang in label_encoder.classes_]
    
    if len(selected_langs) < 4:
        selected_langs = list(label_encoder.classes_)[:8]
    
    y_test_bin = label_binarize(y_test, classes=range(n_classes))
    
    n_langs = len(selected_langs)
    n_cols = 4
    n_rows = (n_langs + n_cols - 1) // n_cols
    
    fig, axes_roc = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 5))
    
    if n_rows == 1:
        axes_roc = np.array([axes_roc])
    
    fig.suptitle('Krzywe ROC dla wybranych jezykow (One-vs-Rest) - powiekszony zakres', 
                 fontsize=18, fontweight='bold', y=1.02)
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(selected_langs)))
    
    for i, lang in enumerate(selected_langs):
        lang_idx = list(label_encoder.classes_).index(lang)
        
        fpr, tpr, _ = roc_curve(y_test_bin[:, lang_idx], y_proba[:, lang_idx])
        roc_auc = auc(fpr, tpr)
        
        lang_name = LANGUAGE_INFO.get(lang, {}).get('name', lang)
        alphabet = LANGUAGE_INFO.get(lang, {}).get('alphabet', 'latin')
        
        row = i // n_cols
        col = i % n_cols
        
        ax = axes_roc[row, col]
        
        line_color = '#e74c3c' if alphabet == 'cyrillic' else '#3498db'
        
        ax.plot(fpr, tpr, color=line_color, lw=3, label=f'AUC = {roc_auc:.4f}')
        ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.4)
        ax.fill_between(fpr, tpr, alpha=0.2, color=line_color)
        
        ax.set_xlim([0.0, 0.1])
        ax.set_ylim([0.9, 1.0])
        ax.set_xlabel('False Positive Rate', fontsize=10)
        ax.set_ylabel('True Positive Rate', fontsize=10)
        ax.set_title(f'{lang.upper()} ({lang_name})', fontsize=13, fontweight='bold')
        ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(labelsize=9)
    
    for i in range(n_langs, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes_roc[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/roc_curves.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Krzywe ROC zapisane w: {OUTPUT_DIR}/roc_curves.png")
    
    plt.figure(figsize=(18, 13))
    
    for i, lang in enumerate(selected_langs[:12]):
        lang_idx = list(label_encoder.classes_).index(lang)
        
        fpr, tpr, _ = roc_curve(y_test_bin[:, lang_idx], y_proba[:, lang_idx])
        roc_auc = auc(fpr, tpr)
        
        lang_name = LANGUAGE_INFO.get(lang, {}).get('name', lang)
        alphabet = LANGUAGE_INFO.get(lang, {}).get('alphabet', 'latin')
        
        linestyle = '-' if alphabet == 'latin' else '--'
        linewidth = 3
        
        plt.plot(fpr, tpr, color=colors[i], lw=linewidth, linestyle=linestyle,
                 label=f'{lang.upper()} ({lang_name}) - AUC = {roc_auc:.4f}')
    
    plt.plot([0, 1], [0, 1], 'k:', lw=2, label='Klasyfikator losowy (AUC = 0.500)', alpha=0.5)
    
    plt.xlim([0.0, 0.1])
    plt.ylim([0.9, 1.0])
    plt.xlabel('False Positive Rate (1 - Specyficznosc)', fontsize=14, fontweight='bold')
    plt.ylabel('True Positive Rate (Czulosc)', fontsize=14, fontweight='bold')
    plt.title('Porownanie krzywych ROC dla wybranych jezykow (powiekszony zakres)\n(linia ciagla = lacina, przerywana = cyrylica)', 
              fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=9, framealpha=0.9, ncol=2)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/roc_curves_combined.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Polaczone krzywe ROC zapisane w: {OUTPUT_DIR}/roc_curves_combined.png")
    
    joblib.dump(model, f'{MODELS_DIR}/language_model.pkl')
    joblib.dump(label_encoder, f'{MODELS_DIR}/language_encoder.pkl')
    print(f"  Model zapisany w: {MODELS_DIR}/")
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision_weighted,
        'recall': recall_weighted,
        'f1': f1_weighted
    }
    
    return model, label_encoder, metrics


def train_alphabet_model(X, y):
    print("\n" + "="*60)
    print("TRENOWANIE MODELU ALFABETU")
    print("="*60)
    
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)
    
    print(f"  Klasy alfabetu: {list(label_encoder.classes_)}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    
    print(f"  Zbior treningowy: {X_train.shape[0]} probek")
    print(f"  Zbior testowy:     {X_test.shape[0]} probek")
    
    start = time.time()
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=25,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print(f"  Model wytrenowany! (czas: {time.time()-start:.1f}s)")
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print("\n  WYNIKI:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-score:  {f1:.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        cmap='Reds',
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
        annot=True, 
        fmt='d',
        annot_kws={'size': 14, 'fontweight': 'bold'},
        cbar_kws={'label': 'Liczba probek'},
        linewidths=2,
        linecolor='white'
    )
    plt.xlabel('Przewidywany alfabet', fontsize=14, fontweight='bold')
    plt.ylabel('Rzeczywisty alfabet', fontsize=14, fontweight='bold')
    plt.title('Macierz pomylek - Model alfabetu', fontsize=16, fontweight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/confusion_matrix_alphabet.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    joblib.dump(model, f'{MODELS_DIR}/alphabet_model.pkl')
    joblib.dump(label_encoder, f'{MODELS_DIR}/alphabet_encoder.pkl')
    print(f"  Model alfabetu zapisany w: {MODELS_DIR}/")
    
    return model, label_encoder, accuracy


def train_complete_model(texts, labels, alphabets):
    print("="*70)
    print("  PELNY TRENING SYSTEMU DETEKCJI JEZYKA")
    print("="*70)
    
    start_time = pd.Timestamp.now()
    
    perform_eda(texts, labels, alphabets)
    
    X, vectorizers = extract_features(texts)
    
    lang_model, lang_encoder, lang_metrics = train_language_model(X, labels)
    
    alpha_model, alpha_encoder, alpha_acc = train_alphabet_model(X, alphabets)
    
    joblib.dump(vectorizers, f'{MODELS_DIR}/vectorizers.pkl')
    joblib.dump(LANGUAGE_INFO, f'{MODELS_DIR}/language_info.pkl')
    
    end_time = pd.Timestamp.now()
    training_time = (end_time - start_time).total_seconds()
    
    print("\n" + "="*70)
    print("  PODSUMOWANIE TRENINGU")
    print("="*70)
    print(f"\n  Czas treningu: {training_time:.0f} sekund ({training_time/60:.1f} minut)")
    print("\n  MODEL JEZYKOWY:")
    print(f"    Liczba jezykow: {len(lang_encoder.classes_)}")
    print(f"    Accuracy:       {lang_metrics['accuracy']:.2%}")
    print(f"    Precision:      {lang_metrics['precision']:.2%}")
    print(f"    Recall:         {lang_metrics['recall']:.2%}")
    print(f"    F1-score:       {lang_metrics['f1']:.2%}")
    
    print("\n  MODEL ALFABETU:")
    print(f"    Klasy:          {list(alpha_encoder.classes_)}")
    print(f"    Accuracy:       {alpha_acc:.2%}")
    
    print("\n  PLIKI WYNIKOWE:")
    print(f"    {OUTPUT_DIR}/eda_analysis.png")
    print(f"    {OUTPUT_DIR}/confusion_matrix_language.png")
    print(f"    {OUTPUT_DIR}/confusion_matrix_alphabet.png")
    print(f"    {OUTPUT_DIR}/roc_curves.png")
    print(f"    {OUTPUT_DIR}/roc_curves_combined.png")
    print(f"    {OUTPUT_DIR}/classification_report.txt")
    print(f"    {MODELS_DIR}/language_model.pkl")
    print(f"    {MODELS_DIR}/language_encoder.pkl")
    print(f"    {MODELS_DIR}/alphabet_model.pkl")
    print(f"    {MODELS_DIR}/alphabet_encoder.pkl")
    print(f"    {MODELS_DIR}/vectorizers.pkl")
    print(f"    {MODELS_DIR}/language_info.pkl")
    
    print("\n  >> Trening zakonczony pomyslnie!")
    
    return {
        'language_accuracy': lang_metrics['accuracy'],
        'language_precision': lang_metrics['precision'],
        'language_recall': lang_metrics['recall'],
        'language_f1': lang_metrics['f1'],
        'alphabet_accuracy': alpha_acc,
        'training_time_seconds': training_time
    }


if __name__ == "__main__":
    print("Modul treningowy gotowy.")
    print("Uzyj funkcji train_complete_model(texts, labels, alphabets)")
    print("lub uruchom main_train.py")