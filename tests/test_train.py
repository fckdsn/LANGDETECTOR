import numpy as np
import path_setup  # noqa: F401
from scipy.sparse import csr_matrix

from train import (
    _drop_rare_classes,
    _select_roc_languages,
    _stratified_subsample,
    evaluate_predictions,
    extract_features,
)

MULTISCRIPT_TEXTS = [
    "machine learning algorithms analyse large amounts of data every single day",
    "algorytmy uczenia maszynowego analizuja ogromne ilosci danych kazdego dnia",
    "алгоритмы машинного обучения анализируют огромные объемы данных каждый день",
    "机器学习算法每天分析大量的数据并识别人类可能忽略的模式和趋势",
    "機械学習アルゴリズムは毎日膨大なデータを分析してパターンを特定します",
    "기계 학습 알고리즘은 매일 방대한 양의 데이터를 분석하고 패턴을 식별합니다",
    "मशीन लर्निंग एल्गोरिदम हर दिन बड़ी मात्रा में डेटा का विश्लेषण करते हैं",
    "خوارزميات التعلم الآلي تحلل كميات هائلة من البيانات كل يوم وتحدد الأنماط",
    "อัลกอริทึมการเรียนรู้ของเครื่องวิเคราะห์ข้อมูลจำนวนมากทุกวันและระบุรูปแบบ",
]
MULTISCRIPT_LABELS = ['en', 'pl', 'ru', 'zh', 'ja', 'ko', 'hi', 'ar', 'th']


# --- Ekstrakcja cech --------------------------------------------------------

def test_extract_features_returns_csr_matrix():
    X, vectorizers = extract_features(MULTISCRIPT_TEXTS, MULTISCRIPT_LABELS)
    assert isinstance(X, csr_matrix)
    assert X.shape[0] == len(MULTISCRIPT_TEXTS)
    assert X.shape[1] > 0
    assert len(vectorizers) == 2


def test_extract_features_gives_every_script_nonzero_features():
    """Kazdy tekst musi miec niepusta reprezentacje, takze w pismach CJK."""
    X, _ = extract_features(MULTISCRIPT_TEXTS, MULTISCRIPT_LABELS)
    nonzero_per_row = np.diff(X.indptr)
    assert (nonzero_per_row > 0).all()


def test_extract_features_word_vectorizer_tokenizes_cjk():
    """Bez wzorca dla pism bez spacji chinskie zdanie bylo jednym tokenem."""
    _, (_, word_vectorizer) = extract_features(MULTISCRIPT_TEXTS, MULTISCRIPT_LABELS)
    vocabulary = word_vectorizer.vocabulary_
    assert any(len(token) == 1 and '\u4e00' <= token <= '\u9fff' for token in vocabulary)


def test_extract_features_separates_scripts():
    """Teksty w roznych pismach nie moga dzielic cech znakowych."""
    X, _ = extract_features(MULTISCRIPT_TEXTS, MULTISCRIPT_LABELS)
    chinese = X[3].toarray()[0]
    thai = X[8].toarray()[0]
    assert float(np.dot(chinese, thai)) == 0.0


# --- Podprobkowanie --------------------------------------------------------

def test_stratified_subsample_returns_all_when_under_limit():
    result = _stratified_subsample(MULTISCRIPT_TEXTS, MULTISCRIPT_LABELS, 100)
    assert len(result) == len(MULTISCRIPT_TEXTS)


def test_stratified_subsample_limits_size():
    texts = [f"tekst numer {i}" for i in range(200)]
    labels = ['a'] * 100 + ['b'] * 100
    result = _stratified_subsample(texts, labels, 40)
    assert len(result) <= 40


# --- Miary jakosci ---------------------------------------------------------

def test_evaluate_predictions_contains_all_required_metrics():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])
    proba = np.eye(3)[y_pred]

    metrics = evaluate_predictions(y_true, y_pred, proba, n_classes=3)

    for key in ('accuracy', 'balanced_accuracy', 'precision_weighted', 'recall_weighted',
                'f1_weighted', 'precision_macro', 'recall_macro', 'f1_macro',
                'top3_accuracy'):
        assert key in metrics
    assert metrics['accuracy'] == 1.0
    assert metrics['f1_weighted'] == 1.0


def test_evaluate_predictions_penalises_errors():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])
    metrics = evaluate_predictions(y_true, y_pred)
    assert metrics['accuracy'] == 0.5
    assert metrics['f1_macro'] < 1.0


def test_evaluate_predictions_macro_flags_weak_rare_class():
    """Miara makro musi silniej karac blad na klasie rzadkiej niz miara wazona."""
    y_true = np.array([0] * 18 + [1, 1])
    y_pred = np.array([0] * 18 + [0, 0])
    metrics = evaluate_predictions(y_true, y_pred)
    assert metrics['f1_macro'] < metrics['f1_weighted']


# --- Filtrowanie klas rzadkich ---------------------------------------------

def test_drop_rare_classes_removes_underpopulated_labels():
    X = csr_matrix(np.arange(20).reshape(10, 2).astype(float))
    y = ['a'] * 6 + ['b'] * 3 + ['c']

    X_filtered, y_filtered, dropped = _drop_rare_classes(X, y, min_count=5)

    assert dropped == ['b', 'c']
    assert set(y_filtered) == {'a'}
    assert X_filtered.shape[0] == 6


def test_drop_rare_classes_keeps_everything_when_balanced():
    X = csr_matrix(np.ones((10, 2)))
    y = ['a'] * 5 + ['b'] * 5
    _, y_filtered, dropped = _drop_rare_classes(X, y, min_count=5)
    assert dropped == []
    assert len(y_filtered) == 10


# --- Wybor jezykow na krzywe ROC -------------------------------------------

def test_select_roc_languages_prefers_script_diversity():
    classes = ['en', 'de', 'nl', 'sv', 'zh', 'ja', 'ko', 'th', 'ar', 'hi']
    selected = _select_roc_languages(classes, limit=6)
    assert len(selected) == 6
    assert 'en' in selected
    assert 'zh' in selected
    assert 'ja' in selected


def test_select_roc_languages_respects_limit_and_membership():
    classes = ['pl', 'cs', 'sk']
    selected = _select_roc_languages(classes, limit=10)
    assert set(selected) <= set(classes)
    assert len(selected) == 3
