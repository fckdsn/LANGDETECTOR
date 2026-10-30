from __future__ import annotations

import joblib
import path_setup  # noqa: F401
import pytest
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

from predict import LanguageDetector
from preprocess import WORD_TOKEN_PATTERN

# Po jednej probce na jezyk - wystarczy, bo probki dobrano tak, by rozniły sie
# pismem albo wyraznym slownictwem. Chodzi o sprawdzenie potoku, nie jakosci.
TRAIN_SAMPLES = [
    ("en", "the english language newspaper about london america and united kingdom today"),
    ("de", "die deutsche sprache zeitung ueber berlin deutschland und oesterreich heute morgen"),
    ("ru", "русский язык газета москва россия красная площадь новости сегодня утром"),
    ("zh", "中文报纸关于北京上海和中国的新闻今天天气很好机器学习算法分析数据"),
    ("ja", "日本語の新聞は東京と大阪について報道します。機械学習アルゴリズムがデータを分析します。"),
    ("ko", "한국어 신문은 서울과 부산에 대해 보도합니다. 기계 학습 알고리즘이 데이터를 분석합니다."),
    ("hi", "हिंदी भाषा का समाचार पत्र दिल्ली और भारत के बारे में आज की खबर देता है"),
    ("ar", "الصحيفة العربية تنشر أخبار القاهرة وبغداد والرياض اليوم عن التعلم الآلي"),
    ("th", "หนังสือพิมพ์ภาษาไทยรายงานข่าวกรุงเทพและเชียงใหม่วันนี้เกี่ยวกับการเรียนรู้ของเครื่อง"),
]


@pytest.fixture
def minimal_models(tmp_path, monkeypatch):
    """Trenuje miniaturowe modele wielopismowe i podstawia je detektorowi."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    labels = [code for code, _ in TRAIN_SAMPLES]
    texts = [text for _, text in TRAIN_SAMPLES]
    scripts = ["latin", "latin", "cyrillic", "han", "japanese",
               "hangul", "devanagari", "arabic", "thai"]

    char_vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 4), max_features=3000)
    word_vec = TfidfVectorizer(analyzer="word", token_pattern=WORD_TOKEN_PATTERN,
                               ngram_range=(1, 1), max_features=1500)
    X = csr_matrix(hstack([char_vec.fit_transform(texts), word_vec.fit_transform(texts)]))

    encoder = LabelEncoder()
    model = MultinomialNB(alpha=0.01).fit(X, encoder.fit_transform(labels))

    script_encoder = LabelEncoder()
    script_model = MultinomialNB(alpha=0.01).fit(X, script_encoder.fit_transform(scripts))

    joblib.dump(model, models_dir / "language_model.pkl")
    joblib.dump(encoder, models_dir / "language_encoder.pkl")
    joblib.dump((char_vec, word_vec), models_dir / "vectorizers.pkl")
    joblib.dump(script_model, models_dir / "script_model.pkl")
    joblib.dump(script_encoder, models_dir / "script_encoder.pkl")
    joblib.dump({'algorithm': 'naive_bayes', 'label': 'Naive Bayes (Multinomial)'},
                models_dir / "model_metadata.pkl")

    monkeypatch.setattr("predict.MODELS_DIR", str(models_dir))

    return LanguageDetector()
