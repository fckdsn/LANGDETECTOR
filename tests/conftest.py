from __future__ import annotations

import joblib
import path_setup  # noqa: F401
import pytest
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from predict import LanguageDetector


@pytest.fixture
def minimal_models(tmp_path, monkeypatch):
    """Train tiny models and patch LanguageDetector to load them."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    texts = [
        "the english language newspaper about london america and united kingdom today",
        "die deutsche sprache zeitung ueber berlin deutschland und oesterreich heute morgen",
        "русский язык газета москва россия красная площадь новости сегодня утром",
    ]
    labels = ["en", "de", "ru"]

    char_vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=500)
    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=200)
    X = hstack([char_vec.fit_transform(texts), word_vec.fit_transform(texts)])

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    joblib.dump(model, models_dir / "language_model.pkl")
    joblib.dump(encoder, models_dir / "language_encoder.pkl")
    joblib.dump((char_vec, word_vec), models_dir / "vectorizers.pkl")

    monkeypatch.setattr("predict.MODELS_DIR", str(models_dir))

    return LanguageDetector()
