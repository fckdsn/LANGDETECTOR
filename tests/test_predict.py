import path_setup  # noqa: F401

from predict import LanguageDetector

SHORT_TEXT = "too short"
EN_TEXT = (
    "The english language newspaper about london america and the united kingdom "
    "published international news today for english readers worldwide."
)
DE_TEXT = (
    "Die rasante Entwicklung der kuenstlichen Intelligenz hat in den letzten Jahren "
    "viele Branchen grundlegend veraendert. Maschinenlernalgorithmen analysieren Daten."
)


def test_predict_returns_none_for_short_text():
    detector = LanguageDetector.__new__(LanguageDetector)
    lang, conf, preds = detector.predict(SHORT_TEXT)
    assert lang is None
    assert conf == 0
    assert preds == []


def test_predict_returns_top_language(minimal_models):
    lang, conf, preds = minimal_models.predict(EN_TEXT)
    assert str(lang) == "en"
    assert conf > 0
    assert len(preds) >= 1
    assert preds[0]["language"] == "en"
    assert "name" in preds[0]
    assert preds[0]["confidence"] > 0.02


def test_predict_top_list_structure(minimal_models):
    _, _, preds = minimal_models.predict(DE_TEXT)
    assert len(preds) >= 1
    for item in preds:
        assert "language" in item
        assert "confidence" in item
        assert item["confidence"] > 0.02


def test_predict_filters_low_confidence(minimal_models):
    _, _, preds = minimal_models.predict(EN_TEXT)
    assert all(p["confidence"] > 0.02 for p in preds)


def test_predict_german_beats_english_on_german_text(minimal_models):
    lang, _, _ = minimal_models.predict(DE_TEXT)
    assert lang == "de"
