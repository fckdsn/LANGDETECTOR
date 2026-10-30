import path_setup  # noqa: F401
import pytest

from predict import LanguageDetector, ModelsNotFoundError, read_test_cases

SHORT_TEXT = "too short"
EN_TEXT = (
    "The english language newspaper about london america and the united kingdom "
    "published international news today for english readers worldwide."
)
DE_TEXT = (
    "Die deutsche sprache zeitung ueber berlin deutschland und oesterreich "
    "berichtet heute morgen ueber die kuenstliche intelligenz und maschinelles lernen."
)
ZH_TEXT = "人工智能的快速发展在近年来改变了许多行业和技术领域的机器学习算法"
JA_TEXT = "機械学習アルゴリズムは膨大なデータを分析することができます。東京の新聞が報道しました。"
TH_TEXT = "หนังสือพิมพ์ภาษาไทยรายงานข่าวกรุงเทพวันนี้เกี่ยวกับการเรียนรู้ของเครื่อง"
AR_TEXT = "الصحيفة العربية تنشر أخبار القاهرة وبغداد اليوم عن خوارزميات التعلم الآلي"


# --- Kontrakt predict() -----------------------------------------------------

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
        assert {'language', 'name', 'family', 'script', 'confidence'} <= set(item)


def test_predict_filters_low_confidence(minimal_models):
    _, _, preds = minimal_models.predict(EN_TEXT)
    assert all(p["confidence"] > 0.02 for p in preds)


def test_predict_german_beats_english_on_german_text(minimal_models):
    lang, _, _ = minimal_models.predict(DE_TEXT)
    assert lang == "de"


def test_predict_respects_top_k(minimal_models):
    _, _, preds = minimal_models.predict(EN_TEXT, top_k=3)
    assert len(preds) <= 3


# --- Pisma zlozone ----------------------------------------------------------

@pytest.mark.parametrize('text,expected', [
    (ZH_TEXT, 'zh'),
    (JA_TEXT, 'ja'),
    (TH_TEXT, 'th'),
    (AR_TEXT, 'ar'),
])
def test_predict_handles_complex_scripts(minimal_models, text, expected):
    lang, conf, _ = minimal_models.predict(text)
    assert lang == expected
    assert conf > 0


def test_short_cjk_text_is_accepted(minimal_models):
    """Kilkanascie ideogramow niesie tyle informacji, co dluzsze zdanie lacinskie."""
    lang, _, preds = minimal_models.predict("人工智能的快速发展改变行业")
    assert lang is not None
    assert preds


def test_short_latin_text_is_rejected(minimal_models):
    lang, _, _ = minimal_models.predict("short one")
    assert lang is None


# --- Rozpoznawanie pisma ----------------------------------------------------

def test_predict_script_uses_model_when_available(minimal_models):
    result = minimal_models.predict_script(ZH_TEXT)
    assert result['script'] == 'han'
    assert result['source'] == 'model ML'
    assert 0 <= result['confidence'] <= 1


def test_predict_script_falls_back_to_heuristic(minimal_models):
    minimal_models.script_model = None
    minimal_models.script_encoder = None
    result = minimal_models.predict_script(TH_TEXT)
    assert result['script'] == 'thai'
    assert 'heurystyka' in result['source']


def test_predict_script_on_empty_text(minimal_models):
    result = minimal_models.predict_script("!!! 123 ???")
    assert result['script'] == 'unknown'


def test_predict_script_reports_profile(minimal_models):
    result = minimal_models.predict_script("hello world привет мир")
    assert set(result['profile']) == {'latin', 'cyrillic'}


# --- analyze() --------------------------------------------------------------

def test_analyze_returns_full_structure(minimal_models):
    analysis = minimal_models.analyze(EN_TEXT)
    assert analysis['language'] == 'en'
    assert analysis['predictions']
    assert analysis['script']['script']
    assert analysis['algorithm']
    assert analysis['text_length'] == len(EN_TEXT)


def test_detector_exposes_metadata(minimal_models):
    assert minimal_models.algorithm == 'Naive Bayes (Multinomial)'
    assert 'en' in minimal_models.languages


# --- Blad braku modeli ------------------------------------------------------

def test_missing_models_raise_clear_error(tmp_path):
    with pytest.raises(ModelsNotFoundError):
        LanguageDetector(models_dir=str(tmp_path))


# --- Zbior ewaluacyjny ------------------------------------------------------

def test_read_test_cases_parses_project_file():
    cases = read_test_cases()
    assert len(cases) > 100
    codes = [code for code, _ in cases]
    for expected in ('en', 'zh', 'ja', 'ko', 'hi', 'ar', 'th', 'he', 'el', 'ta'):
        assert expected in codes


def test_read_test_cases_missing_file_returns_empty():
    assert read_test_cases('nie_ma_takiego_pliku.txt') == []
