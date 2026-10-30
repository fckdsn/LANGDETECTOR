import re

import path_setup  # noqa: F401

from preprocess import (
    WORD_TOKEN_PATTERN,
    clean_text,
    detect_alphabet,
    detect_script,
    script_profile,
    script_share,
)

SAMPLES = {
    'latin': "hello world from london and new york city",
    'cyrillic': "привет мир из москвы и санкт петербурга",
    'greek': "η ταχεία εξέλιξη της τεχνητής νοημοσύνης στην ελλάδα",
    'han': "人工智能的快速发展在近年来改变了许多行业和技术领域",
    'japanese': "機械学習アルゴリズムは膨大なデータを分析することができます",
    'hangul': "기계 학습 알고리즘은 방대한 양의 데이터를 분석할 수 있습니다",
    'devanagari': "मशीन लर्निंग एल्गोरिदम बड़ी मात्रा में डेटा का विश्लेषण करते हैं",
    'bengali': "মেশিন লার্নিং অ্যালগরিদম বিপুল পরিমাণ তথ্য বিশ্লেষণ করতে পারে",
    'tamil': "இயந்திர கற்றல் வழிமுறைகள் தரவுகளை பகுப்பாய்வு செய்யும்",
    'arabic': "خوارزميات التعلم الآلي تحلل كميات هائلة من البيانات",
    'hebrew': "אלגוריתמים של למידת מכונה מנתחים כמויות עצומות של נתונים",
    'thai': "อัลกอริทึมการเรียนรู้ของเครื่องสามารถวิเคราะห์ข้อมูลจำนวนมาก",
    'georgian': "მანქანური სწავლების ალგორითმები აანალიზებენ მონაცემებს",
    'armenian': "մեքենայական ուսուցման ալգորիթմները վերլուծում են տվյալները",
    'ethiopic': "የማሽን መማር አልጎሪዝሞች መረጃን መተንተን ይችላሉ",
    'khmer': "អាល់ហ្គោរីធុមនៃការរៀនម៉ាស៊ីនអាចវិភាគទិន្នន័យ",
    'myanmar': "စက်သင်ယူမှု အယ်လဂိုရစ်သမ်များသည် ဒေတာကို ခွဲခြမ်းစိတ်ဖြာနိုင်သည်",
    'sinhala': "යන්ත්‍ර ඉගෙනුම් ඇල්ගොරිතම දත්ත විශ්ලේෂණය කරයි",
}


# --- clean_text --------------------------------------------------------------

def test_clean_text_empty():
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_clean_text_lowercase_and_strip():
    assert clean_text("  HELLO World  ") == "hello world"


def test_clean_text_removes_urls_and_emails():
    text = "visit https://example.com or mail user@domain.com for info today"
    cleaned = clean_text(text)
    assert "https" not in cleaned
    assert "example.com" not in cleaned
    assert "user@domain.com" not in cleaned
    assert "visit" in cleaned
    assert "info" in cleaned


def test_clean_text_removes_punctuation():
    cleaned = clean_text("Hello, world! #test — ok.")
    assert "," not in cleaned
    assert "!" not in cleaned
    assert cleaned == "hello world test ok"


def test_clean_text_drops_single_char_tokens():
    assert clean_text("a I am ok now") == "am ok now"


def test_clean_text_keep_digits_false():
    cleaned = clean_text("room 42 is open", keep_digits=False)
    assert "42" not in cleaned
    assert "room" in cleaned
    assert "open" in cleaned


def test_clean_text_preserves_cyrillic():
    cleaned = clean_text("Привет мир сегодня хорошо")
    assert "привет" in cleaned
    assert "мир" in cleaned


def test_clean_text_preserves_every_supported_script():
    """Czyszczenie nie moze usuwac liter zadnego obslugiwanego pisma."""
    for script, sample in SAMPLES.items():
        cleaned = clean_text(sample)
        assert cleaned, f"tekst w pismie {script} zostal wyczyszczony do pustego"
        # Przynajmniej 80% znakow niebedacych spacja musi przetrwac.
        original_chars = len([c for c in sample if not c.isspace()])
        kept_chars = len([c for c in cleaned if not c.isspace()])
        assert kept_chars >= 0.8 * original_chars, f"pismo {script}: utracono znaki"


def test_clean_text_preserves_combining_marks():
    """Matry i znaki tonu to osobne punkty kodowe - musza zostac zachowane."""
    hindi = "कृत्रिम बुद्धिमत्ता"
    thai = "ปัญญาประดิษฐ์"
    assert clean_text(hindi) == hindi
    assert clean_text(thai) == thai


def test_clean_text_keeps_cjk_without_spaces():
    text = "人工智能的快速发展"
    assert clean_text(text) == text


# --- detect_script ----------------------------------------------------------

def test_detect_script_recognizes_each_sample():
    for expected, sample in SAMPLES.items():
        assert detect_script(sample) == expected, f"blad dla pisma {expected}"


def test_detect_alphabet_latin():
    assert detect_alphabet("hello world from london") == "latin"


def test_detect_alphabet_cyrillic():
    assert detect_alphabet("привет мир из москвы") == "cyrillic"


def test_detect_alphabet_mixed_or_equal():
    assert detect_alphabet("hello привет") in ("mixed", "latin", "cyrillic")


def test_detect_script_empty_text():
    assert detect_script("") == "unknown"
    assert detect_script("12345 !!! ???") == "unknown"


def test_detect_script_mixed_returns_mixed():
    # Rowny udzial dwoch pism - zadne nie osiaga progu dominacji.
    assert detect_script("hello world привет мир", threshold=0.6) == "mixed"


def test_japanese_not_confused_with_chinese():
    assert detect_script(SAMPLES['japanese']) == "japanese"
    assert detect_script(SAMPLES['han']) == "han"


def test_stray_katakana_keeps_text_chinese():
    """Nazwa wlasna zapisana katakana nie moze przeklasyfikowac tekstu na japonski."""
    chinese_with_katakana = SAMPLES['han'] * 4 + "ト"
    assert detect_script(chinese_with_katakana) == "han"


# --- script_profile / script_share ------------------------------------------

def test_script_profile_sums_to_one():
    profile = script_profile("hello мир 中文")
    assert abs(sum(profile.values()) - 1.0) < 1e-9
    assert set(profile) == {'latin', 'cyrillic', 'han'}


def test_script_profile_ignores_digits_and_punctuation():
    assert script_profile("12345 --- ???") == {}


def test_script_share():
    assert script_share(SAMPLES['thai'], 'thai') == 1.0
    assert script_share(SAMPLES['thai'], 'latin') == 0.0


# --- tokenizacja ------------------------------------------------------------

def test_word_token_pattern_splits_scriptio_continua():
    """Chinskie zdanie musi rozpasc sie na tokeny, a nie zostac jednym."""
    tokens = re.findall(WORD_TOKEN_PATTERN, "人工智能")
    assert tokens == ["人", "工", "智", "能"]


def test_word_token_pattern_keeps_latin_words_whole():
    tokens = re.findall(WORD_TOKEN_PATTERN, "machine learning algorithms")
    assert tokens == ["machine", "learning", "algorithms"]


def test_word_token_pattern_handles_mixed_text():
    tokens = re.findall(WORD_TOKEN_PATTERN, "data 数据 analysis")
    assert "data" in tokens
    assert "数" in tokens
    assert "analysis" in tokens
