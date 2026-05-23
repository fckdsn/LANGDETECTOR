import path_setup  # noqa: F401

from preprocess import clean_text, detect_alphabet

EN_SAMPLE = (
    "The rapid advancement of artificial intelligence has transformed many industries "
    "in recent years worldwide."
)


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


def test_detect_alphabet_latin():
    assert detect_alphabet("hello world from london") == "latin"


def test_detect_alphabet_cyrillic():
    assert detect_alphabet("привет мир из москвы") == "cyrillic"


def test_detect_alphabet_mixed_or_equal():
    assert detect_alphabet("hello привет") in ("mixed", "latin", "cyrillic")
