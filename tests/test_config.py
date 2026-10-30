import path_setup  # noqa: F401

from config import (
    DENSE_SCRIPTS,
    LANGUAGE_INFO,
    LANGUAGES,
    MIN_TEXT_LENGTH,
    MIN_TEXT_LENGTH_DENSE,
    SCRIPT_INFO,
    is_dense_script,
    language_family,
    language_name,
    language_script,
    languages_by_script,
    min_text_length,
    script_name,
)


def test_language_pool_reaches_target_size():
    """Zakres pracy wymaga puli okolo 100 jezykow."""
    assert len(LANGUAGES) >= 100


def test_languages_match_language_info():
    assert LANGUAGES == list(LANGUAGE_INFO)


def test_every_language_has_complete_metadata():
    for code, info in LANGUAGE_INFO.items():
        assert info['name'], f"brak nazwy dla {code}"
        assert info['family'], f"brak rodziny dla {code}"
        assert info['script'] in SCRIPT_INFO, f"nieznane pismo dla {code}: {info['script']}"


def test_alphabet_key_kept_for_backward_compatibility():
    for code, info in LANGUAGE_INFO.items():
        assert info['alphabet'] == info['script'], code


def test_complex_scripts_are_represented():
    """Projekt musi obslugiwac pisma logograficzne, indyjskie i arabskie."""
    grouped = languages_by_script()
    for script in ('han', 'japanese', 'hangul', 'devanagari', 'arabic', 'thai',
                   'hebrew', 'bengali', 'tamil', 'telugu', 'kannada', 'malayalam',
                   'georgian', 'armenian', 'ethiopic', 'khmer', 'myanmar', 'tibetan'):
        assert grouped.get(script), f"brak jezyka dla pisma {script}"


def test_expected_languages_present():
    for code in ('zh', 'ja', 'ko', 'hi', 'ar', 'th', 'he', 'el', 'ta', 'bn', 'ka', 'am'):
        assert code in LANGUAGES


def test_script_count_is_substantial():
    scripts = set(languages_by_script())
    assert len(scripts) >= 20


def test_dense_scripts_have_lower_length_threshold():
    assert 'han' in DENSE_SCRIPTS
    assert 'japanese' in DENSE_SCRIPTS
    assert 'thai' in DENSE_SCRIPTS
    assert 'latin' not in DENSE_SCRIPTS

    assert min_text_length('han') == MIN_TEXT_LENGTH_DENSE
    assert min_text_length('latin') == MIN_TEXT_LENGTH
    assert min_text_length('han') < min_text_length('latin')


def test_training_thresholds_are_stricter_than_inference():
    assert min_text_length('latin', training=True) > min_text_length('latin')
    assert min_text_length('han', training=True) > min_text_length('han')


def test_is_dense_script():
    assert is_dense_script('japanese')
    assert not is_dense_script('cyrillic')


def test_lookup_helpers():
    assert language_name('ja') == 'Japanese'
    assert language_script('ja') == 'japanese'
    assert language_family('ja') == 'Japonic'
    assert script_name('han').startswith('Han')


def test_lookup_helpers_fall_back_for_unknown_code():
    assert language_name('xx') == 'xx'
    assert language_script('xx') == 'unknown'
    assert language_family('xx') == 'unknown'
