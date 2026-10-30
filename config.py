import os

HF_TOKEN = os.environ.get("HF_TOKEN", "HF_TOKEN")
os.environ["HF_TOKEN"] = HF_TOKEN

OUTPUT_DIR = 'pliki'
MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')
REPORTS_DIR = os.path.join(OUTPUT_DIR, 'reports')

# --- Parametry zbioru danych i ekstrakcji cech -------------------------------
# Wszystkie mozna nadpisac zmienna srodowiskowa, bez zmiany kodu.
SAMPLES_PER_LANGUAGE = int(os.environ.get("LD_SAMPLES", 900))
MAX_CHAR_FEATURES = int(os.environ.get("LD_CHAR_FEATURES", 50000))
MAX_WORD_FEATURES = int(os.environ.get("LD_WORD_FEATURES", 15000))
CHAR_NGRAM_RANGE = (1, 4)
WORD_NGRAM_RANGE = (1, 2)
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Do rozpoznania jezyka wystarcza kilkaset znakow, a skracanie probek trzyma
# slownik n-gramow w rozsadnych granicach pamieciowych.
MAX_TEXT_CHARS = int(os.environ.get("LD_MAX_TEXT_CHARS", 1200))
# Slownik n-gramow budujemy na warstwowej podprobce korpusu, a nastepnie
# transformujemy caly zbior. Peak pamieci przy ~100 jezykach spada kilkukrotnie,
# a dobor cech pozostaje reprezentatywny.
VECTORIZER_FIT_SAMPLES = int(os.environ.get("LD_VECTORIZER_FIT_SAMPLES", 25000))

# Minimalna dlugosc tekstu przyjmowana do klasyfikacji. Pisma bez odstepow
# miedzy wyrazami (chinski, japonski, tajski) niosa duzo wiecej informacji na
# znak, wiec maja nizszy prog.
MIN_TEXT_LENGTH = 30
MIN_TEXT_LENGTH_DENSE = 8
MIN_TRAIN_TEXT_LENGTH = 50
MIN_TRAIN_TEXT_LENGTH_DENSE = 20

# --- Pisma (alfabety / sylabariusze / pisma logograficzne) -------------------
# 'dense' oznacza pismo bez spacji miedzy wyrazami albo o bardzo duzym
# repertuarze znakow - takie teksty wymagaja tokenizacji znakowej.
SCRIPT_INFO = {
    'latin':      {'name': 'Lacinski', 'dense': False, 'type': 'alfabet'},
    'cyrillic':   {'name': 'Cyrylica', 'dense': False, 'type': 'alfabet'},
    'greek':      {'name': 'Grecki', 'dense': False, 'type': 'alfabet'},
    'armenian':   {'name': 'Ormianski', 'dense': False, 'type': 'alfabet'},
    'georgian':   {'name': 'Gruzinski', 'dense': False, 'type': 'alfabet'},
    'hebrew':     {'name': 'Hebrajski', 'dense': False, 'type': 'abdzad'},
    'arabic':     {'name': 'Arabski', 'dense': False, 'type': 'abdzad'},
    'thaana':     {'name': 'Thaana', 'dense': False, 'type': 'abugida'},
    'devanagari': {'name': 'Dewanagari', 'dense': False, 'type': 'abugida'},
    'bengali':    {'name': 'Bengalski', 'dense': False, 'type': 'abugida'},
    'gurmukhi':   {'name': 'Gurmukhi', 'dense': False, 'type': 'abugida'},
    'gujarati':   {'name': 'Gudzaracki', 'dense': False, 'type': 'abugida'},
    'oriya':      {'name': 'Odia', 'dense': False, 'type': 'abugida'},
    'tamil':      {'name': 'Tamilski', 'dense': False, 'type': 'abugida'},
    'telugu':     {'name': 'Telugu', 'dense': False, 'type': 'abugida'},
    'kannada':    {'name': 'Kannada', 'dense': False, 'type': 'abugida'},
    'malayalam':  {'name': 'Malajalam', 'dense': False, 'type': 'abugida'},
    'sinhala':    {'name': 'Syngaleski', 'dense': False, 'type': 'abugida'},
    'ethiopic':   {'name': 'Etiopski (ge’ez)', 'dense': False, 'type': 'abugida'},
    'thai':       {'name': 'Tajski', 'dense': True, 'type': 'abugida'},
    'lao':        {'name': 'Laotanski', 'dense': True, 'type': 'abugida'},
    'khmer':      {'name': 'Khmerski', 'dense': True, 'type': 'abugida'},
    'myanmar':    {'name': 'Birmanski', 'dense': True, 'type': 'abugida'},
    'tibetan':    {'name': 'Tybetanski', 'dense': True, 'type': 'abugida'},
    'han':        {'name': 'Han (hanzi)', 'dense': True, 'type': 'logograficzne'},
    'japanese':   {'name': 'Japonski (kanji + kana)', 'dense': True, 'type': 'mieszane'},
    'hangul':     {'name': 'Hangul', 'dense': True, 'type': 'alfasylabariusz'},
    'mixed':      {'name': 'Mieszany', 'dense': False, 'type': 'nieokreslony'},
    'unknown':    {'name': 'Nieznany', 'dense': False, 'type': 'nieokreslony'},
}

DENSE_SCRIPTS = frozenset(code for code, info in SCRIPT_INFO.items() if info['dense'])

# --- Jezyki -----------------------------------------------------------------
# Kod ISO -> rodzina jezykowa, pismo, nazwa.
LANGUAGE_INFO = {
    # Germanskie
    'en': {'family': 'Germanic', 'script': 'latin', 'name': 'English'},
    'de': {'family': 'Germanic', 'script': 'latin', 'name': 'German'},
    'nl': {'family': 'Germanic', 'script': 'latin', 'name': 'Dutch'},
    'sv': {'family': 'Germanic', 'script': 'latin', 'name': 'Swedish'},
    'da': {'family': 'Germanic', 'script': 'latin', 'name': 'Danish'},
    'no': {'family': 'Germanic', 'script': 'latin', 'name': 'Norwegian'},
    'af': {'family': 'Germanic', 'script': 'latin', 'name': 'Afrikaans'},
    'is': {'family': 'Germanic', 'script': 'latin', 'name': 'Icelandic'},
    'fy': {'family': 'Germanic', 'script': 'latin', 'name': 'Frisian'},
    'lb': {'family': 'Germanic', 'script': 'latin', 'name': 'Luxembourgish'},
    'yi': {'family': 'Germanic', 'script': 'hebrew', 'name': 'Yiddish'},

    # Romanskie
    'es': {'family': 'Romance', 'script': 'latin', 'name': 'Spanish'},
    'fr': {'family': 'Romance', 'script': 'latin', 'name': 'French'},
    'it': {'family': 'Romance', 'script': 'latin', 'name': 'Italian'},
    'pt': {'family': 'Romance', 'script': 'latin', 'name': 'Portuguese'},
    'ca': {'family': 'Romance', 'script': 'latin', 'name': 'Catalan'},
    'ro': {'family': 'Romance', 'script': 'latin', 'name': 'Romanian'},
    'gl': {'family': 'Romance', 'script': 'latin', 'name': 'Galician'},
    'oc': {'family': 'Romance', 'script': 'latin', 'name': 'Occitan'},
    'la': {'family': 'Romance', 'script': 'latin', 'name': 'Latin'},

    # Slowianskie
    'pl': {'family': 'Slavic', 'script': 'latin', 'name': 'Polish'},
    'cs': {'family': 'Slavic', 'script': 'latin', 'name': 'Czech'},
    'sk': {'family': 'Slavic', 'script': 'latin', 'name': 'Slovak'},
    'sl': {'family': 'Slavic', 'script': 'latin', 'name': 'Slovenian'},
    'hr': {'family': 'Slavic', 'script': 'latin', 'name': 'Croatian'},
    'sh': {'family': 'Slavic', 'script': 'latin', 'name': 'Serbo-Croatian'},
    'ru': {'family': 'Slavic', 'script': 'cyrillic', 'name': 'Russian'},
    'uk': {'family': 'Slavic', 'script': 'cyrillic', 'name': 'Ukrainian'},
    'bg': {'family': 'Slavic', 'script': 'cyrillic', 'name': 'Bulgarian'},
    'sr': {'family': 'Slavic', 'script': 'cyrillic', 'name': 'Serbian'},
    'mk': {'family': 'Slavic', 'script': 'cyrillic', 'name': 'Macedonian'},
    'be': {'family': 'Slavic', 'script': 'cyrillic', 'name': 'Belarusian'},

    # Baltyckie / ugrofinskie
    'lv': {'family': 'Baltic', 'script': 'latin', 'name': 'Latvian'},
    'lt': {'family': 'Baltic', 'script': 'latin', 'name': 'Lithuanian'},
    'fi': {'family': 'Finno-Ugric', 'script': 'latin', 'name': 'Finnish'},
    'hu': {'family': 'Finno-Ugric', 'script': 'latin', 'name': 'Hungarian'},
    'et': {'family': 'Finno-Ugric', 'script': 'latin', 'name': 'Estonian'},
    'mhr': {'family': 'Finno-Ugric', 'script': 'cyrillic', 'name': 'Mari'},
    'mrj': {'family': 'Finno-Ugric', 'script': 'cyrillic', 'name': 'Hill Mari'},
    'udm': {'family': 'Finno-Ugric', 'script': 'cyrillic', 'name': 'Udmurt'},
    'kv': {'family': 'Finno-Ugric', 'script': 'cyrillic', 'name': 'Komi'},
    'koi': {'family': 'Finno-Ugric', 'script': 'cyrillic', 'name': 'Komi-Permyak'},
    'myv': {'family': 'Finno-Ugric', 'script': 'cyrillic', 'name': 'Erzya'},
    'mdf': {'family': 'Finno-Ugric', 'script': 'cyrillic', 'name': 'Moksha'},

    # Turkijskie
    'tr': {'family': 'Turkic', 'script': 'latin', 'name': 'Turkish'},
    'az': {'family': 'Turkic', 'script': 'latin', 'name': 'Azerbaijani'},
    'uz': {'family': 'Turkic', 'script': 'latin', 'name': 'Uzbek'},
    'tk': {'family': 'Turkic', 'script': 'latin', 'name': 'Turkmen'},
    'kk': {'family': 'Turkic', 'script': 'cyrillic', 'name': 'Kazakh'},
    'ky': {'family': 'Turkic', 'script': 'cyrillic', 'name': 'Kyrgyz'},
    'tt': {'family': 'Turkic', 'script': 'cyrillic', 'name': 'Tatar'},
    'ba': {'family': 'Turkic', 'script': 'cyrillic', 'name': 'Bashkir'},
    'cv': {'family': 'Turkic', 'script': 'cyrillic', 'name': 'Chuvash'},
    'sah': {'family': 'Turkic', 'script': 'cyrillic', 'name': 'Yakut'},
    'tyv': {'family': 'Turkic', 'script': 'cyrillic', 'name': 'Tuvan'},
    'krc': {'family': 'Turkic', 'script': 'cyrillic', 'name': 'Karachay-Balkar'},
    'ug': {'family': 'Turkic', 'script': 'arabic', 'name': 'Uyghur'},

    # Kaukaskie
    'av': {'family': 'Caucasian', 'script': 'cyrillic', 'name': 'Avar'},
    'ce': {'family': 'Caucasian', 'script': 'cyrillic', 'name': 'Chechen'},
    'kbd': {'family': 'Caucasian', 'script': 'cyrillic', 'name': 'Kabardian'},
    'ady': {'family': 'Caucasian', 'script': 'cyrillic', 'name': 'Adyghe'},
    'lez': {'family': 'Caucasian', 'script': 'cyrillic', 'name': 'Lezgian'},
    'ka': {'family': 'Kartvelian', 'script': 'georgian', 'name': 'Georgian'},

    # Mongolskie
    'mn': {'family': 'Mongolic', 'script': 'cyrillic', 'name': 'Mongolian'},
    'xal': {'family': 'Mongolic', 'script': 'cyrillic', 'name': 'Kalmyk'},
    'bxr': {'family': 'Mongolic', 'script': 'cyrillic', 'name': 'Buryat'},

    # Iranskie
    'fa': {'family': 'Iranian', 'script': 'arabic', 'name': 'Persian'},
    'ps': {'family': 'Iranian', 'script': 'arabic', 'name': 'Pashto'},
    'ckb': {'family': 'Iranian', 'script': 'arabic', 'name': 'Central Kurdish'},
    'tg': {'family': 'Iranian', 'script': 'cyrillic', 'name': 'Tajik'},
    'os': {'family': 'Iranian', 'script': 'cyrillic', 'name': 'Ossetian'},

    # Semickie / afroazjatyckie
    'ar': {'family': 'Semitic', 'script': 'arabic', 'name': 'Arabic'},
    'arz': {'family': 'Semitic', 'script': 'arabic', 'name': 'Egyptian Arabic'},
    'he': {'family': 'Semitic', 'script': 'hebrew', 'name': 'Hebrew'},
    'am': {'family': 'Semitic', 'script': 'ethiopic', 'name': 'Amharic'},
    'mt': {'family': 'Semitic', 'script': 'latin', 'name': 'Maltese'},
    'so': {'family': 'Cushitic', 'script': 'latin', 'name': 'Somali'},

    # Indoaryjskie
    'hi': {'family': 'Indo-Aryan', 'script': 'devanagari', 'name': 'Hindi'},
    'mr': {'family': 'Indo-Aryan', 'script': 'devanagari', 'name': 'Marathi'},
    'ne': {'family': 'Indo-Aryan', 'script': 'devanagari', 'name': 'Nepali'},
    'sa': {'family': 'Indo-Aryan', 'script': 'devanagari', 'name': 'Sanskrit'},
    'mai': {'family': 'Indo-Aryan', 'script': 'devanagari', 'name': 'Maithili'},
    'bn': {'family': 'Indo-Aryan', 'script': 'bengali', 'name': 'Bengali'},
    'as': {'family': 'Indo-Aryan', 'script': 'bengali', 'name': 'Assamese'},
    'pa': {'family': 'Indo-Aryan', 'script': 'gurmukhi', 'name': 'Punjabi'},
    'gu': {'family': 'Indo-Aryan', 'script': 'gujarati', 'name': 'Gujarati'},
    'or': {'family': 'Indo-Aryan', 'script': 'oriya', 'name': 'Odia'},
    'si': {'family': 'Indo-Aryan', 'script': 'sinhala', 'name': 'Sinhala'},
    'ur': {'family': 'Indo-Aryan', 'script': 'arabic', 'name': 'Urdu'},
    'sd': {'family': 'Indo-Aryan', 'script': 'arabic', 'name': 'Sindhi'},
    'dv': {'family': 'Indo-Aryan', 'script': 'thaana', 'name': 'Dhivehi'},

    # Drawidyjskie
    'ta': {'family': 'Dravidian', 'script': 'tamil', 'name': 'Tamil'},
    'te': {'family': 'Dravidian', 'script': 'telugu', 'name': 'Telugu'},
    'kn': {'family': 'Dravidian', 'script': 'kannada', 'name': 'Kannada'},
    'ml': {'family': 'Dravidian', 'script': 'malayalam', 'name': 'Malayalam'},

    # Wschodnia i poludniowo-wschodnia Azja
    'zh': {'family': 'Sino-Tibetan', 'script': 'han', 'name': 'Chinese (Mandarin)'},
    'yue': {'family': 'Sino-Tibetan', 'script': 'han', 'name': 'Cantonese'},
    'ja': {'family': 'Japonic', 'script': 'japanese', 'name': 'Japanese'},
    'ko': {'family': 'Koreanic', 'script': 'hangul', 'name': 'Korean'},
    'my': {'family': 'Sino-Tibetan', 'script': 'myanmar', 'name': 'Burmese'},
    'bo': {'family': 'Sino-Tibetan', 'script': 'tibetan', 'name': 'Tibetan'},
    'th': {'family': 'Tai-Kadai', 'script': 'thai', 'name': 'Thai'},
    'lo': {'family': 'Tai-Kadai', 'script': 'lao', 'name': 'Lao'},
    'km': {'family': 'Austroasiatic', 'script': 'khmer', 'name': 'Khmer'},
    'vi': {'family': 'Austroasiatic', 'script': 'latin', 'name': 'Vietnamese'},

    # Austronezyjskie
    'id': {'family': 'Austronesian', 'script': 'latin', 'name': 'Indonesian'},
    'ms': {'family': 'Austronesian', 'script': 'latin', 'name': 'Malay'},
    'tl': {'family': 'Austronesian', 'script': 'latin', 'name': 'Tagalog'},
    'jv': {'family': 'Austronesian', 'script': 'latin', 'name': 'Javanese'},
    'su': {'family': 'Austronesian', 'script': 'latin', 'name': 'Sundanese'},
    'ceb': {'family': 'Austronesian', 'script': 'latin', 'name': 'Cebuano'},

    # Pozostale
    'el': {'family': 'Hellenic', 'script': 'greek', 'name': 'Greek'},
    'hy': {'family': 'Armenian', 'script': 'armenian', 'name': 'Armenian'},
    'sq': {'family': 'Albanian', 'script': 'latin', 'name': 'Albanian'},
    'cy': {'family': 'Celtic', 'script': 'latin', 'name': 'Welsh'},
    'ga': {'family': 'Celtic', 'script': 'latin', 'name': 'Irish'},
    'gd': {'family': 'Celtic', 'script': 'latin', 'name': 'Scottish Gaelic'},
    'br': {'family': 'Celtic', 'script': 'latin', 'name': 'Breton'},
    'eu': {'family': 'Isolate', 'script': 'latin', 'name': 'Basque'},
    'sw': {'family': 'Bantu', 'script': 'latin', 'name': 'Swahili'},
    'yo': {'family': 'Niger-Congo', 'script': 'latin', 'name': 'Yoruba'},
    'eo': {'family': 'Constructed', 'script': 'latin', 'name': 'Esperanto'},
}

# Starsze moduly i zapisane artefakty uzywaja klucza 'alphabet'.
for _info in LANGUAGE_INFO.values():
    _info.setdefault('alphabet', _info['script'])

LANGUAGES = list(LANGUAGE_INFO)

# Jezyki o wiekszym budzecie probek: rzadkie pisma oraz pary latwe do pomylenia
# (cs/sk, hr/sr/sh, id/ms, hi/mr/ne) potrzebuja wiecej danych.
SAMPLE_OVERRIDES = {
    'pl': 1600, 'cs': 1600, 'sk': 1600, 'hr': 1600, 'sl': 1600, 'sh': 1600,
    'sr': 1600, 'bg': 1600, 'mk': 1600, 'be': 1600, 'uk': 1600,
    'kk': 1600, 'ky': 1600, 'uz': 1400, 'tr': 1400, 'az': 1400, 'tk': 1400,
    'id': 1600, 'ms': 1600, 'tl': 1400, 'vi': 1400, 'sw': 1400,
    'zh': 1600, 'yue': 1600, 'ja': 1600, 'ko': 1600,
    'hi': 1600, 'mr': 1600, 'ne': 1600, 'sa': 1200, 'mai': 1200,
    'bn': 1600, 'as': 1200, 'ur': 1600, 'sd': 1200,
    'ar': 1600, 'arz': 1200, 'fa': 1600, 'ps': 1200, 'ckb': 1200, 'ug': 1200,
    'th': 1400, 'lo': 1200, 'km': 1200, 'my': 1200, 'bo': 1200,
    'he': 1400, 'yi': 1200, 'am': 1200, 'ka': 1400, 'hy': 1400, 'el': 1400,
    'ta': 1400, 'te': 1400, 'kn': 1400, 'ml': 1400, 'si': 1200,
    'pa': 1200, 'gu': 1200, 'or': 1200, 'dv': 1200,
}

# Zachowana nazwa dla zgodnosci ze starszym kodem.
HIGH_PRIORITY = SAMPLE_OVERRIDES


def language_name(code):
    return LANGUAGE_INFO.get(code, {}).get('name', code)


def language_family(code):
    return LANGUAGE_INFO.get(code, {}).get('family', 'unknown')


def language_script(code):
    info = LANGUAGE_INFO.get(code, {})
    return info.get('script') or info.get('alphabet', 'unknown')


def script_name(script):
    return SCRIPT_INFO.get(script, {}).get('name', script)


def is_dense_script(script):
    return script in DENSE_SCRIPTS


def min_text_length(script, training=False):
    if training:
        return MIN_TRAIN_TEXT_LENGTH_DENSE if is_dense_script(script) else MIN_TRAIN_TEXT_LENGTH
    return MIN_TEXT_LENGTH_DENSE if is_dense_script(script) else MIN_TEXT_LENGTH


def languages_by_script():
    grouped = {}
    for code in LANGUAGES:
        grouped.setdefault(language_script(code), []).append(code)
    return grouped
