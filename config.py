import os

HF_TOKEN = os.environ.get("HF_TOKEN", "")
os.environ["HF_TOKEN"] = HF_TOKEN

OUTPUT_DIR = 'pliki'
MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')

LANGUAGES = [
    'en', 'de', 'nl', 'sv', 'da', 'no', 'af',
    'es', 'fr', 'it', 'pt', 'ca',
    'pl', 'sk', 'sl',
    'ru', 'uk', 'bg', 'sr', 'mk', 'be',
    'fi', 'hu', 'et', 'lv', 'lt',
    'tr', 'uz', 'ky', 'sq', 'mt', 'ce',
    'id', 'ms', 'tl', 'vi', 'sw', 'sh'
]

LANGUAGE_INFO = {
    'en': {'family': 'Germanic', 'alphabet': 'latin', 'name': 'English'},
    'de': {'family': 'Germanic', 'alphabet': 'latin', 'name': 'German'},
    'nl': {'family': 'Germanic', 'alphabet': 'latin', 'name': 'Dutch'},
    'sv': {'family': 'Germanic', 'alphabet': 'latin', 'name': 'Swedish'},
    'da': {'family': 'Germanic', 'alphabet': 'latin', 'name': 'Danish'},
    'no': {'family': 'Germanic', 'alphabet': 'latin', 'name': 'Norwegian'},
    'af': {'family': 'Germanic', 'alphabet': 'latin', 'name': 'Afrikaans'},
    'es': {'family': 'Romance', 'alphabet': 'latin', 'name': 'Spanish'},
    'fr': {'family': 'Romance', 'alphabet': 'latin', 'name': 'French'},
    'it': {'family': 'Romance', 'alphabet': 'latin', 'name': 'Italian'},
    'pt': {'family': 'Romance', 'alphabet': 'latin', 'name': 'Portuguese'},
    'ca': {'family': 'Romance', 'alphabet': 'latin', 'name': 'Catalan'},
    'pl': {'family': 'Slavic', 'alphabet': 'latin', 'name': 'Polish'},
    'sk': {'family': 'Slavic', 'alphabet': 'latin', 'name': 'Slovak'},
    'sl': {'family': 'Slavic', 'alphabet': 'latin', 'name': 'Slovenian'},
    'ru': {'family': 'Slavic', 'alphabet': 'cyrillic', 'name': 'Russian'},
    'uk': {'family': 'Slavic', 'alphabet': 'cyrillic', 'name': 'Ukrainian'},
    'bg': {'family': 'Slavic', 'alphabet': 'cyrillic', 'name': 'Bulgarian'},
    'sr': {'family': 'Slavic', 'alphabet': 'cyrillic', 'name': 'Serbian'},
    'mk': {'family': 'Slavic', 'alphabet': 'cyrillic', 'name': 'Macedonian'},
    'be': {'family': 'Slavic', 'alphabet': 'cyrillic', 'name': 'Belarusian'},
    'fi': {'family': 'Finno-Ugric', 'alphabet': 'latin', 'name': 'Finnish'},
    'hu': {'family': 'Finno-Ugric', 'alphabet': 'latin', 'name': 'Hungarian'},
    'et': {'family': 'Finno-Ugric', 'alphabet': 'latin', 'name': 'Estonian'},
    'lv': {'family': 'Baltic', 'alphabet': 'latin', 'name': 'Latvian'},
    'lt': {'family': 'Baltic', 'alphabet': 'latin', 'name': 'Lithuanian'},
    'tr': {'family': 'Turkic', 'alphabet': 'latin', 'name': 'Turkish'},
    'uz': {'family': 'Turkic', 'alphabet': 'latin', 'name': 'Uzbek'},
    'ky': {'family': 'Turkic', 'alphabet': 'cyrillic', 'name': 'Kyrgyz'},
    'sq': {'family': 'Albanian', 'alphabet': 'latin', 'name': 'Albanian'},
    'mt': {'family': 'Semitic', 'alphabet': 'latin', 'name': 'Maltese'},
    'ce': {'family': 'Caucasian', 'alphabet': 'cyrillic', 'name': 'Chechen'},
    'id': {'family': 'Austronesian', 'alphabet': 'latin', 'name': 'Indonesian'},
    'ms': {'family': 'Austronesian', 'alphabet': 'latin', 'name': 'Malay'},
    'tl': {'family': 'Austronesian', 'alphabet': 'latin', 'name': 'Tagalog'},
    'vi': {'family': 'Austroasiatic', 'alphabet': 'latin', 'name': 'Vietnamese'},
    'sw': {'family': 'Bantu', 'alphabet': 'latin', 'name': 'Swahili'},
    'sh': {'family': 'Slavic', 'alphabet': 'latin', 'name': 'Serbo-Croatian'},
}