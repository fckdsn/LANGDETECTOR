"""Przetwarzanie wstepne tekstu i rozpoznawanie pisma.

Modul jest swiadomy Unicode: obsluguje alfabety (lacinski, cyrylica, grecki),
abdzady (arabski, hebrajski), abugidy (dewanagari, tamilski, tajski) oraz pisma
logograficzne i mieszane (chinski, japonski, koreanski).
"""

import re

# Zakresy Unicode dla poszczegolnych pism. Kazdy zakres obejmuje litery oraz
# znaki diakrytyczne/samogloskowe danego pisma - w abugidach (dewanagari,
# tamilski, tajski) matry i tony to osobne punkty kodowe kategorii Mn/Mc i ich
# usuniecie zniszczylo by tekst.
SCRIPT_RANGES = {
    'latin': [
        (0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F),
        (0x1E00, 0x1EFF), (0x2C60, 0x2C7F), (0xA720, 0xA7FF),
    ],
    'cyrillic': [
        (0x0400, 0x052F), (0x1C80, 0x1C8F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F),
    ],
    'greek': [(0x0370, 0x03FF), (0x1F00, 0x1FFF)],
    'armenian': [(0x0530, 0x058F), (0xFB13, 0xFB17)],
    'hebrew': [(0x0590, 0x05FF), (0xFB1D, 0xFB4F)],
    'arabic': [
        (0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF),
        (0xFB50, 0xFDFF), (0xFE70, 0xFEFF),
    ],
    'thaana': [(0x0780, 0x07BF)],
    'ethiopic': [(0x1200, 0x139F), (0x2D80, 0x2DDF)],
    'georgian': [(0x10A0, 0x10FF), (0x1C90, 0x1CBF), (0x2D00, 0x2D2F)],
    'devanagari': [(0x0900, 0x097F), (0xA8E0, 0xA8FF)],
    'bengali': [(0x0980, 0x09FF)],
    'gurmukhi': [(0x0A00, 0x0A7F)],
    'gujarati': [(0x0A80, 0x0AFF)],
    'oriya': [(0x0B00, 0x0B7F)],
    'tamil': [(0x0B80, 0x0BFF)],
    'telugu': [(0x0C00, 0x0C7F)],
    'kannada': [(0x0C80, 0x0CFF)],
    'malayalam': [(0x0D00, 0x0D7F)],
    'sinhala': [(0x0D80, 0x0DFF)],
    'thai': [(0x0E00, 0x0E7F)],
    'lao': [(0x0E80, 0x0EFF)],
    'tibetan': [(0x0F00, 0x0FFF)],
    'myanmar': [(0x1000, 0x109F), (0xAA60, 0xAA7F)],
    'khmer': [(0x1780, 0x17FF), (0x19E0, 0x19FF)],
    'han': [
        (0x2E80, 0x2EFF), (0x3005, 0x3005), (0x3007, 0x3007),
        (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF),
    ],
    # Kana jest osobnym licznikiem: japonski to mieszanka kana i kanji, wiec
    # dopiero obecnosc kana pozwala odroznic japonski od chinskiego.
    'kana': [
        (0x3041, 0x3096), (0x3099, 0x309F), (0x30A1, 0x30FF),
        (0x31F0, 0x31FF), (0xFF66, 0xFF9D),
    ],
    'hangul': [
        (0x1100, 0x11FF), (0x3130, 0x318F), (0xA960, 0xA97F), (0xAC00, 0xD7AF),
    ],
}

# Pisma bez odstepow miedzy wyrazami. Tekst w nich trzeba tokenizowac znak po
# znaku, bo domyslny tokenizer widzi cale zdanie jako jeden token. Hangul jest
# poza ta lista - koreanski rozdziela wyrazy spacjami.
SCRIPTIO_CONTINUA_SCRIPTS = ('han', 'kana', 'thai', 'lao', 'khmer', 'myanmar', 'tibetan')

# Zero-width joiner / non-joiner sa niewidoczne, ale niosa informacje o jezyku
# (perski, dewanagari), dlatego nie sa usuwane.
_ZERO_WIDTH = '\u200c\u200d'


def _char_class(ranges):
    parts = []
    for low, high in ranges:
        if low == high:
            parts.append(re.escape(chr(low)))
        else:
            parts.append(f"{re.escape(chr(low))}-{re.escape(chr(high))}")
    return ''.join(parts)


_SCRIPT_PATTERNS = {
    name: re.compile(f"[{_char_class(ranges)}]")
    for name, ranges in SCRIPT_RANGES.items()
}

_ALL_LETTERS = ''.join(_char_class(ranges) for ranges in SCRIPT_RANGES.values())
_DENSE_LETTERS = ''.join(
    _char_class(SCRIPT_RANGES[name]) for name in SCRIPTIO_CONTINUA_SCRIPTS
)

_KEEP_WITH_DIGITS = re.compile(f"[^{_ALL_LETTERS}{_ZERO_WIDTH}0-9\\s]")
_KEEP_NO_DIGITS = re.compile(f"[^{_ALL_LETTERS}{_ZERO_WIDTH}\\s]")
_WHITESPACE = re.compile(r"\s+")
_URL = re.compile(r"https?://\S+|www\.\S+")
_EMAIL = re.compile(r"\S+@\S+")

# Wzorzec tokenizacji dla TfidfVectorizer(analyzer='word'). Pierwszy warian
# alternatywy dopasowuje pojedyncze znaki pism bez spacji, dzieki czemu chinskie
# lub tajskie zdanie rozpada sie na tokeny zamiast zostac jednym.
WORD_TOKEN_PATTERN = f"[{_DENSE_LETTERS}]|\\b\\w\\w+\\b"


def clean_text(text, keep_digits=True):
    """Normalizuje tekst, zachowujac litery wszystkich obslugiwanych pism."""
    if not text:
        return ""
    text = text.lower()
    text = _WHITESPACE.sub(" ", text)
    text = _URL.sub("", text)
    text = _EMAIL.sub("", text)
    pattern = _KEEP_WITH_DIGITS if keep_digits else _KEEP_NO_DIGITS
    text = pattern.sub("", text)
    text = " ".join(w for w in text.split() if len(w) > 1)
    return text.strip()


def script_profile(text):
    """Zwraca udzial poszczegolnych pism w tekscie (suma udzialow = 1.0)."""
    if not text:
        return {}

    counts = {}
    for name, pattern in _SCRIPT_PATTERNS.items():
        found = len(pattern.findall(text))
        if found:
            counts[name] = found

    kana = counts.pop('kana', 0)
    if kana:
        han = counts.get('han', 0)
        if han and kana / (kana + han) < 0.05:
            # Pojedyncza katakana (nazwa wlasna, marka) w tekscie chinskim nie
            # czyni go japonskim - kanji zostaja przy pismie han.
            counts['japanese'] = kana
        else:
            # W prawdziwym tekscie japonskim kana stanowi okolo polowy znakow,
            # a kanji naleza do tego samego systemu pisma.
            counts['japanese'] = kana + han
            counts.pop('han', None)

    total = sum(counts.values())
    if not total:
        return {}
    return {name: count / total for name, count in counts.items()}


def detect_script(text, threshold=0.5):
    """Rozpoznaje dominujace pismo. Zwraca 'mixed' przy braku dominacji."""
    profile = script_profile(text)
    if not profile:
        return 'unknown'
    name, share = max(profile.items(), key=lambda item: item[1])
    return name if share >= threshold else 'mixed'


def script_share(text, script):
    """Udzial wskazanego pisma w tekscie - uzywany do filtrowania korpusu."""
    return script_profile(text).get(script, 0.0)


# Zachowana nazwa z wczesniejszej wersji API (wtedy tylko lacina i cyrylica).
detect_alphabet = detect_script
