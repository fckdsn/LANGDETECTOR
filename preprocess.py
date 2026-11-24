"""Przetwarzanie wstepne tekstu i rozpoznawanie pisma.

Modul jest swiadomy Unicode: obsluguje alfabety (lacinski, cyrylica, grecki),
abdzady (arabski, hebrajski), abugidy (dewanagari, tamilski, tajski) oraz pisma
logograficzne i mieszane (chinski, japonski, koreanski).

Algorytm dla pism innych niz lacinka i cyrylica:

1. Normalizacja NFKC. Pelnoszerokosciowa lacinka, polszerokosciowa kana, ligatury
   i ideogramy kompatybilnosciowe zamieniaja sie na zwykle znaki. Bez tego
   np. „ＡＩ” z tekstu japonskiego bylo usuwane.
2. Litery biora sie z zakresow Unicode (``SCRIPT_RANGES``), a nie z listy
   liter lacinskich i cyrylickich. Zachowane sa znaki kategorii Mn/Mc: matry
   dewanagari, virama, ogloski arabskie, tony tajskie. Interpunkcja lezaca
   w tym samym bloku (np. danda) jest odrzucana.
3. Pisma bez spacji (han, kana, tajski, laotanski, khmerski, birmanski,
   tybetanski): odstepy miedzy ich znakami sa usuwane, wiec „人 工 智 能”
   nie znika i pokrywa sie z zapisem ciaglym. Token slowy to znak bazowy
   plus jego znaki kombinujace (sylaba), a nie cale zdanie.
4. Abugidy i abdzady ze spacjami (dewanagari / hindi, bengalski, tamilski,
   gurmukhi, gudzaracki, odia, telugu, kannada, malajalam, syngaleski, thaana,
   arabski, hebrajski): cale slowo, razem z matrami, virama i ZWJ/ZWNJ, jest
   jednym tokenem. Wzorzec ``\\w`` rozcinalby je, bo Mn/Mc nie sa „literami”.
"""

import re
import unicodedata

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
        (0x0600, 0x06FF), (0x0750, 0x077F), (0x0870, 0x089F), (0x08A0, 0x08FF),
        (0xFB50, 0xFDFF), (0xFE70, 0xFEFF), (0x10EC0, 0x10EFF),
    ],
    'thaana': [(0x0780, 0x07BF)],
    'ethiopic': [
        (0x1200, 0x139F), (0x2D80, 0x2DDF), (0xAB00, 0xAB2F), (0x1E7E0, 0x1E7FF),
    ],
    'georgian': [(0x10A0, 0x10FF), (0x1C90, 0x1CBF), (0x2D00, 0x2D2F)],
    'devanagari': [
        (0x0900, 0x097F), (0x1CD0, 0x1CFF), (0xA8E0, 0xA8FF), (0x11B00, 0x11B5F),
    ],
    'bengali': [(0x0980, 0x09FF), (0x11DF0, 0x11DFF)],
    'gurmukhi': [(0x0A00, 0x0A7F)],
    'gujarati': [(0x0A80, 0x0AFF)],
    'oriya': [(0x0B00, 0x0B7F)],
    'tamil': [(0x0B80, 0x0BFF), (0x11FC0, 0x11FFF)],
    'telugu': [(0x0C00, 0x0C7F)],
    'kannada': [(0x0C80, 0x0CFF)],
    'malayalam': [(0x0D00, 0x0D7F)],
    'sinhala': [(0x0D80, 0x0DFF), (0x111E0, 0x111FF)],
    'thai': [(0x0E00, 0x0E7F)],
    'lao': [(0x0E80, 0x0EFF)],
    'tibetan': [(0x0F00, 0x0FFF)],
    'myanmar': [
        (0x1000, 0x109F), (0xA9E0, 0xA9FF), (0xAA60, 0xAA7F), (0x116D0, 0x116FF),
    ],
    'khmer': [(0x1780, 0x17FF), (0x19E0, 0x19FF)],
    'han': [
        (0x2E80, 0x2FDF), (0x3005, 0x3005), (0x3007, 0x3007),
        (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF),
        # Rozszerzenia CJK B–J i ideogramy kompatybilnosciowe (Unicode 18).
        # Bez nich rzadkie hanzi sa wycinane przy czyszczeniu.
        (0x20000, 0x2A6DF), (0x2A700, 0x2B73F), (0x2B740, 0x2B81F),
        (0x2B820, 0x2CEAF), (0x2CEB0, 0x2EBEF), (0x2EBF0, 0x2EE5F),
        (0x2F800, 0x2FA1F), (0x30000, 0x3134F), (0x31350, 0x323AF),
        (0x323B0, 0x3347F),
    ],
    # Kana jest osobnym licznikiem: japonski to mieszanka kana i kanji, wiec
    # dopiero obecnosc kana pozwala odroznic japonski od chinskiego.
    'kana': [
        (0x3041, 0x3096), (0x3099, 0x309F), (0x30A1, 0x30FF),
        (0x31F0, 0x31FF), (0xFF66, 0xFF9D),
        (0x1AFF0, 0x1AFFF), (0x1B000, 0x1B16F),
    ],
    'hangul': [
        (0x1100, 0x11FF), (0x3130, 0x318F), (0xA960, 0xA97F),
        (0xAC00, 0xD7AF), (0xD7B0, 0xD7FF),
    ],
}

# Pisma bez odstepow miedzy wyrazami. Tekst w nich trzeba tokenizowac znak po
# znaku (z doklejonymi znakami kombinujacymi), bo domyslny tokenizer widzi cale
# zdanie jako jeden token. Hangul jest poza ta lista - koreanski rozdziela
# wyrazy spacjami.
SCRIPTIO_CONTINUA_SCRIPTS = ('han', 'kana', 'thai', 'lao', 'khmer', 'myanmar', 'tibetan')

# Zero-width joiner / non-joiner sa niewidoczne, ale niosa informacje o jezyku
# (perski, dewanagari), dlatego nie sa usuwane.
_ZERO_WIDTH = '\u200c\u200d'


def _compress(codepoints):
    points = sorted(set(codepoints))
    if not points:
        return []
    ranges = []
    start = prev = points[0]
    for cp in points[1:]:
        if cp == prev + 1:
            prev = cp
            continue
        ranges.append((start, prev))
        start = prev = cp
    ranges.append((start, prev))
    return ranges


def _keep_codepoint(cp):
    """Litera, znak diakrytyczny, cyfra albo znak jeszcze nieznany tej wersji Unicode."""
    category = unicodedata.category(chr(cp))
    # Cn: Python 3.12 zna Unicode 15.0, a bloki CJK Extension I/J i czesc
    # uzupelnien wskazich doszly pozniej. Zostawiamy je, skoro zakres jest nasz.
    return category == 'Cn' or category[0] in 'LMN'


def _filtered_ranges(name, ranges):
    # Bloki han sa waskie (same ideogramy i radykały). Nie odsiewamy ich po
    # kategorii, bo rzadkie hanzi sa w starszym unicodedata oznaczone jako Cn.
    if name == 'han':
        return list(ranges)
    kept = (
        cp
        for low, high in ranges
        for cp in range(low, high + 1)
        if _keep_codepoint(cp)
    )
    return _compress(kept)


_FILTERED_RANGES = {
    name: _filtered_ranges(name, ranges) for name, ranges in SCRIPT_RANGES.items()
}


def _char_class(ranges):
    parts = []
    for low, high in ranges:
        if low == high:
            parts.append(re.escape(chr(low)))
        else:
            parts.append(f"{re.escape(chr(low))}-{re.escape(chr(high))}")
    return ''.join(parts)


def _dense_bases_and_marks():
    """Bazy i znaki kombinujace pism bez spacji.

    Han traktujemy w calosci jako bazy (nie iterujemy dziesiatek tysiecy
    ideogramow). Pozostale pisma ciagle sa male, wiec matry i tony da sie
    odlozyc do osobnej klasy i dokleic do poprzedzajacego znaku.
    """
    base_ranges = list(_FILTERED_RANGES['han'])
    base_cps = []
    mark_cps = []
    for name in SCRIPTIO_CONTINUA_SCRIPTS:
        if name == 'han':
            continue
        for low, high in _FILTERED_RANGES[name]:
            for cp in range(low, high + 1):
                if unicodedata.category(chr(cp)).startswith('M'):
                    mark_cps.append(cp)
                else:
                    base_cps.append(cp)
    base_ranges.extend(_compress(base_cps))
    return base_ranges, _compress(mark_cps)


_SCRIPT_PATTERNS = {
    name: re.compile(f"[{_char_class(ranges)}]")
    for name, ranges in _FILTERED_RANGES.items()
    if ranges
}

_ALL_LETTERS = ''.join(_char_class(ranges) for ranges in _FILTERED_RANGES.values() if ranges)
_DENSE_BASES, _DENSE_MARKS = _dense_bases_and_marks()
_DENSE_LETTERS = _char_class(_DENSE_BASES) + _char_class(_DENSE_MARKS)
_SPACED_LETTERS = ''.join(
    _char_class(ranges)
    for name, ranges in _FILTERED_RANGES.items()
    if name not in SCRIPTIO_CONTINUA_SCRIPTS and ranges
)

_KEEP_WITH_DIGITS = re.compile(f"[^{_ALL_LETTERS}{_ZERO_WIDTH}0-9\\s]")
_KEEP_NO_DIGITS = re.compile(f"[^{_ALL_LETTERS}{_ZERO_WIDTH}\\s]")
_WHITESPACE = re.compile(r"\s+")
_URL = re.compile(r"https?://\S+|www\.\S+")
_EMAIL = re.compile(r"\S+@\S+")
_CONTINUA_GAP = re.compile(
    f"(?<=[{_DENSE_LETTERS}])\\s+(?=[{_DENSE_LETTERS}])"
)

# Pierwsza alternatywa: jeden znak pisma ciaglego plus jego matry/tony.
# Druga: cale slowo pisma ze spacjami, lacznie ze znakami Mn/Mc i ZWJ/ZWNJ.
# Trzecia: liczby arabskie, ktorych nie ma w zakresach pism.
_DENSE_TOKEN = f"[{_char_class(_DENSE_BASES)}]"
if _DENSE_MARKS:
    _DENSE_TOKEN += f"[{_char_class(_DENSE_MARKS)}]*"
WORD_TOKEN_PATTERN = (
    f"{_DENSE_TOKEN}"
    f"|[{_SPACED_LETTERS}](?:[{_SPACED_LETTERS}{_ZERO_WIDTH}0-9])*"
    f"|[0-9]+"
)
_WORD_TOKEN_RE = re.compile(WORD_TOKEN_PATTERN)


def normalize_text(text):
    """NFKC: pelna szerokosc, ligatury i formy prezentacyjne."""
    if not text:
        return ""
    return unicodedata.normalize("NFKC", str(text))


def _keep_token(token):
    """Odrzuca samotna litere lacinska. Samotny ideogram albo litera innego pisma zostaje."""
    if len(token) > 1:
        return True
    return not token.isascii()


def clean_text(text, keep_digits=True):
    """Normalizuje tekst, zachowujac litery wszystkich obslugiwanych pism."""
    if not text:
        return ""
    text = normalize_text(text).lower()
    text = _WHITESPACE.sub(" ", text)
    text = _URL.sub("", text)
    text = _EMAIL.sub("", text)
    pattern = _KEEP_WITH_DIGITS if keep_digits else _KEEP_NO_DIGITS
    text = pattern.sub("", text)
    text = _WHITESPACE.sub(" ", text).strip()
    text = _CONTINUA_GAP.sub("", text)
    text = " ".join(token for token in text.split() if _keep_token(token))
    return text.strip()


def tokenize_words(text):
    """Tokeny slowne swiadome pisma. Patrz opis algorytmu w naglowku modulu."""
    if not text:
        return []
    return _WORD_TOKEN_RE.findall(text)


def script_profile(text):
    """Zwraca udzial poszczegolnych pism w tekscie (suma udzialow = 1.0)."""
    if not text:
        return {}
    text = normalize_text(text)

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
