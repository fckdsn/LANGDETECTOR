import re


def clean_text(text, keep_digits=True):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    pattern = r"[^a-zа-яё0-9\s]" if keep_digits else r"[^a-zа-яё\s]"
    text = re.sub(pattern, "", text)
    text = " ".join([w for w in text.split() if len(w) > 1])
    return text.strip()

def detect_alphabet(text):
    text = text.lower()
    latin = len(re.findall(r'[a-z]', text))
    cyrillic = len(re.findall(r'[а-яё]', text))
    if latin > cyrillic:
        return 'latin'
    elif cyrillic > latin:
        return 'cyrillic'
    return 'mixed'
