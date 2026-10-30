"""Interfejs predykcji: rozpoznawanie jezyka oraz pisma pojedynczego tekstu."""

import os

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack

from config import (
    LANGUAGE_INFO,
    MODELS_DIR,
    language_family,
    language_name,
    language_script,
    min_text_length,
    script_name,
)
from preprocess import clean_text, detect_script, script_profile

TEST_FILE = 'test_languages.txt'


class ModelsNotFoundError(RuntimeError):
    """Podnoszony, gdy w katalogu modeli brakuje wytrenowanych artefaktow."""


def read_test_cases(path=TEST_FILE):
    """Czyta zbior ewaluacyjny w formacie 'kod_jezyka|tekst'."""
    if not os.path.exists(path):
        return []

    cases = []
    with open(path, 'r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith('#') or '|' not in line:
                continue
            code, text = line.split('|', 1)
            code, text = code.strip(), text.strip()
            if code and text:
                cases.append((code, text))
    return cases


class LanguageDetector:
    def __init__(self, models_dir=None):
        base = models_dir or MODELS_DIR
        language_model_path = os.path.join(base, 'language_model.pkl')
        if not os.path.exists(language_model_path):
            raise ModelsNotFoundError(
                f"Brak modelu w '{base}'. Uruchom najpierw: python main.py train"
            )

        self.models_dir = base
        self.model = joblib.load(language_model_path)
        self.encoder = joblib.load(os.path.join(base, 'language_encoder.pkl'))
        self.char_vec, self.word_vec = joblib.load(os.path.join(base, 'vectorizers.pkl'))

        self.language_info = self._load_optional('language_info.pkl') or LANGUAGE_INFO
        self.metadata = self._load_optional('model_metadata.pkl') or {}
        self.script_model = self._load_optional('script_model.pkl')
        self.script_encoder = self._load_optional('script_encoder.pkl')

    def _load_optional(self, filename):
        path = os.path.join(self.models_dir, filename)
        return joblib.load(path) if os.path.exists(path) else None

    @property
    def algorithm(self):
        """Nazwa algorytmu wybranego w porownaniu modeli."""
        return self.metadata.get('label', type(self.model).__name__)

    @property
    def languages(self):
        return list(self.encoder.classes_)

    def _vectorize(self, cleaned):
        return csr_matrix(hstack([
            self.char_vec.transform([cleaned]),
            self.word_vec.transform([cleaned]),
        ]))

    def _info(self, code):
        info = self.language_info.get(code, {})
        return {
            'language': code,
            'name': info.get('name', language_name(code)),
            'family': info.get('family', language_family(code)),
            'script': info.get('script') or info.get('alphabet') or language_script(code),
        }

    def predict(self, text, top_k=5, min_confidence=0.02):
        """Zwraca (kod jezyka, pewnosc, lista top-k przewidywan)."""
        cleaned = clean_text(text)
        script = detect_script(cleaned)
        if len(cleaned) < min_text_length(script):
            return None, 0, []

        probs = self.model.predict_proba(self._vectorize(cleaned))[0]
        order = np.argsort(probs)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(order):
            # Najlepsza hipoteza jest zawsze raportowana; kolejne tylko wtedy,
            # gdy przekraczaja prog istotnosci.
            if rank > 0 and probs[idx] <= min_confidence:
                continue
            code = self.encoder.inverse_transform([idx])[0]
            entry = self._info(code)
            entry['confidence'] = float(probs[idx])
            entry['alphabet'] = entry['script']
            results.append(entry)

        if not results:
            return None, 0, []
        return results[0]['language'], results[0]['confidence'], results

    def predict_script(self, text):
        """Rozpoznaje pismo tekstu: modelem ML, z heurystyka Unicode jako rezerwa."""
        cleaned = clean_text(text)
        profile = script_profile(cleaned)
        if not profile:
            # Brak liter w tekscie (same cyfry lub interpunkcja) - nie ma czego
            # klasyfikowac, a model dostalby wektor bez cech pisma.
            return {'script': 'unknown', 'name': script_name('unknown'),
                    'confidence': 0.0, 'source': 'heurystyka Unicode', 'profile': {}}

        if self.script_model is not None and self.script_encoder is not None:
            probs = self.script_model.predict_proba(self._vectorize(cleaned))[0]
            idx = int(np.argmax(probs))
            code = self.script_encoder.inverse_transform([idx])[0]
            return {
                'script': code,
                'name': script_name(code),
                'confidence': float(probs[idx]),
                'source': 'model ML',
                'profile': profile,
            }

        heuristic = detect_script(cleaned)
        return {
            'script': heuristic,
            'name': script_name(heuristic),
            'confidence': profile.get(heuristic, 0.0),
            'source': 'heurystyka Unicode',
            'profile': profile,
        }

    def analyze(self, text, top_k=5):
        """Pelna analiza tekstu - jezyk, pismo i rozklad prawdopodobienstwa."""
        language, confidence, predictions = self.predict(text, top_k=top_k)
        return {
            'text_length': len(text),
            'language': language,
            'confidence': confidence,
            'predictions': predictions,
            'script': self.predict_script(text),
            'algorithm': self.algorithm,
        }


def _print_analysis(analysis):
    if not analysis['language']:
        print("  Nie rozpoznano jezyka (tekst zbyt krotki lub bez liter).")
        return

    script = analysis['script']
    top = analysis['predictions'][0]
    print(f"\n  JEZYK:    {analysis['language'].upper()} ({top['name']})")
    print(f"  PEWNOSC:  {analysis['confidence']:.1%}")
    print(f"  RODZINA:  {top['family']}")
    print(f"  PISMO:    {script['name']} ({script['script']}) "
          f"- {script['confidence']:.1%}, zrodlo: {script['source']}")
    print("\n  Top przewidywania:")
    for i, item in enumerate(analysis['predictions'], 1):
        bar = '#' * int(item['confidence'] * 30)
        print(f"    {i}. {item['language']:<5} {item['name'][:22]:<23} "
              f"{item['confidence']:>6.1%} {bar}")


def main():
    print("=" * 60)
    print("DETEKTOR JEZYKA - TRYB INTERAKTYWNY")
    print("=" * 60)

    try:
        detector = LanguageDetector()
    except ModelsNotFoundError as exc:
        print(f"\n{exc}")
        return

    print(f"Algorytm:      {detector.algorithm}")
    print(f"Liczba jezykow: {len(detector.languages)}")
    print("Wpisz 'exit', aby zakonczyc.\n")

    while True:
        text = input("\n>> Tekst: ").strip()
        if text.lower() in ('exit', 'quit'):
            break
        if not text:
            continue
        _print_analysis(detector.analyze(text))


if __name__ == "__main__":
    main()
