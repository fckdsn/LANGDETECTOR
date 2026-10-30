"""Orkiestracja treningu: pobranie korpusu, przygotowanie probek, trening."""

import sys
from collections import Counter

from config import (
    LANGUAGES,
    MAX_TEXT_CHARS,
    OUTPUT_DIR,
    language_script,
    min_text_length,
)
from data_loader import load_multiple_languages
from preprocess import clean_text, script_share
from train import train_complete_model

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Dokument musi zawierac przynajmniej tyle znakow pisma deklarowanego dla
# danego jezyka. Filtr usuwa z korpusu strony bedace w praktyce w innym jezyku
# (np. angielskie menu na stronie tajskiej) oraz smieciowe wpisy.
MIN_SCRIPT_SHARE = 0.3


def prepare_samples(records, min_script_share=MIN_SCRIPT_SHARE):
    """Czysci teksty i przypisuje etykiety jezyka oraz pisma.

    Etykieta pisma pochodzi z konfiguracji jezyka (a nie z heurystyki), bo to
    ona jest prawdziwa klasa docelowa. Heurystyka Unicode sluzy tu wylacznie do
    odrzucenia dokumentow, ktore nie sa napisane deklarowanym pismem.
    """
    texts, labels, scripts = [], [], []
    dropped_short = 0
    dropped_script = 0

    for record in records:
        language = record['language']
        script = language_script(language)
        text = clean_text(record.get('text', '')[:MAX_TEXT_CHARS])

        if len(text) < min_text_length(script, training=True):
            dropped_short += 1
            continue
        if script_share(text, script) < min_script_share:
            dropped_script += 1
            continue

        texts.append(text)
        labels.append(language)
        scripts.append(script)

    print(f"\nPrzygotowano {len(texts)} probek, {len(set(labels))} jezykow, "
          f"{len(set(scripts))} pism")
    print(f"  Odrzucono zbyt krotkie:        {dropped_short}")
    print(f"  Odrzucono niezgodne z pismem:  {dropped_script}")

    counts = Counter(labels)
    if counts:
        rarest = counts.most_common()[-5:]
        print(f"  Najmniej liczne jezyki: "
              f"{', '.join(f'{code}={count}' for code, count in rarest)}")

    return texts, labels, scripts


def load_and_prepare_data(languages=None, samples=None):
    print("=" * 70)
    print("POBIERANIE I PRZYGOTOWANIE DANYCH")
    print("=" * 70)
    records = load_multiple_languages(languages or LANGUAGES, samples)
    if not records:
        raise RuntimeError(
            "Nie pobrano zadnych danych. Sprawdz token HF_TOKEN i polaczenie z siecia."
        )
    return prepare_samples(records)


def run_training(languages=None, samples=None, model_names=None,
                 keep_candidates=False, run_eda=True):
    texts, labels, scripts = load_and_prepare_data(languages, samples)
    results = train_complete_model(
        texts, labels, scripts,
        model_names=model_names,
        keep_candidates=keep_candidates,
        run_eda=run_eda,
    )
    print(f"\nWszystkie pliki wynikowe w katalogu '{OUTPUT_DIR}/'")
    return results


def main():
    run_training()


if __name__ == "__main__":
    main()
