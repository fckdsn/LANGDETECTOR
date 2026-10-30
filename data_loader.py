"""Pobieranie korpusu OSCAR z Hugging Face dla wszystkich skonfigurowanych jezykow."""

import json
import os
import sys
import time

from datasets import load_dataset
from huggingface_hub import HfFileSystem

from config import (
    HF_TOKEN,
    LANGUAGES,
    REPORTS_DIR,
    SAMPLE_OVERRIDES,
    SAMPLES_PER_LANGUAGE,
    is_dense_script,
    language_name,
    language_script,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATASET_REPO = "oscar-corpus/community-oscar"

# Minimalna dlugosc dokumentu w znakach. Pisma bez spacji (chinski, japonski,
# tajski) koduja te sama tresc na kilka razy mniejszej liczbie znakow, wiec prog
# musi byc dla nich nizszy - inaczej odrzucalibysmy poprawne dokumenty.
MIN_DOC_CHARS = 200
MIN_DOC_CHARS_DENSE = 60

# Zachowana nazwa dla zgodnosci ze starszym kodem.
HIGH_PRIORITY = SAMPLE_OVERRIDES

_SNAPSHOT_CACHE = None
_LANGUAGES_CACHE = None


def _fs():
    return HfFileSystem(token=HF_TOKEN)


def get_snapshots():
    global _SNAPSHOT_CACHE
    if _SNAPSHOT_CACHE is not None:
        return _SNAPSHOT_CACHE

    try:
        items = _fs().ls(f"datasets/{DATASET_REPO}/data", detail=False)
        snapshots = sorted({item.split('/')[-1] for item in items if '/' in item})
        snapshots = [s for s in snapshots if s and not s.startswith('.')]
        snapshots.sort(reverse=True)
        _SNAPSHOT_CACHE = snapshots
        return snapshots
    except Exception as exc:
        print(f"Error listing snapshots: {exc}")
        return []


def get_available_languages():
    global _LANGUAGES_CACHE
    if _LANGUAGES_CACHE is not None:
        return _LANGUAGES_CACHE

    print("Getting available languages...")
    snapshots = get_snapshots()
    if not snapshots:
        print("No snapshots found")
        return []

    snapshot = snapshots[0]
    print(f"Using snapshot: {snapshot}")

    try:
        items = _fs().ls(f"datasets/{DATASET_REPO}/data/{snapshot}", detail=False)
        languages = []
        for item in items:
            name = item.split('/')[-1]
            if name.endswith('_meta'):
                languages.append(name[:-len('_meta')])
        languages = sorted(set(languages))
        print(f"Found {len(languages)} languages in OSCAR (snapshot {snapshot})")
        _LANGUAGES_CACHE = languages
        return languages
    except Exception as exc:
        print(f"Error listing languages: {exc}")
        return []


def min_document_length(language):
    return MIN_DOC_CHARS_DENSE if is_dense_script(language_script(language)) else MIN_DOC_CHARS


def load_data(language, samples=None, timeout=90, min_len=None, max_chars=8000):
    if samples is None:
        samples = SAMPLE_OVERRIDES.get(language, SAMPLES_PER_LANGUAGE)
    if min_len is None:
        min_len = min_document_length(language)

    script = language_script(language)
    print(f"  {language:<5} {language_name(language)[:22]:<23} [{script:<10}]",
          end=" ", flush=True)

    snapshots = get_snapshots()
    if not snapshots:
        print("FAILED: no snapshots available")
        return []

    last_err = None
    for snapshot in snapshots:
        pattern = f"data/{snapshot}/{language}_meta/*.jsonl.zst"
        try:
            dataset = load_dataset(
                DATASET_REPO,
                data_files=pattern,
                split="train",
                token=HF_TOKEN,
                streaming=True,
            )

            result, start = [], time.time()
            for item in dataset:
                if len(result) >= samples or time.time() - start > timeout:
                    break
                content = item.get('content', '')
                if content and len(content) >= min_len:
                    result.append({'text': content[:max_chars], 'language': language})

            if result:
                print(f"OK {len(result)}/{samples} [{snapshot}]")
                return result
        except Exception as exc:
            last_err = exc
            continue

    print(f"FAILED: {str(last_err)[:70] if last_err else 'no data'}")
    return []


def load_multiple_languages(languages=None, samples=None, timeout=90,
                            write_report=True):
    """Pobiera dane dla listy jezykow, pomijajac te nieobecne w korpusie."""
    if languages is None:
        languages = LANGUAGES

    available = set(get_available_languages())
    to_load = [lang for lang in languages if lang in available]
    skipped = [lang for lang in languages if lang not in available]

    print(f"\nSkonfigurowane jezyki: {len(languages)}")
    print(f"Dostepne w OSCAR:      {len(to_load)}")
    if skipped:
        print(f"Pominiete (brak w korpusie, {len(skipped)}): {', '.join(skipped)}")

    print("\nPobieranie danych...")
    all_data, stats, failed = [], {}, []

    for i, lang in enumerate(to_load, 1):
        print(f"[{i:3d}/{len(to_load)}]", end="")
        budget = samples if samples is not None else SAMPLE_OVERRIDES.get(lang, SAMPLES_PER_LANGUAGE)
        data = load_data(lang, samples=budget, timeout=timeout)
        if data:
            all_data.extend(data)
            stats[lang] = len(data)
        else:
            failed.append(lang)

    by_script = {}
    for lang, count in stats.items():
        script = language_script(lang)
        by_script.setdefault(script, {'languages': 0, 'samples': 0})
        by_script[script]['languages'] += 1
        by_script[script]['samples'] += count

    print(f"\nRazem: {len(all_data)} probek z {len(stats)} jezykow")
    print("\nRozklad po pismach:")
    for script in sorted(by_script, key=lambda s: -by_script[s]['samples']):
        info = by_script[script]
        print(f"  {script:<12} {info['languages']:>3} jezykow, {info['samples']:>7} probek")
    if failed:
        print(f"\nBez danych ({len(failed)}): {', '.join(failed)}")

    if write_report:
        _write_load_report(stats, skipped, failed, by_script)

    return all_data


def _write_load_report(stats, skipped, failed, by_script):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = os.path.join(REPORTS_DIR, 'dataset_composition.json')
    payload = {
        'dataset': DATASET_REPO,
        'total_samples': sum(stats.values()),
        'languages_loaded': len(stats),
        'per_language': dict(sorted(stats.items())),
        'per_script': by_script,
        'not_in_corpus': skipped,
        'no_data': failed,
    }
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"\nSklad zbioru zapisany w: {path}")


def test_single_language(language="pl", samples=10):
    print(f"\nTesting single language: {language}")
    print("-" * 40)
    data = load_data(language, samples=samples, timeout=30)
    if data:
        print("\nFirst sample preview:")
        print(f"  Text: {data[0]['text'][:200]}...")
        return True
    print(f"Failed to load {language}")
    return False


def get_language_list():
    return get_available_languages()


if __name__ == "__main__":
    print("=" * 60)
    print("DATA LOADER TEST")
    print("=" * 60)

    snaps = get_snapshots()
    print(f"\nSnapshots ({len(snaps)}): {snaps[:5]}{'...' if len(snaps) > 5 else ''}")

    langs = get_available_languages()
    print(f"\nAvailable languages: {len(langs)}")
    missing = [code for code in LANGUAGES if code not in langs]
    print(f"Configured but unavailable ({len(missing)}): {missing}")

    for probe in ("en", "zh", "ja", "hi", "ar", "th"):
        test_single_language(probe, 3)
