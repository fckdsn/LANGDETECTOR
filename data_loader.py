import sys
import time

from datasets import load_dataset
from huggingface_hub import HfFileSystem

from config import HF_TOKEN, LANGUAGES

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HIGH_PRIORITY = {'pl':2000, 'cs':2000, 'sk':2000, 'hr':2000, 'sl':2000,
                 'kk':2500, 'ky':2500, 'uz':2000, 'tr':2000, 'az':2000,
                 'id':2000, 'ms':2000, 'tl':2000, 'vi':2000, 'sw':2000,
                 'be':2000, 'bg':2000, 'sr':2000, 'mk':2000}

DATASET_REPO = "oscar-corpus/community-oscar"

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
    except Exception as e:
        print(f"Error listing snapshots: {e}")
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
    except Exception as e:
        print(f"Error listing languages: {e}")
        return []


def _build_data_files(language, snapshots=None):
    if snapshots is None:
        snapshots = get_snapshots()
    return [f"data/{snap}/{language}_meta/*.jsonl.zst" for snap in snapshots]


def load_data(language, samples=1500, timeout=60, min_len=200):
    print(f"  Loading {language.upper()}...", end=" ", flush=True)
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
                streaming=True
            )

            result, start = [], time.time()
            for item in dataset:
                if len(result) >= samples or time.time() - start > timeout:
                    break
                content = item.get('content', '')
                if content and len(content) >= min_len:
                    result.append({
                        'text': content[:8000],
                        'language': language
                    })

            if result:
                print(f"OK ({len(result)}/{samples}) [{snapshot}]")
                return result
        except Exception as e:
            last_err = e
            continue

    msg = str(last_err)[:80] if last_err else "no data"
    print(f"FAILED: {msg}")
    return []


def load_multiple_languages(languages=None, samples=1200):
    if languages is None:
        languages = LANGUAGES

    available = set(get_available_languages())
    to_load = [lang for lang in languages if lang in available]
    skipped = [lang for lang in languages if lang not in available]
    if skipped:
        print(f"Skipping (not in OSCAR): {skipped}")

    print(f"\nLoading {len(to_load)} languages...")
    all_data, stats = [], {}

    for i, lang in enumerate(to_load, 1):
        print(f"[{i:3d}/{len(to_load)}] ", end="")
        data = load_data(lang, HIGH_PRIORITY.get(lang, samples))
        if data:
            all_data.extend(data)
            stats[lang] = len(data)

    print(f"\nTotal: {len(all_data)} samples, {len(stats)} languages")
    return all_data


def test_single_language(language="pl", samples=10):
    print(f"\nTesting single language: {language}")
    print("-"*40)
    data = load_data(language, samples=samples, timeout=30)
    if data:
        print("\nFirst sample preview:")
        print(f"  Text: {data[0]['text'][:200]}...")
        return True
    else:
        print(f"Failed to load {language}")
        return False


def get_language_list():
    return get_available_languages()


if __name__ == "__main__":
    print("="*60)
    print("DATA LOADER TEST")
    print("="*60)

    snaps = get_snapshots()
    print(f"\nSnapshots ({len(snaps)}): {snaps[:5]}{'...' if len(snaps) > 5 else ''}")

    langs = get_available_languages()
    print(f"\nAvailable languages: {len(langs)}")
    print(f"First 20: {langs[:20]}")

    test_single_language("en", 5)
