import os
import time
from datasets import load_dataset
from huggingface_hub import HfFileSystem
from config import HF_TOKEN, LANGUAGES

os.environ["HF_TOKEN"] = HF_TOKEN

HIGH_PRIORITY = {'pl':2000, 'cs':2000, 'sk':2000, 'hr':2000, 'sl':2000,
                 'kk':2500, 'ky':2500, 'uz':2000, 'tr':2000, 'az':2000,
                 'id':2000, 'ms':2000, 'tl':2000, 'vi':2000, 'sw':2000,
                 'be':2000, 'bg':2000, 'sr':2000, 'mk':2000}

def get_available_languages(snapshot="2024-22"):
    try:
        fs = HfFileSystem(token=HF_TOKEN)
        items = fs.ls(f"datasets/oscar-corpus/community-oscar/data/{snapshot}", detail=False)
        return sorted([i.split('/')[-1].replace('_meta', '') for i in items if '_meta' in i])
    except Exception as e:
        print(f"Error: {e}")
        return []

def load_data(language, samples=1500, timeout=40, min_len=200):
    print(f"  Loading {language.upper()}...", end=" ", flush=True)
    try:
        dataset = load_dataset("oscar-corpus/community-oscar",
            data_files=f"data/2024-22/{language}_meta/*.jsonl.zst",
            split="train", token=HF_TOKEN, streaming=True)
        
        result, start = [], time.time()
        for item in dataset:
            if len(result) >= samples or time.time() - start > timeout:
                break
            content = item.get('content', '')
            if content and len(content) >= min_len:
                result.append({'text': content[:8000], 'language': language})
        
        print(f"OK ({len(result)}/{samples})")
        return result
    except Exception as e:
        print(f"FAILED")
        return []

def load_multiple_languages(languages=None, samples=1200):
    if languages is None:
        languages = LANGUAGES
    
    available = set(get_available_languages())
    to_load = [l for l in languages if l in available]
    
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