import os
import sys

import joblib
from datasets import load_dataset
from huggingface_hub import HfFileSystem

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import re
import time

import numpy as np
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATASET_REPO = "oscar-corpus/community-oscar"
_SNAPSHOT_CACHE = None
_LANG_CACHE = None

HF_TOKEN = "HF_TOKEN"
os.environ["HF_TOKEN"] = HF_TOKEN

LANGUAGES = [
    'en', 'de', 'nl', 'sv', 'da', 'no', 'af', 'is', 'fy',
    'es', 'fr', 'it', 'pt', 'ca', 'ro', 'gl',
    'pl', 'cs', 'sk', 'sl', 'hr',
    'ru', 'uk', 'bg', 'sr', 'mk', 'be',
    'cv', 'ba', 'os', 'tt', 'sah', 'av', 'ce',
    'kbd', 'ady', 'mhr', 'mrj', 'udm', 'kv', 'xal', 'myv', 'mdf', 'koi',
    'kk', 'ky', 'uz', 'tr', 'az', 'tk',
    'fi', 'hu', 'et', 'lv', 'lt',
    'sq', 'mt',
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
    'is': {'family': 'Germanic', 'alphabet': 'latin', 'name': 'Icelandic'},
    'fy': {'family': 'Germanic', 'alphabet': 'latin', 'name': 'Frisian'},
    'es': {'family': 'Romance', 'alphabet': 'latin', 'name': 'Spanish'},
    'fr': {'family': 'Romance', 'alphabet': 'latin', 'name': 'French'},
    'it': {'family': 'Romance', 'alphabet': 'latin', 'name': 'Italian'},
    'pt': {'family': 'Romance', 'alphabet': 'latin', 'name': 'Portuguese'},
    'ca': {'family': 'Romance', 'alphabet': 'latin', 'name': 'Catalan'},
    'ro': {'family': 'Romance', 'alphabet': 'latin', 'name': 'Romanian'},
    'gl': {'family': 'Romance', 'alphabet': 'latin', 'name': 'Galician'},
    'pl': {'family': 'Slavic', 'alphabet': 'latin', 'name': 'Polish'},
    'cs': {'family': 'Slavic', 'alphabet': 'latin', 'name': 'Czech'},
    'sk': {'family': 'Slavic', 'alphabet': 'latin', 'name': 'Slovak'},
    'sl': {'family': 'Slavic', 'alphabet': 'latin', 'name': 'Slovenian'},
    'hr': {'family': 'Slavic', 'alphabet': 'latin', 'name': 'Croatian'},
    'ru': {'family': 'Slavic', 'alphabet': 'cyrillic', 'name': 'Russian'},
    'uk': {'family': 'Slavic', 'alphabet': 'cyrillic', 'name': 'Ukrainian'},
    'bg': {'family': 'Slavic', 'alphabet': 'cyrillic', 'name': 'Bulgarian'},
    'sr': {'family': 'Slavic', 'alphabet': 'cyrillic', 'name': 'Serbian'},
    'mk': {'family': 'Slavic', 'alphabet': 'cyrillic', 'name': 'Macedonian'},
    'be': {'family': 'Slavic', 'alphabet': 'cyrillic', 'name': 'Belarusian'},
    'cv': {'family': 'Turkic', 'alphabet': 'cyrillic', 'name': 'Chuvash'},
    'ba': {'family': 'Turkic', 'alphabet': 'cyrillic', 'name': 'Bashkir'},
    'os': {'family': 'Iranian', 'alphabet': 'cyrillic', 'name': 'Ossetian'},
    'tt': {'family': 'Turkic', 'alphabet': 'cyrillic', 'name': 'Tatar'},
    'sah': {'family': 'Turkic', 'alphabet': 'cyrillic', 'name': 'Yakut'},
    'av': {'family': 'Caucasian', 'alphabet': 'cyrillic', 'name': 'Avar'},
    'ce': {'family': 'Caucasian', 'alphabet': 'cyrillic', 'name': 'Chechen'},
    'kbd': {'family': 'Caucasian', 'alphabet': 'cyrillic', 'name': 'Kabardian'},
    'ady': {'family': 'Caucasian', 'alphabet': 'cyrillic', 'name': 'Adyghe'},
    'mhr': {'family': 'Finno-Ugric', 'alphabet': 'cyrillic', 'name': 'Mari'},
    'mrj': {'family': 'Finno-Ugric', 'alphabet': 'cyrillic', 'name': 'Hill Mari'},
    'udm': {'family': 'Finno-Ugric', 'alphabet': 'cyrillic', 'name': 'Udmurt'},
    'kv': {'family': 'Finno-Ugric', 'alphabet': 'cyrillic', 'name': 'Komi'},
    'xal': {'family': 'Mongolic', 'alphabet': 'cyrillic', 'name': 'Kalmyk'},
    'myv': {'family': 'Finno-Ugric', 'alphabet': 'cyrillic', 'name': 'Erzya'},
    'mdf': {'family': 'Finno-Ugric', 'alphabet': 'cyrillic', 'name': 'Moksha'},
    'koi': {'family': 'Finno-Ugric', 'alphabet': 'cyrillic', 'name': 'Komi-Permyak'},
    'kk': {'family': 'Turkic', 'alphabet': 'cyrillic', 'name': 'Kazakh'},
    'ky': {'family': 'Turkic', 'alphabet': 'cyrillic', 'name': 'Kyrgyz'},
    'uz': {'family': 'Turkic', 'alphabet': 'latin', 'name': 'Uzbek'},
    'tr': {'family': 'Turkic', 'alphabet': 'latin', 'name': 'Turkish'},
    'az': {'family': 'Turkic', 'alphabet': 'latin', 'name': 'Azerbaijani'},
    'tk': {'family': 'Turkic', 'alphabet': 'latin', 'name': 'Turkmen'},
    'fi': {'family': 'Finno-Ugric', 'alphabet': 'latin', 'name': 'Finnish'},
    'hu': {'family': 'Finno-Ugric', 'alphabet': 'latin', 'name': 'Hungarian'},
    'et': {'family': 'Finno-Ugric', 'alphabet': 'latin', 'name': 'Estonian'},
    'lv': {'family': 'Baltic', 'alphabet': 'latin', 'name': 'Latvian'},
    'lt': {'family': 'Baltic', 'alphabet': 'latin', 'name': 'Lithuanian'},
    'sq': {'family': 'Albanian', 'alphabet': 'latin', 'name': 'Albanian'},
    'mt': {'family': 'Semitic', 'alphabet': 'latin', 'name': 'Maltese'},
    'id': {'family': 'Austronesian', 'alphabet': 'latin', 'name': 'Indonesian'},
    'ms': {'family': 'Austronesian', 'alphabet': 'latin', 'name': 'Malay'},
    'tl': {'family': 'Austronesian', 'alphabet': 'latin', 'name': 'Tagalog'},
    'vi': {'family': 'Austroasiatic', 'alphabet': 'latin', 'name': 'Vietnamese'},
    'sw': {'family': 'Bantu', 'alphabet': 'latin', 'name': 'Swahili'},
    'sh': {'family': 'Slavic', 'alphabet': 'latin', 'name': 'Serbo-Croatian'},
}

HIGH_PRIORITY = {
    'pl': 2000, 'cs': 2000, 'sk': 2000, 'hr': 2000, 'sl': 2000,
    'kk': 3000, 'ky': 3000, 'uz': 2000, 'tr': 2000, 'az': 2000, 'tk': 2000,
    'cv': 2000, 'ba': 2000, 'os': 2000, 'tt': 2000, 'sah': 2000,
    'av': 2000, 'ce': 2000,
    'kbd': 2000, 'ady': 2000, 'mhr': 2000, 'mrj': 2000, 'udm': 2000,
    'kv': 2000, 'xal': 2000, 'myv': 2000, 'mdf': 2000, 'koi': 2000,
    'id': 2000, 'ms': 2000, 'tl': 2000, 'vi': 2000, 'sw': 2000,
    'be': 2000, 'bg': 2000, 'sr': 2000, 'mk': 2000,
    'ro': 2000, 'gl': 2000, 'is': 2000, 'fy': 2000,
    'fi': 2000, 'hu': 2000, 'et': 2000, 'lv': 2000, 'lt': 2000,
    'sq': 2000, 'mt': 2000, 'sh': 2000,
}

def get_snapshots():
    global _SNAPSHOT_CACHE
    if _SNAPSHOT_CACHE is not None:
        return _SNAPSHOT_CACHE
    try:
        fs = HfFileSystem(token=HF_TOKEN)
        items = fs.ls(f"datasets/{DATASET_REPO}/data", detail=False)
        snaps = sorted({i.split('/')[-1] for i in items if '/' in i})
        snaps = [s for s in snaps if s and not s.startswith('.')]
        snaps.sort(reverse=True)
        _SNAPSHOT_CACHE = snaps
        return snaps
    except Exception as e:
        print(f"Error listing snapshots: {e}")
        return []

def get_available_languages():
    global _LANG_CACHE
    if _LANG_CACHE is not None:
        return _LANG_CACHE
    print("Getting available languages...")
    snaps = get_snapshots()
    if not snaps:
        return []
    snapshot = snaps[0]
    print(f"Using snapshot: {snapshot}")
    try:
        fs = HfFileSystem(token=HF_TOKEN)
        items = fs.ls(f"datasets/{DATASET_REPO}/data/{snapshot}", detail=False)
        langs = sorted({i.split('/')[-1][:-len('_meta')] for i in items if i.split('/')[-1].endswith('_meta')})
        print(f"Found {len(langs)} languages in OSCAR (snapshot {snapshot})")
        _LANG_CACHE = langs
        return langs
    except Exception as e:
        print(f"Error: {e}")
        return []

def load_data(language="pl", samples=800, timeout=30):
    lang_name = LANGUAGE_INFO.get(language, {}).get('name', language)
    print(f"  Loading {language} ({lang_name})...", end=" ", flush=True)
    
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
            
            limited_dataset = []
            start_time = time.time()
            
            for i, item in enumerate(dataset):
                if i >= samples:
                    break
                
                if time.time() - start_time > timeout:
                    break
                
                if item.get('content') and len(item['content']) > 150:
                    content = item['content'][:4000]
                    limited_dataset.append({
                        'text': content,
                        'language': language,
                        'lang_name': lang_name,
                        'alphabet': LANGUAGE_INFO.get(language, {}).get('alphabet', 'unknown'),
                        'family': LANGUAGE_INFO.get(language, {}).get('family', 'unknown')
                    })
            
            if limited_dataset:
                print(f"OK ({len(limited_dataset)} samples) [{snapshot}]")
                return limited_dataset
        except Exception as e:
            last_err = e
            continue
    
    msg = str(last_err)[:60] if last_err else "no data"
    print(f"FAILED: {msg}")
    return []

def extract_features(texts, max_char_features=20000, max_word_features=6000):
    print("  Extracting char features (2-6 n-grams)...")
    char_vectorizer = TfidfVectorizer(
        analyzer='char',
        ngram_range=(2, 6),
        max_features=max_char_features,
        sublinear_tf=True,
        lowercase=True
    )
    X_char = char_vectorizer.fit_transform(texts)
    print(f"    Char features: {X_char.shape[1]}")
    
    print("  Extracting word features (1-3 n-grams)...")
    word_vectorizer = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 3),
        max_features=max_word_features,
        sublinear_tf=True,
        lowercase=True,
        min_df=2
    )
    X_word = word_vectorizer.fit_transform(texts)
    print(f"    Word features: {X_word.shape[1]}")
    
    X_combined = hstack([X_char, X_word])
    print(f"  Total features: {X_combined.shape[1]}")
    
    return X_combined, (char_vectorizer, word_vectorizer)

def train_model():
    print("="*70)
    print(" TRAINING LANGUAGE DETECTION MODEL")
    print("="*70)
    
    print("\n[1] Getting available languages...")
    available_langs = get_available_languages()
    
    if not available_langs:
        print("Cannot get language list")
        return False
    
    languages_to_load = [lang for lang in LANGUAGES if lang in available_langs]
    
    print(f"\n[2] Will load {len(languages_to_load)} languages...")
    print("-"*70)
    
    all_data = []
    failed_langs = []
    
    for i, lang in enumerate(languages_to_load, 1):
        print(f"  {i:3d}/{len(languages_to_load)} ", end="")
        data = load_data(lang, HIGH_PRIORITY.get(lang, 800))
        
        if data:
            all_data.extend(data)
        else:
            failed_langs.append(lang)
    
    print("\n[3] Loading summary:")
    print(f"    Total samples: {len(all_data)}")
    print(f"    Successful languages: {len(languages_to_load) - len(failed_langs)}")
    print(f"    Failed languages: {len(failed_langs)}")
    
    if len(all_data) < 2000:
        print("Not enough data for training")
        return False
    
    texts = [item['text'] for item in all_data]
    labels = [item['language'] for item in all_data]
    
    print("\n[4] Extracting features...")
    X, vectorizers = extract_features(texts)
    
    print("\n[5] Preparing data...")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    
    print(f"    Number of language classes: {len(label_encoder.classes_)}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("\n[6] Training Random Forest...")
    print(f"    Training samples: {X_train.shape[0]}")
    print(f"    Test samples: {X_test.shape[0]}")
    print(f"    Features: {X_train.shape[1]}")
    
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=50,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced_subsample'
    )
    
    model.fit(X_train, y_train)
    
    print("\n[7] Evaluating model...")
    accuracy = model.score(X_test, y_test)
    print(f"    Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    y_proba = model.predict_proba(X_test)
    avg_conf = np.mean(np.max(y_proba, axis=1))
    print(f"    Average confidence: {avg_conf:.1%}")
    
    print("\n[8] Saving models...")
    os.makedirs('pliki/models', exist_ok=True)
    
    joblib.dump(model, 'pliki/models/language_model.pkl')
    joblib.dump(label_encoder, 'pliki/models/language_encoder.pkl')
    joblib.dump(vectorizers, 'pliki/models/vectorizers.pkl')
    joblib.dump(LANGUAGE_INFO, 'pliki/models/language_info.pkl')
    
    print("    Models saved in pliki/models/ folder")
    
    return True

def predict_language(text, model, label_encoder, vectorizers, language_info=None):
    char_vectorizer, word_vectorizer = vectorizers
    
    text_clean = re.sub(r'[^a-zа-яё0-9\s]', '', text.lower())
    text_clean = re.sub(r'\s+', ' ', text_clean).strip()
    
    if len(text_clean) < 20:
        return "unknown", 0.0, []
    
    X_char = char_vectorizer.transform([text_clean])
    X_word = word_vectorizer.transform([text_clean])
    X_combined = hstack([X_char, X_word])
    
    probs = model.predict_proba(X_combined)[0]
    top_indices = np.argsort(probs)[::-1][:5]
    
    results = []
    for idx in top_indices:
        if probs[idx] > 0.01:
            language = label_encoder.inverse_transform([idx])[0]
            confidence = probs[idx]
            
            info = language_info.get(language, {}) if language_info else {}
            
            results.append({
                'language': language,
                'name': info.get('name', language),
                'family': info.get('family', 'unknown'),
                'alphabet': info.get('alphabet', 'unknown'),
                'confidence': confidence
            })
    
    if not results:
        return "unknown", 0.0, []
    
    return results[0]['language'], results[0]['confidence'], results

def run_auto_test(test_file='test_languages.txt'):
    print("\n" + "="*70)
    print(" AUTO TEST - LOADING TESTS FROM FILE")
    print("="*70)
    
    if not os.path.exists(test_file):
        print(f"\n File {test_file} not found!")
        print(" Create file with format: language_code|text")
        return
    
    with open(test_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    tests = []
    for line in lines:
        line = line.strip()
        if line and '|' in line:
            parts = line.split('|', 1)
            if len(parts) == 2:
                expected = parts[0].strip()
                text = parts[1].strip()
                tests.append((expected, text))
    
    if not tests:
        print(" No valid tests found in file")
        return
    
    print(f"\n Found {len(tests)} test cases")
    print("-"*70)
    
    models_path = 'pliki/models'
    
    if not os.path.exists(f'{models_path}/language_model.pkl'):
        print(" Models not found. Please train model first.")
        return
    
    model = joblib.load(f'{models_path}/language_model.pkl')
    label_encoder = joblib.load(f'{models_path}/language_encoder.pkl')
    vectorizers = joblib.load(f'{models_path}/vectorizers.pkl')
    language_info = joblib.load(f'{models_path}/language_info.pkl')
    
    results = []
    correct = 0
    
    for expected, text in tests:
        detected, confidence, _ = predict_language(text, model, label_encoder, vectorizers, language_info)
        is_correct = (detected == expected)
        if is_correct:
            correct += 1
        results.append((expected, detected, confidence, is_correct, text[:80]))
    
    print(f"\n {'Expected':<10} {'Detected':<10} {'Confidence':<12} {'Status':<8} {'Text preview'}")
    print("-"*70)
    
    for expected, detected, confidence, is_correct, preview in results:
        status = "OK" if is_correct else "FAIL"
        print(f" {expected:<10} {detected:<10} {confidence:<11.1%} {status:<8} {preview}...")
    
    print("\n" + "="*70)
    print(f" SUMMARY: {correct}/{len(tests)} correct ({correct/len(tests)*100:.1f}%)")
    print(f" Average confidence: {np.mean([c for _,_,c,_,_ in results]):.1%}")
    print("="*70)

def interactive_mode():
    print("="*70)
    print(" INTERACTIVE LANGUAGE DETECTION")
    print("="*70)
    
    models_path = 'pliki/models'
    
    if not os.path.exists(f'{models_path}/language_model.pkl'):
        print("\n No trained model found.")
        print(" Starting training (this may take 10-15 minutes)...")
        if not train_model():
            print(" Training failed")
            return
    
    print("\n Loading models...")
    model = joblib.load(f'{models_path}/language_model.pkl')
    label_encoder = joblib.load(f'{models_path}/language_encoder.pkl')
    vectorizers = joblib.load(f'{models_path}/vectorizers.pkl')
    
    language_info = {}
    if os.path.exists(f'{models_path}/language_info.pkl'):
        language_info = joblib.load(f'{models_path}/language_info.pkl')
    
    print(f" Model loaded: {len(label_encoder.classes_)} languages")
    print(f" Languages: {', '.join(list(label_encoder.classes_)[:20])}...")
    
    print("\n" + "="*70)
    print(" INSTRUCTIONS:")
    print("   - Enter text in any language (MANUAL MODE)")
    print("   - Type 'auto' to run automatic test from test_languages.txt")
    print("   - Type 'languages' to see all languages")
    print("   - Type 'exit' to quit")
    print("   - Minimum 30 characters for manual mode")
    print("-"*70)
    
    while True:
        print()
        text = input(" >> Enter text (or command): ").strip()
        
        if text.lower() in ['exit', 'quit', 'exit()']:
            print("\n Goodbye!")
            break
        
        if text.lower() == 'languages':
            print(f"\n Supported languages ({len(label_encoder.classes_)}):")
            langs_list = list(label_encoder.classes_)
            for i in range(0, len(langs_list), 10):
                print(f"   {', '.join(langs_list[i:i+10])}")
            continue
        
        if text.lower() == 'auto':
            run_auto_test('test_languages.txt')
            continue
        
        if not text:
            print(" Please enter text")
            continue
        
        if len(text) < 30:
            print(" Text too short. Minimum 30 characters for better accuracy.")
            print(" Or use 'auto' command for automatic testing.")
            continue
        
        language, confidence, predictions = predict_language(
            text, model, label_encoder, vectorizers, language_info
        )
        
        print("\n" + "-"*70)
        print(f" DETECTED LANGUAGE: {language.upper()} ({language_info.get(language, {}).get('name', language)})")
        print(f" Confidence: {confidence:.1%}")
        print(f" Alphabet: {language_info.get(language, {}).get('alphabet', 'unknown')}")
        print(f" Family: {language_info.get(language, {}).get('family', 'unknown')}")
        
        print("\n Top 5 predictions:")
        for i, pred in enumerate(predictions[:5], 1):
            bar = "#" * int(pred['confidence'] * 30)
            empty = "." * (30 - int(pred['confidence'] * 30))
            print(f"   {i}. {pred['language']} ({pred['name']}) - {pred['confidence']:.1%} [{bar}{empty}]")
        
        print(f"\n Text preview: {text[:100]}{'...' if len(text) > 100 else ''}")
        print("-"*70)

if __name__ == "__main__":
    interactive_mode()