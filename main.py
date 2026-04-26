import os
from huggingface_hub import HfFileSystem
from datasets import load_dataset
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
import numpy as np
import re
import time

HF_TOKEN = os.environ.get("HF_TOKEN", "")
os.environ["HF_TOKEN"] = HF_TOKEN

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

def get_available_languages(snapshot="2024-22"):
    print("Getting available languages...")
    fs = HfFileSystem(token=HF_TOKEN)
    snapshot_path = f"datasets/oscar-corpus/community-oscar/data/{snapshot}"
    
    try:
        items = fs.ls(snapshot_path, detail=False)
    except Exception as e:
        print(f"Error: {e}")
        return []
    
    languages = []
    for item in items:
        item_name = item.split('/')[-1]
        if '_meta' in item_name:
            lang_code = item_name.replace('_meta', '')
            languages.append(lang_code)
    
    print(f"Found {len(languages)} languages in OSCAR")
    return languages

def load_data(language="pl", samples=800, snapshot="2024-22", timeout=30):
    lang_name = LANGUAGE_INFO.get(language, {}).get('name', language)
    print(f"  Loading {language} ({lang_name})...", end=" ", flush=True)
    
    try:
        data_files = f"data/{snapshot}/{language}_meta/*.jsonl.zst"
        
        dataset = load_dataset(
            "oscar-corpus/community-oscar",
            data_files=data_files,
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
                print("TIMEOUT")
                return []
            
            if item.get('content') and len(item['content']) > 150:
                content = item['content'][:4000]
                limited_dataset.append({
                    'text': content,
                    'language': language,
                    'lang_name': lang_name,
                    'alphabet': LANGUAGE_INFO.get(language, {}).get('alphabet', 'unknown'),
                    'family': LANGUAGE_INFO.get(language, {}).get('family', 'unknown')
                })
        
        print(f"OK ({len(limited_dataset)} samples)")
        return limited_dataset
        
    except Exception as e:
        print(f"FAILED")
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
    print(" TRAINING LANGUAGE DETECTION MODEL - 50 LANGUAGES")
    print("="*70)
    
    print("\n[1] Getting available languages...")
    available_langs = get_available_languages("2024-22")
    
    if not available_langs:
        print("Cannot get language list")
        return False
    
    languages_to_load = [l for l in LANGUAGE_INFO.keys() if l in available_langs]
    languages_to_load = languages_to_load[:50]
    
    print(f"\n[2] Will load {len(languages_to_load)} languages...")
    print("-"*70)
    
    all_data = []
    failed_langs = []
    
    for i, lang in enumerate(languages_to_load, 1):
        print(f"  {i:3d}/{len(languages_to_load)} ", end="")
        samples = 800
        data = load_data(lang, samples=samples)
        
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
    
    print(f"\n[6] Training Random Forest...")
    print(f"    Training samples: {X_train.shape[0]}")
    print(f"    Test samples: {X_test.shape[0]}")
    print(f"    Features: {X_train.shape[1]}")
    
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=35,
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
    
    from sklearn.metrics import classification_report
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    avg_conf = np.mean(np.max(y_proba, axis=1))
    print(f"    Average confidence: {avg_conf:.1%}")
    
    print("\n[8] Saving models...")
    os.makedirs('models', exist_ok=True)
    
    joblib.dump(model, 'models/language_model.pkl')
    joblib.dump(label_encoder, 'models/language_encoder.pkl')
    joblib.dump(vectorizers, 'models/vectorizers.pkl')
    joblib.dump(LANGUAGE_INFO, 'models/language_info.pkl')
    
    print("    Models saved in models/ folder")
    
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
    """Automatyczny test z pliku"""
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
    
    model = joblib.load('models/language_model.pkl')
    label_encoder = joblib.load('models/language_encoder.pkl')
    vectorizers = joblib.load('models/vectorizers.pkl')
    language_info = joblib.load('models/language_info.pkl')
    
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
        bar = "#" * int(confidence * 20)
        print(f" {expected:<10} {detected:<10} {confidence:<11.1%} {status:<8} {preview}...")
    
    print("\n" + "="*70)
    print(f" SUMMARY: {correct}/{len(tests)} correct ({correct/len(tests)*100:.1f}%)")
    print(f" Average confidence: {np.mean([c for _,_,c,_,_ in results]):.1%}")
    print("="*70)

def interactive_mode():
    print("="*70)
    print(" INTERACTIVE LANGUAGE DETECTION - 50 LANGUAGES")
    print("="*70)
    
    if not os.path.exists('models/language_model.pkl'):
        print("\n No trained model found.")
        print(" Starting training (this may take 10-15 minutes)...")
        if not train_model():
            print(" Training failed")
            return
    
    print("\n Loading models...")
    model = joblib.load('models/language_model.pkl')
    label_encoder = joblib.load('models/language_encoder.pkl')
    vectorizers = joblib.load('models/vectorizers.pkl')
    
    language_info = {}
    if os.path.exists('models/language_info.pkl'):
        language_info = joblib.load('models/language_info.pkl')
    
    print(f" Model loaded: {len(label_encoder.classes_)} languages")
    print(f" Languages: {', '.join(label_encoder.classes_[:20])}...")
    
    print("\n" + "="*70)
    print(" INSTRUCTIONS:")
    print("   - Enter text in any language (MANUAL MODE)")
    print("   - Type 'auto' to run automatic test from test_languages.txt")
    print("   - Type 'languages' to see all 50 languages")
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