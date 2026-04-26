import joblib
import numpy as np
import re
from scipy.sparse import hstack
from config import MODELS_DIR, LANGUAGE_INFO
from preprocess import clean_text

class LanguageDetector:
    def __init__(self):
        self.model = joblib.load(f'{MODELS_DIR}/language_model.pkl')
        self.encoder = joblib.load(f'{MODELS_DIR}/language_encoder.pkl')
        self.char_vec, self.word_vec = joblib.load(f'{MODELS_DIR}/vectorizers.pkl')
    
    def predict(self, text):
        cleaned = clean_text(text)
        if len(cleaned) < 30:
            return None, 0, []
        
        X = hstack([self.char_vec.transform([cleaned]), self.word_vec.transform([cleaned])])
        probs = self.model.predict_proba(X)[0]
        top = np.argsort(probs)[::-1][:5]
        
        results = []
        for idx in top:
            if probs[idx] > 0.02:
                lang = self.encoder.inverse_transform([idx])[0]
                info = LANGUAGE_INFO.get(lang, {})
                results.append({'language': lang, 'name': info.get('name', lang),
                               'family': info.get('family', ''), 'confidence': probs[idx]})
        return (results[0]['language'], results[0]['confidence'], results) if results else (None, 0, [])

def main():
    print("="*50)
    print("LANGUAGE DETECTOR - INTERACTIVE")
    print("="*50)
    
    detector = LanguageDetector()
    print(f"Loaded {len(detector.encoder.classes_)} languages\n")
    
    while True:
        text = input("\n>> Enter text (or 'exit'): ").strip()
        if text.lower() == 'exit':
            break
        if len(text) < 30:
            print("Minimum 30 characters")
            continue
        
        lang, conf, preds = detector.predict(text)
        if lang:
            print(f"\nDetected: {lang.upper()} ({conf:.1%})")
            print("Top 5:")
            for i, p in enumerate(preds[:5], 1):
                print(f"  {i}. {p['language']} - {p['confidence']:.1%}")
        else:
            print("Could not detect language")

if __name__ == "__main__":
    main()