from datasets import load_dataset
import joblib
import os

HF_TOKEN = os.environ.get("HF_TOKEN", "")

print("Ladowanie dodatkowych danych dla jezyka polskiego...")
polish_data = load_dataset("wikipedia", "20220301.pl", split="train[:3000]")
polish_texts = [item['text'].lower().replace('\n', ' ') for item in polish_data if len(item['text']) > 200]

print(f"Dodano {len(polish_texts)} tekstow polskich")
vectorizers = joblib.load('pliki/models/vectorizers.pkl')
char_vectorizer = vectorizers[0]
word_vectorizer = vectorizers[1]

X_char = char_vectorizer.transform(polish_texts)
X_word = word_vectorizer.transform(polish_texts)
from scipy.sparse import hstack
X_new = hstack([X_char, X_word])


print("UWAGA: Random Forest nie wspiera doszkalania. Aby dodac dane, uruchom ponownie pelne trenowanie.")