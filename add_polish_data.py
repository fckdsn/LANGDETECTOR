import os

import joblib
from datasets import load_dataset
from scipy.sparse import hstack

HF_TOKEN = "HF_TOKEN"
os.environ["HF_TOKEN"] = HF_TOKEN

print("Ladowanie dodatkowych danych dla jezyka polskiego...")
polish_data = load_dataset("wikipedia", "20220301.pl", split="train[:3000]")
polish_texts = [item['text'].lower().replace('\n', ' ') for item in polish_data if len(item['text']) > 200]

print(f"Dodano {len(polish_texts)} tekstow polskich")

vectorizers = joblib.load('pliki/models/vectorizers.pkl')
char_vectorizer = vectorizers[0]
word_vectorizer = vectorizers[1]

X_char = char_vectorizer.transform(polish_texts)
X_word = word_vectorizer.transform(polish_texts)
X_new = hstack([X_char, X_word])

print(f"Nowe dane: {X_new.shape}")
print("Aby dodac dane, uruchom ponownie pelne trenowanie z wieksza liczba probek dla polskiego.")