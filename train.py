import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from collections import Counter
from config import OUTPUT_DIR, MODELS_DIR, LANGUAGE_INFO
import warnings
warnings.filterwarnings('ignore')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def extract_features(texts):
    print("Extracting features...")
    char_vec = TfidfVectorizer(analyzer='char', ngram_range=(2,6), max_features=20000, sublinear_tf=True)
    word_vec = TfidfVectorizer(analyzer='word', ngram_range=(1,3), max_features=6000, sublinear_tf=True, min_df=2)
    X = hstack([char_vec.fit_transform(texts), word_vec.fit_transform(texts)])
    print(f"  Features: {X.shape[1]}")
    return X, (char_vec, word_vec)

def train_language_model(X, y):
    print("\nTraining language model...")
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
    
    model = RandomForestClassifier(n_estimators=400, max_depth=30, min_samples_split=2,
                                   random_state=42, n_jobs=-1, class_weight='balanced_subsample')
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, model.predict(X_test))
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.savefig(f'{OUTPUT_DIR}/confusion_matrix_language.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    joblib.dump(model, f'{MODELS_DIR}/language_model.pkl')
    joblib.dump(label_encoder, f'{MODELS_DIR}/language_encoder.pkl')
    return model, label_encoder, accuracy

def train_alphabet_model(X, y):
    print("\nTraining alphabet model...")
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
    
    model = RandomForestClassifier(n_estimators=300, max_depth=25, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    joblib.dump(model, f'{MODELS_DIR}/alphabet_model.pkl')
    joblib.dump(label_encoder, f'{MODELS_DIR}/alphabet_encoder.pkl')
    return model, label_encoder, accuracy

def train_complete_model(texts, labels, alphabets):
    print("="*60)
    print("TRAINING LANGUAGE DETECTOR")
    print("="*60)
    
    X, vectorizers = extract_features(texts)
    
    lang_model, lang_enc, lang_acc = train_language_model(X, labels)
    alpha_model, alpha_enc, alpha_acc = train_alphabet_model(X, alphabets)
    
    joblib.dump(vectorizers, f'{MODELS_DIR}/vectorizers.pkl')
    joblib.dump(LANGUAGE_INFO, f'{MODELS_DIR}/language_info.pkl')
    
    print(f"\nResults saved to '{OUTPUT_DIR}/'")
    return {'language_accuracy': lang_acc, 'alphabet_accuracy': alpha_acc}