from config import LANGUAGES, OUTPUT_DIR
from data_loader import load_multiple_languages
from preprocess import clean_text, detect_alphabet
from train import train_complete_model


def load_and_prepare_data():
    print("Loading data...")
    data = load_multiple_languages(LANGUAGES, samples=800)
    
    texts, labels, alphabets = [], [], []
    for item in data:
        text = clean_text(item.get('text', '')[:5000])
        if len(text) > 50:
            texts.append(text)
            labels.append(item['language'])
            alphabets.append(detect_alphabet(text))
    
    print(f"Loaded {len(texts)} samples, {len(set(labels))} languages")
    return texts, labels, alphabets

def main():
    print("="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    texts, labels, alphabets = load_and_prepare_data()
    results = train_complete_model(texts, labels, alphabets)
    
    print(f"\nLanguage accuracy: {results['language_accuracy']:.2%}")
    print(f"Alphabet accuracy: {results['alphabet_accuracy']:.2%}")
    print(f"\nAll files saved to '{OUTPUT_DIR}/'")

if __name__ == "__main__":
    main()
