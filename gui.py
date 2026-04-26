import tkinter as tk
from tkinter import scrolledtext, messagebox
import joblib
import os
import re
import threading
import numpy as np
from scipy.sparse import hstack

class LanguageDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Detektor Jezyka i Alfabetu")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)
        self.root.configure(bg='#1a1a2e')
        
        self.model = None
        self.label_encoder = None
        self.char_vectorizer = None
        self.word_vectorizer = None
        self.language_info = {}
        
        self.create_widgets()
        self.load_models()
    
    def create_widgets(self):
        main_frame = tk.Frame(self.root, bg='#1a1a2e')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)
        
        title = tk.Label(main_frame, text="DETEKTOR JEZYKA I ALFABETU", 
                        font=('Arial', 20, 'bold'), bg='#1a1a2e', fg='#e94560')
        title.pack(pady=(0, 5))
        
        subtitle = tk.Label(main_frame, text="Rozpoznawanie jezyka i alfabetu na podstawie tekstu", 
                           font=('Arial', 10), bg='#1a1a2e', fg='#888888')
        subtitle.pack(pady=(0, 20))
        
        status_frame = tk.LabelFrame(main_frame, text=" STATUS ", font=('Arial', 10, 'bold'),
                                     bg='#16213e', fg='white', bd=1, relief=tk.GROOVE)
        status_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.status_label = tk.Label(status_frame, text="Ladowanie modeli...", 
                                     font=('Arial', 9), bg='#16213e', fg='#f39c12')
        self.status_label.pack(pady=5)
        
        self.lang_count_label = tk.Label(status_frame, text="", 
                                        font=('Arial', 9), bg='#16213e', fg='#cccccc')
        self.lang_count_label.pack(pady=(0, 5))
        
        input_frame = tk.LabelFrame(main_frame, text=" WPROWADZ TEKST ", font=('Arial', 10, 'bold'),
                                    bg='#16213e', fg='white', bd=1, relief=tk.GROOVE)
        input_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        self.text_input = scrolledtext.ScrolledText(input_frame, height=8, font=('Arial', 11), 
                                                     wrap=tk.WORD, bg='#0f3460', fg='white',
                                                     insertbackground='white', bd=0)
        self.text_input.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.text_input.insert(tk.END, "Wprowadz tekst do analizy (minimum 30 znakow)...")
        self.text_input.bind('<FocusIn>', self.on_focus_in)
        self.text_input.bind('<FocusOut>', self.on_focus_out)
        
        btn_frame = tk.Frame(main_frame, bg='#1a1a2e')
        btn_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.detect_btn = tk.Button(btn_frame, text="ANALIZUJ", 
                                   font=('Arial', 12, 'bold'), bg='#e94560', fg='white',
                                   activebackground='#c62a48', cursor='hand2', height=1, width=18,
                                   command=self.detect_language)
        self.detect_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.clear_btn = tk.Button(btn_frame, text="WYCZYSC", 
                                  font=('Arial', 10), bg='#533483', fg='white',
                                  activebackground='#3b1f6e', cursor='hand2', width=14,
                                  command=self.clear_text)
        self.clear_btn.pack(side=tk.LEFT)
        
        self.test_btn = tk.Button(btn_frame, text="TEKSTY TESTOWE", 
                                 font=('Arial', 10), bg='#0f3460', fg='white',
                                 activebackground='#0a2848', cursor='hand2', width=16,
                                 command=self.show_test_texts)
        self.test_btn.pack(side=tk.RIGHT)
        
        result_frame = tk.LabelFrame(main_frame, text=" WYNIKI ANALIZY ", font=('Arial', 10, 'bold'),
                                     bg='#16213e', fg='white', bd=1, relief=tk.GROOVE)
        result_frame.pack(fill=tk.BOTH, expand=True)
        
        self.result_text = scrolledtext.ScrolledText(result_frame, height=12, font=('Arial', 10), 
                                                      wrap=tk.WORD, bg='#0f3460', fg='white',
                                                      insertbackground='white', bd=0)
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.test_texts = {
           
    'Angielski (EN)': "The rapid advancement of artificial intelligence has transformed many industries in recent years. Machine learning algorithms can now analyze massive amounts of data and identify patterns that humans might miss. This technology is revolutionizing healthcare, finance, transportation and education worldwide.",

    'Niemiecki (DE)': "Die rasante Entwicklung der künstlichen Intelligenz hat in den letzten Jahren viele Branchen grundlegend verändert. Maschinenlernalgorithmen können heute riesige Datenmengen analysieren und Muster erkennen, die Menschen möglicherweise übersehen. Diese Technologie revolutioniert weltweit das Gesundheitswesen, Finanzwesen, Transportwesen und Bildungswesen.",

    'Hiszpański (ES)': "El rápido avance de la inteligencia artificial ha transformado muchas industrias en los últimos años. Los algoritmos de aprendizaje automático pueden analizar grandes cantidades de datos e identificar patrones que los humanos podrían pasar por alto. Esta tecnología está revolucionando la atención médica, las finanzas, el transporte y la educación en todo el mundo.",

    'Francuski (FR)': "L'intelligence artificielle a transformé de nombreux secteurs ces dernières années. Les algorithmes d'apprentissage automatique peuvent analyser d'énormes quantités de données et identifier des schémas que les humains pourraient manquer. Cette technologie révolutionne les soins de santé, la finance, les transports et l'éducation dans le monde entier.",

    'Włoski (IT)': "Il rapido progresso dell'intelligenza artificiale ha trasformato molte industrie negli ultimi anni. Gli algoritmi di apprendimento automatico possono analizzare grandi quantità di dati e identificare modelli che gli umani potrebbero trascurare. Questa tecnologia sta rivoluzionando l'assistenza sanitaria, la finanza, i trasporti e l'istruzione in tutto il mondo.",

    'Polski (PL)': "Sztuczna inteligencja zmieniła wiele branż w ostatnich latach. Algorytmy uczenia maszynowego potrafią analizować ogromne ilości danych i identyfikować wzorce, które ludzie mogą przeoczyć. Ta technologia rewolucjonizuje opiekę zdrowotną, finanse, transport i edukację na całym świecie.",

    'Rosyjski (RU)': "Быстрое развитие искусственного интеллекта преобразовало многие отрасли в последние годы. Алгоритмы машинного обучения могут анализировать огромные объемы данных и выявлять закономерности, которые люди могут пропустить. Эта технология революционизирует здравоохранение, финансы, транспорт и образование во всем мире.",

    'Ukraiński (UK)': "Швидкий розвиток штучного інтелекту перетворив багато галузей за останні роки. Алгоритми машинного навчання можуть аналізувати величезні обсяги даних і виявляти закономірності, які люди можуть пропустити. Ця технологія революціонізує охорону здоров'я, фінанси, транспорт та освіту в усьому світі.",
}
        
    
    def on_focus_in(self, event):
        if self.text_input.get("1.0", tk.END).strip() == "Wprowadz tekst do analizy (minimum 30 znakow)...":
            self.text_input.delete("1.0", tk.END)
    
    def on_focus_out(self, event):
        if not self.text_input.get("1.0", tk.END).strip():
            self.text_input.insert(tk.END, "Wprowadz tekst do analizy (minimum 30 znakow)...")
    
    def load_models(self):
        thread = threading.Thread(target=self._load_models)
        thread.daemon = True
        thread.start()
    
    def _load_models(self):
        try:
            models_path = 'pliki/models'
            
            if not os.path.exists(f'{models_path}/language_model.pkl'):
                self.root.after(0, self.show_error, "Nie znaleziono modeli. Uruchom najpierw trenowanie (python main_train.py)")
                return
            
            self.model = joblib.load(f'{models_path}/language_model.pkl')
            self.label_encoder = joblib.load(f'{models_path}/language_encoder.pkl')
            
            vectorizers = joblib.load(f'{models_path}/vectorizers.pkl')
            self.char_vectorizer = vectorizers[0]
            self.word_vectorizer = vectorizers[1]
            
            info_path = f'{models_path}/language_info.pkl'
            if os.path.exists(info_path):
                self.language_info = joblib.load(info_path)
            
            self.root.after(0, self.update_status, "Modele zaladowane", 
                           f"Rozpoznawanych jezykow: {len(self.label_encoder.classes_)}")
            
        except Exception as e:
            self.root.after(0, self.show_error, f"Blad ladowania: {str(e)}")
    
    def update_status(self, status_msg, lang_msg):
        self.status_label.config(text=status_msg, fg='#2ecc71')
        self.lang_count_label.config(text=lang_msg)
    
    def show_error(self, msg):
        messagebox.showerror("Blad", msg)
    
    def clean_text(self, text):
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)
        text = re.sub(r"\S+@\S+", "", text)
        text = re.sub(r"[^a-zа-яё0-9\s]", "", text)
        text = " ".join([w for w in text.split() if len(w) > 1])
        return text.strip()
    
    def predict(self, text):
        cleaned = self.clean_text(text)
        
        if len(cleaned) < 20:
            return None, 0, []
        
        X_char = self.char_vectorizer.transform([cleaned])
        X_word = self.word_vectorizer.transform([cleaned])
        X_combined = hstack([X_char, X_word])
        
        probs = self.model.predict_proba(X_combined)[0]
        top = np.argsort(probs)[::-1][:5]
        
        results = []
        for idx in top:
            if probs[idx] > 0.01:
                lang = self.label_encoder.inverse_transform([idx])[0]
                info = self.language_info.get(lang, {})
                results.append({
                    'language': lang,
                    'name': info.get('name', lang),
                    'family': info.get('family', 'unknown'),
                    'alphabet': info.get('alphabet', 'unknown'),
                    'confidence': probs[idx]
                })
        
        if not results:
            return None, 0, []
        
        return results[0]['language'], results[0]['confidence'], results
    
    def detect_language(self):
        text = self.text_input.get("1.0", tk.END).strip()
        
        if not text or text == "Wprowadz tekst do analizy (minimum 30 znakow)...":
            messagebox.showwarning("Uwaga", "Wprowadz tekst do analizy")
            return
        
        if len(text) < 30:
            messagebox.showwarning("Uwaga", "Tekst za krotki. Minimum 30 znakow")
            return
        
        self.detect_btn.config(state=tk.DISABLED, text="ANALIZUJE...")
        thread = threading.Thread(target=self._detect_thread, args=(text,))
        thread.daemon = True
        thread.start()
    
    def _detect_thread(self, text):
        lang, conf, preds = self.predict(text)
        self.root.after(0, self.display_results, text, lang, conf, preds)
    
    def display_results(self, text, lang, conf, preds):
        self.result_text.delete("1.0", tk.END)
        
        self.result_text.insert(tk.END, "="*70 + "\n")
        self.result_text.insert(tk.END, "WYNIK ANALIZY\n")
        self.result_text.insert(tk.END, "="*70 + "\n\n")
        
        if lang:
            self.result_text.insert(tk.END, f"WYKRYTY JEZYK: {lang.upper()}\n")
            self.result_text.insert(tk.END, f"PEWNOSC: {conf:.1%}\n\n")
            
            bar = "#" * int(conf * 30) + "-" * (30 - int(conf * 30))
            self.result_text.insert(tk.END, f"[{bar}] {conf:.1%}\n\n")
            
            self.result_text.insert(tk.END, "-"*70 + "\n")
            self.result_text.insert(tk.END, "TOP 5 PRZEWIDYWAN\n")
            self.result_text.insert(tk.END, "-"*70 + "\n")
            
            for i, p in enumerate(preds[:5], 1):
                info = f"{p['language'].upper()} - {p['confidence']:.1%}"
                if 'name' in p:
                    info = f"{p['language'].upper()} ({p['name']}) - {p['confidence']:.1%}"
                bar2 = "#" * int(p['confidence'] * 20)
                self.result_text.insert(tk.END, f"{i}. {info} [{bar2}]\n")
            
            if preds and 'family' in preds[0]:
                self.result_text.insert(tk.END, f"\nRodzina: {preds[0]['family']}")
                self.result_text.insert(tk.END, f"\nAlfabet: {preds[0]['alphabet']}")
        else:
            self.result_text.insert(tk.END, "NIE ROZPOZNANO JEZYKA\n")
            self.result_text.insert(tk.END, "Tekst za krotki lub zawiera nieobslugiwane znaki\n")
        
        preview = text[:200] + "..." if len(text) > 200 else text
        self.result_text.insert(tk.END, f"\n\nTEKST:\n{preview}\n")
        
        self.detect_btn.config(state=tk.NORMAL, text="ANALIZUJ")
    
    def clear_text(self):
        self.text_input.delete("1.0", tk.END)
        self.text_input.insert(tk.END, "Wprowadz tekst do analizy (minimum 30 znakow)...")
        self.result_text.delete("1.0", tk.END)
    
    def show_test_texts(self):
        win = tk.Toplevel(self.root)
        win.title("Teksty Testowe")
        win.geometry("550x450")
        win.configure(bg='#1a1a2e')
        
        tk.Label(win, text="Wybierz testowy tekst:", font=('Arial', 12, 'bold'),
                bg='#1a1a2e', fg='white').pack(pady=10)
        
        listbox = tk.Listbox(win, font=('Arial', 10), bg='#0f3460', fg='white',
                             selectbackground='#e94560', height=12)
        listbox.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 10))
        
        for name in self.test_texts.keys():
            listbox.insert(tk.END, name)
        
        def on_select():
            sel = listbox.curselection()
            if sel:
                name = listbox.get(sel[0])
                self.text_input.delete("1.0", tk.END)
                self.text_input.insert(tk.END, self.test_texts[name])
                win.destroy()
        
        btn_frame = tk.Frame(win, bg='#1a1a2e')
        btn_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        tk.Button(btn_frame, text="WYBIERZ", font=('Arial', 10), bg='#e94560', fg='white',
                 activebackground='#c62a48', cursor='hand2', width=12,
                 command=on_select).pack(side=tk.LEFT, padx=(0, 10))
        
        tk.Button(btn_frame, text="ANULUJ", font=('Arial', 10), bg='#533483', fg='white',
                 activebackground='#3b1f6e', cursor='hand2', width=12,
                 command=win.destroy).pack(side=tk.LEFT)

def main():
    root = tk.Tk()
    app = LanguageDetectorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()