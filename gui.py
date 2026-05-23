import os
import re
import threading
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, scrolledtext, ttk

import joblib
import numpy as np
from scipy.sparse import hstack


class LanguageDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Detektor Jezyka i Alfabetu")
        self.root.geometry("1000x800")
        self.root.minsize(900, 700)
        self.root.configure(bg='#1a1a2e')
        
        self.model = None
        self.label_encoder = None
        self.char_vectorizer = None
        self.word_vectorizer = None
        self.language_info = {}
        self.test_texts = {}
        self.models_loaded = False
        self.log_lines = []
        
        self.create_widgets()
        self.load_models_async()
    
    def create_widgets(self):
        main_frame = tk.Frame(self.root, bg='#1a1a2e')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)
        
        tk.Label(main_frame, text="DETEKTOR JEZYKA I ALFABETU", 
                font=('Arial', 20, 'bold'), bg='#1a1a2e', fg='#e94560').pack(pady=(0, 5))
        
        self.status_label = tk.Label(main_frame, text="Ladowanie modeli...", 
                font=('Arial', 10), bg='#1a1a2e', fg='#f39c12')
        self.status_label.pack(pady=(0, 10))
        
        self.tab_control = ttk.Notebook(main_frame)
        self.tab_detect = tk.Frame(self.tab_control, bg='#1a1a2e')
        self.tab_auto = tk.Frame(self.tab_control, bg='#1a1a2e')
        self.tab_log = tk.Frame(self.tab_control, bg='#1a1a2e')
        self.tab_control.add(self.tab_detect, text='  DETEKCJA  ')
        self.tab_control.add(self.tab_auto, text='  AUTO-TEST  ')
        self.tab_control.add(self.tab_log, text='  LOGI  ')
        self.tab_control.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        style = ttk.Style()
        style.theme_use('default')
        style.configure('TNotebook', background='#1a1a2e', borderwidth=0)
        style.configure('TNotebook.Tab', background='#16213e', foreground='white', padding=[15, 5], font=('Arial', 10, 'bold'))
        style.map('TNotebook.Tab', background=[('selected', '#e94560')], foreground=[('selected', 'white')])
        
        input_frame = tk.LabelFrame(self.tab_detect, text=" WPROWADZ TEKST ", font=('Arial', 10, 'bold'),
                                    bg='#16213e', fg='white', bd=1, relief=tk.GROOVE)
        input_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.text_input = scrolledtext.ScrolledText(input_frame, height=6, font=('Arial', 11), 
                                                     wrap=tk.WORD, bg='#0f3460', fg='white',
                                                     insertbackground='white', bd=0)
        self.text_input.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.text_input.insert(tk.END, "Wprowadz tekst do analizy (minimum 30 znakow)...")
        self.text_input.bind('<FocusIn>', lambda e: self.text_input.delete("1.0", tk.END) if "Wprowadz tekst" in self.text_input.get("1.0", tk.END) else None)
        
        btn_frame = tk.Frame(self.tab_detect, bg='#1a1a2e')
        btn_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.detect_btn = tk.Button(btn_frame, text="ANALIZUJ", font=('Arial', 12, 'bold'), bg='#e94560', fg='white',
                                   activebackground='#c62a48', cursor='hand2', height=1, width=18,
                                   command=self.detect_language)
        self.detect_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        tk.Button(btn_frame, text="WYCZYSC", font=('Arial', 10), bg='#533483', fg='white',
                 activebackground='#3b1f6e', cursor='hand2', width=14,
                 command=lambda: [self.text_input.delete("1.0", tk.END), self.result_text.delete("1.0", tk.END)]).pack(side=tk.LEFT)
        
        tk.Button(btn_frame, text="TEKSTY TESTOWE", font=('Arial', 10), bg='#0f3460', fg='white',
                 activebackground='#0a2848', cursor='hand2', width=16,
                 command=self.show_test_texts).pack(side=tk.RIGHT)
        
        result_frame = tk.LabelFrame(self.tab_detect, text=" WYNIKI ANALIZY ", font=('Arial', 10, 'bold'),
                                     bg='#16213e', fg='white', bd=1, relief=tk.GROOVE)
        result_frame.pack(fill=tk.BOTH, expand=True)
        
        self.result_text = scrolledtext.ScrolledText(result_frame, height=10, font=('Arial', 10), 
                                                      wrap=tk.WORD, bg='#0f3460', fg='white',
                                                      insertbackground='white', bd=0)
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        auto_frame = tk.LabelFrame(self.tab_auto, text=" AUTO-TEST Z test_languages.txt ", font=('Arial', 10, 'bold'),
                                   bg='#16213e', fg='white', bd=1, relief=tk.GROOVE)
        auto_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.auto_result_text = scrolledtext.ScrolledText(auto_frame, height=18, font=('Consolas', 9), 
                                                           wrap=tk.WORD, bg='#0f3460', fg='#00ff00',
                                                           insertbackground='white', bd=0)
        self.auto_result_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        auto_btn_frame = tk.Frame(self.tab_auto, bg='#1a1a2e')
        auto_btn_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.auto_btn = tk.Button(auto_btn_frame, text="URUCHOM AUTO-TEST", font=('Arial', 12, 'bold'), 
                                  bg='#e94560', fg='white', activebackground='#c62a48', cursor='hand2', 
                                  height=1, width=20, command=self.run_auto_test_gui)
        self.auto_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        tk.Button(auto_btn_frame, text="ZAPISZ RAPORT", font=('Arial', 10), bg='#0f3460', fg='white',
                 activebackground='#0a2848', cursor='hand2', width=16,
                 command=self.save_report).pack(side=tk.LEFT)
        
        log_frame = tk.LabelFrame(self.tab_log, text=" LOGI SYSTEMOWE ", font=('Arial', 10, 'bold'),
                                  bg='#16213e', fg='white', bd=1, relief=tk.GROOVE)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=20, font=('Consolas', 9), 
                                                   wrap=tk.WORD, bg='#0f3460', fg='#00ff00',
                                                   insertbackground='white', bd=0)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        log_btn_frame = tk.Frame(self.tab_log, bg='#1a1a2e')
        log_btn_frame.pack(fill=tk.X)
        tk.Button(log_btn_frame, text="ZAPISZ LOGI", font=('Arial', 10), bg='#0f3460', fg='white',
                 activebackground='#0a2848', cursor='hand2', width=16,
                 command=self.save_logs).pack(side=tk.LEFT)
    
    def log(self, msg):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {msg}"
        self.log_lines.append(line)
        self.root.after(0, lambda: self.log_text.insert(tk.END, line + "\n"))
        self.root.after(0, lambda: self.log_text.see(tk.END))
    
    def load_models_async(self):
        self.log("Rozpoczynanie ladowania modeli...")
        threading.Thread(target=self._load_models, daemon=True).start()
    
    def _load_models(self):
        try:
            models_path = 'pliki/models'
            if not os.path.exists(f'{models_path}/language_model.pkl'):
                self.root.after(0, lambda: self.status_label.config(text="Nie znaleziono modeli. Uruchom python main_train.py", fg='#e74c3c'))
                self.log("BLAD: Nie znaleziono modeli w pliki/models/")
                return
            
            self.model = joblib.load(f'{models_path}/language_model.pkl')
            self.label_encoder = joblib.load(f'{models_path}/language_encoder.pkl')
            vectorizers = joblib.load(f'{models_path}/vectorizers.pkl')
            self.char_vectorizer = vectorizers[0]
            self.word_vectorizer = vectorizers[1]
            
            info_path = f'{models_path}/language_info.pkl'
            if os.path.exists(info_path):
                self.language_info = joblib.load(info_path)
            
            self.load_test_texts()
            self.models_loaded = True
            
            n_langs = len(self.label_encoder.classes_)
            n_tests = len(self.test_texts)
            self.log(f"Modele zaladowane: {n_langs} jezykow, {n_tests} tekstow testowych")
            
            self.root.after(0, lambda: self.status_label.config(
                text=f"Zaladowano {n_langs} jezykow | {n_tests} tekstow testowych", 
                fg='#2ecc71'))
        except Exception as exc:
            err_short = str(exc)[:80]
            self.log(f"BLAD ladowania: {str(exc)[:100]}")
            self.root.after(
                0,
                lambda: self.status_label.config(text=f"Blad: {err_short}", fg='#e74c3c'),
            )
    
    def load_test_texts(self):
        self.test_texts = {}
        if os.path.exists('test_languages.txt'):
            with open('test_languages.txt', 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and '|' in line:
                        parts = line.split('|', 1)
                        if len(parts) == 2:
                            code = parts[0].strip()
                            text = parts[1].strip()
                            name = self.language_info.get(code, {}).get('name', code)
                            self.test_texts[f"{name} ({code.upper()})"] = text
            self.log(f"Zaladowano {len(self.test_texts)} tekstow z test_languages.txt")
    
    def predict(self, text):
        if not self.models_loaded:
            return None, 0, []
        text = re.sub(r'[^a-zа-яё0-9\s]', '', text.lower())
        text = re.sub(r'\s+', ' ', text).strip()
        if len(text) < 20:
            return None, 0, []
        
        X_char = self.char_vectorizer.transform([text])
        X_word = self.word_vectorizer.transform([text])
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
        return (results[0]['language'], results[0]['confidence'], results) if results else (None, 0, [])
    
    def detect_language(self):
        if not self.models_loaded:
            messagebox.showwarning("Uwaga", "Modele nie sa jeszcze zaladowane")
            return
        text = self.text_input.get("1.0", tk.END).strip()
        if not text or len(text) < 30:
            messagebox.showwarning("Uwaga", "Tekst za krotki. Minimum 30 znakow")
            return
        
        self.log(f"Analiza tekstu: {text[:50]}...")
        self.detect_btn.config(state=tk.DISABLED, text="ANALIZUJE...")
        threading.Thread(target=lambda: self._detect_thread(text), daemon=True).start()
    
    def _detect_thread(self, text):
        lang, conf, preds = self.predict(text)
        self.log(f"Wynik: {lang} (pewnosc: {conf:.1%})" if lang else "Nie rozpoznano jezyka")
        self.root.after(0, lambda: self.display_results(text, lang, conf, preds))
    
    def display_results(self, text, lang, conf, preds):
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, "="*70 + "\nWYNIK ANALIZY\n" + "="*70 + "\n\n")
        
        if lang:
            self.result_text.insert(tk.END, f"WYKRYTY JEZYK: {lang.upper()}\nPEWNOSC: {conf:.1%}\n\n")
            bar = "#" * int(conf * 30) + "-" * (30 - int(conf * 30))
            self.result_text.insert(tk.END, f"[{bar}] {conf:.1%}\n\n")
            self.result_text.insert(tk.END, "-"*70 + "\nTOP 5 PRZEWIDYWAN\n" + "-"*70 + "\n")
            
            for i, p in enumerate(preds[:5], 1):
                bar2 = "#" * int(p['confidence'] * 20)
                self.result_text.insert(tk.END, f"{i}. {p['language'].upper()} ({p['name']}) - {p['confidence']:.1%} [{bar2}]\n")
            
            if preds:
                self.result_text.insert(tk.END, f"\nRodzina: {preds[0]['family']}\nAlfabet: {preds[0]['alphabet']}")
        else:
            self.result_text.insert(tk.END, "NIE ROZPOZNANO JEZYKA\n")
        
        preview = text[:200] + "..." if len(text) > 200 else text
        self.result_text.insert(tk.END, f"\n\nTEKST:\n{preview}\n")
        self.detect_btn.config(state=tk.NORMAL, text="ANALIZUJ")
    
    def show_test_texts(self):
        if not self.test_texts:
            messagebox.showwarning("Uwaga", "Brak tekstow testowych. Sprawdz plik test_languages.txt")
            return
        win = tk.Toplevel(self.root)
        win.title("Teksty Testowe")
        win.geometry("550x450")
        win.configure(bg='#1a1a2e')
        
        tk.Label(win, text="Wybierz testowy tekst:", font=('Arial', 12, 'bold'), bg='#1a1a2e', fg='white').pack(pady=10)
        
        listbox = tk.Listbox(win, font=('Arial', 10), bg='#0f3460', fg='white', selectbackground='#e94560', height=12)
        listbox.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 10))
        for name in sorted(self.test_texts.keys()):
            listbox.insert(tk.END, name)
        
        def on_select():
            sel = listbox.curselection()
            if sel:
                self.text_input.delete("1.0", tk.END)
                self.text_input.insert(tk.END, self.test_texts[listbox.get(sel[0])])
                win.destroy()
        
        frame = tk.Frame(win, bg='#1a1a2e')
        frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        tk.Button(frame, text="WYBIERZ", font=('Arial', 10), bg='#e94560', fg='white', cursor='hand2', width=12, command=on_select).pack(side=tk.LEFT, padx=(0, 10))
        tk.Button(frame, text="ANULUJ", font=('Arial', 10), bg='#533483', fg='white', cursor='hand2', width=12, command=win.destroy).pack(side=tk.LEFT)
    
    def run_auto_test_gui(self):
        if not self.models_loaded or not self.test_texts:
            messagebox.showwarning("Uwaga", "Modele lub teksty testowe nie sa gotowe")
            return
        self.log(f"Uruchamianie auto-testu dla {len(self.test_texts)} jezykow...")
        self.auto_btn.config(state=tk.DISABLED, text="TESTOWANIE...")
        self.auto_result_text.delete("1.0", tk.END)
        self.auto_result_text.insert(tk.END, "Uruchamianie auto-testu...\n\n")
        threading.Thread(target=self._run_auto_test_thread, daemon=True).start()
    
    def _run_auto_test_thread(self):
        results = []
        correct = 0
        for name, text in self.test_texts.items():
            expected = name.split('(')[1].split(')')[0].lower()
            detected, confidence, _ = self.predict(text)
            is_correct = (detected == expected)
            if is_correct:
                correct += 1
            results.append((name, expected, detected, confidence, is_correct))
        
        self.log(f"Auto-test zakonczony: {correct}/{len(results)} poprawnych ({correct/len(results)*100:.1f}%)")
        self.root.after(0, lambda: self.display_auto_results(results, correct, len(results)))
    
    def display_auto_results(self, results, correct, total):
        self.auto_result_text.delete("1.0", tk.END)
        
        header = f"{'='*90}\n  AUTO-TEST: {correct}/{total} POPRAWNYCH ({correct/total*100:.1f}%)\n{'='*90}\n\n"
        self.auto_result_text.insert(tk.END, header)
        self.auto_result_text.insert(tk.END, f"  {'Jezyk':<30} {'Oczek':<6} {'Wykryty':<6} {'Pewnosc':<10} {'Status':<6}\n  {'-'*80}\n")
        
        for name, expected, detected, confidence, is_correct in results:
            status = "OK" if is_correct else "FAIL"
            self.auto_result_text.insert(tk.END, f"  {name:<30} {expected:<6} {detected:<6} {confidence:<9.1%} {status:<6}\n")
        
        summary = f"\n  {'='*90}\n  PODSUMOWANIE: {correct}/{total} ({correct/total*100:.1f}%)\n"
        self.auto_result_text.insert(tk.END, summary)
        
        failed = [(n, e, d, c) for n, e, d, c, ok in results if not ok]
        if failed:
            self.auto_result_text.insert(tk.END, f"\n  Niepoprawne ({len(failed)}):\n")
            for name, expected, detected, confidence in failed:
                self.auto_result_text.insert(tk.END, f"    - {name}: wykryto '{detected}' ({confidence:.1%})\n")
        
        self.auto_btn.config(state=tk.NORMAL, text="URUCHOM AUTO-TEST")
    
    def generate_report_text(self):
        report = []
        report.append("="*70)
        report.append("RAPORT - DETEKTOR JEZYKA I ALFABETU")
        report.append("="*70)
        report.append(f"Data wygenerowania: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Liczba zaladowanych jezykow: {len(self.label_encoder.classes_) if self.label_encoder else 'N/A'}")
        report.append(f"Liczba tekstow testowych: {len(self.test_texts)}")
        report.append("="*70)
        report.append("")
        
        if self.auto_result_text.get("1.0", tk.END).strip():
            report.append("--- WYNIKI AUTO-TESTU ---")
            report.append(self.auto_result_text.get("1.0", tk.END).strip())
            report.append("")
        
        report.append("--- LOGI SYSTEMOWE ---")
        for log_line in self.log_lines:
            report.append(log_line)
        
        report.append("")
        report.append("="*70)
        report.append("KONIEC RAPORTU")
        
        return "\n".join(report)
    
    def save_report(self):
        content = self.auto_result_text.get("1.0", tk.END).strip()
        if not content or "AUTO-TEST" not in content:
            messagebox.showwarning("Uwaga", "Najpierw uruchom auto-test")
            return
        
        filename = f"raport_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            initialfile=filename,
            filetypes=[("Pliki tekstowe", "*.txt"), ("Wszystkie pliki", "*.*")]
        )
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(self.generate_report_text())
            self.log(f"Raport zapisany: {filepath}")
            messagebox.showinfo("OK", f"Raport zapisany:\n{filepath}")
    
    def save_logs(self):
        filename = f"logi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            initialfile=filename,
            filetypes=[("Pliki tekstowe", "*.txt"), ("Wszystkie pliki", "*.*")]
        )
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                for line in self.log_lines:
                    f.write(line + "\n")
            self.log(f"Logi zapisane: {filepath}")
            messagebox.showinfo("OK", f"Logi zapisane:\n{filepath}")

if __name__ == "__main__":
    root = tk.Tk()
    LanguageDetectorGUI(root)
    root.mainloop()