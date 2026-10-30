"""Graficzny interfejs detektora jezyka i pisma (Tkinter)."""

import threading
import tkinter as tk
from collections import defaultdict
from datetime import datetime
from tkinter import filedialog, messagebox, scrolledtext, ttk

from config import language_name, language_script, script_name
from predict import LanguageDetector, ModelsNotFoundError, read_test_cases

BG = '#1a1a2e'
PANEL = '#16213e'
FIELD = '#0f3460'
ACCENT = '#e94560'
ACCENT_DARK = '#c62a48'
PURPLE = '#533483'


class LanguageDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Detektor Jezyka i Alfabetu")
        self.root.geometry("1060x840")
        self.root.minsize(940, 720)
        self.root.configure(bg=BG)

        self.detector = None
        self.test_cases = []
        self.log_lines = []

        self.create_widgets()
        self.load_models_async()

    # --- Budowa interfejsu ---------------------------------------------------

    def create_widgets(self):
        main_frame = tk.Frame(self.root, bg=BG)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)

        tk.Label(main_frame, text="DETEKTOR JEZYKA I ALFABETU",
                 font=('Arial', 20, 'bold'), bg=BG, fg=ACCENT).pack(pady=(0, 5))

        self.status_label = tk.Label(main_frame, text="Ladowanie modeli...",
                                     font=('Arial', 10), bg=BG, fg='#f39c12')
        self.status_label.pack(pady=(0, 10))

        style = ttk.Style()
        style.theme_use('default')
        style.configure('TNotebook', background=BG, borderwidth=0)
        style.configure('TNotebook.Tab', background=PANEL, foreground='white',
                        padding=[15, 5], font=('Arial', 10, 'bold'))
        style.map('TNotebook.Tab', background=[('selected', ACCENT)],
                  foreground=[('selected', 'white')])

        self.tab_control = ttk.Notebook(main_frame)
        self.tab_detect = tk.Frame(self.tab_control, bg=BG)
        self.tab_auto = tk.Frame(self.tab_control, bg=BG)
        self.tab_models = tk.Frame(self.tab_control, bg=BG)
        self.tab_log = tk.Frame(self.tab_control, bg=BG)
        self.tab_control.add(self.tab_detect, text='  DETEKCJA  ')
        self.tab_control.add(self.tab_auto, text='  AUTO-TEST  ')
        self.tab_control.add(self.tab_models, text='  MODELE  ')
        self.tab_control.add(self.tab_log, text='  LOGI  ')
        self.tab_control.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self._build_detect_tab()
        self._build_auto_tab()
        self._build_models_tab()
        self._build_log_tab()

    def _panel(self, parent, title):
        frame = tk.LabelFrame(parent, text=title, font=('Arial', 10, 'bold'),
                              bg=PANEL, fg='white', bd=1, relief=tk.GROOVE)
        frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        return frame

    def _text_area(self, parent, height, font=('Arial', 10), fg='white'):
        area = scrolledtext.ScrolledText(parent, height=height, font=font, wrap=tk.WORD,
                                         bg=FIELD, fg=fg, insertbackground='white', bd=0)
        area.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        return area

    def _build_detect_tab(self):
        input_frame = self._panel(self.tab_detect, " WPROWADZ TEKST ")
        self.text_input = self._text_area(input_frame, 6, ('Arial', 11))
        placeholder = "Wprowadz tekst do analizy (dowolny jezyk i alfabet)..."
        self.text_input.insert(tk.END, placeholder)
        self.text_input.bind(
            '<FocusIn>',
            lambda e: self.text_input.delete("1.0", tk.END)
            if placeholder[:20] in self.text_input.get("1.0", tk.END) else None,
        )

        btn_frame = tk.Frame(self.tab_detect, bg=BG)
        btn_frame.pack(fill=tk.X, pady=(0, 10))

        self.detect_btn = tk.Button(btn_frame, text="ANALIZUJ", font=('Arial', 12, 'bold'),
                                    bg=ACCENT, fg='white', activebackground=ACCENT_DARK,
                                    cursor='hand2', width=18, command=self.detect_language)
        self.detect_btn.pack(side=tk.LEFT, padx=(0, 10))

        tk.Button(btn_frame, text="WYCZYSC", font=('Arial', 10), bg=PURPLE, fg='white',
                  activebackground='#3b1f6e', cursor='hand2', width=14,
                  command=self.clear_input).pack(side=tk.LEFT)

        tk.Button(btn_frame, text="TEKSTY TESTOWE", font=('Arial', 10), bg=FIELD, fg='white',
                  activebackground='#0a2848', cursor='hand2', width=16,
                  command=self.show_test_texts).pack(side=tk.RIGHT)

        result_frame = self._panel(self.tab_detect, " WYNIKI ANALIZY ")
        self.result_text = self._text_area(result_frame, 12, ('Consolas', 10))

    def _build_auto_tab(self):
        auto_frame = self._panel(self.tab_auto, " AUTO-TEST Z test_languages.txt ")
        self.auto_result_text = self._text_area(auto_frame, 18, ('Consolas', 9), '#00ff00')

        auto_btn_frame = tk.Frame(self.tab_auto, bg=BG)
        auto_btn_frame.pack(fill=tk.X, pady=(0, 5))

        self.auto_btn = tk.Button(auto_btn_frame, text="URUCHOM AUTO-TEST",
                                  font=('Arial', 12, 'bold'), bg=ACCENT, fg='white',
                                  activebackground=ACCENT_DARK, cursor='hand2', width=20,
                                  command=self.run_auto_test_gui)
        self.auto_btn.pack(side=tk.LEFT, padx=(0, 10))

        tk.Button(auto_btn_frame, text="ZAPISZ RAPORT", font=('Arial', 10), bg=FIELD,
                  fg='white', activebackground='#0a2848', cursor='hand2', width=16,
                  command=self.save_report).pack(side=tk.LEFT)

    def _build_models_tab(self):
        models_frame = self._panel(self.tab_models, " POROWNANIE ALGORYTMOW UCZENIA MASZYNOWEGO ")
        self.models_text = self._text_area(models_frame, 20, ('Consolas', 9))

    def _build_log_tab(self):
        log_frame = self._panel(self.tab_log, " LOGI SYSTEMOWE ")
        self.log_text = self._text_area(log_frame, 20, ('Consolas', 9), '#00ff00')

        log_btn_frame = tk.Frame(self.tab_log, bg=BG)
        log_btn_frame.pack(fill=tk.X)
        tk.Button(log_btn_frame, text="ZAPISZ LOGI", font=('Arial', 10), bg=FIELD, fg='white',
                  activebackground='#0a2848', cursor='hand2', width=16,
                  command=self.save_logs).pack(side=tk.LEFT)

    # --- Ladowanie modeli ----------------------------------------------------

    def log(self, msg):
        line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
        self.log_lines.append(line)
        self.root.after(0, lambda: self.log_text.insert(tk.END, line + "\n"))
        self.root.after(0, lambda: self.log_text.see(tk.END))

    def load_models_async(self):
        self.log("Rozpoczynanie ladowania modeli...")
        threading.Thread(target=self._load_models, daemon=True).start()

    def _load_models(self):
        try:
            self.detector = LanguageDetector()
        except ModelsNotFoundError as exc:
            self.log(f"BLAD: {exc}")
            self.root.after(0, lambda: self.status_label.config(
                text="Nie znaleziono modeli. Uruchom: python main.py train", fg='#e74c3c'))
            return
        except Exception as exc:
            message = str(exc)[:90]
            self.log(f"BLAD ladowania: {message}")
            self.root.after(0, lambda: self.status_label.config(
                text=f"Blad: {message}", fg='#e74c3c'))
            return

        self.test_cases = read_test_cases()
        n_langs = len(self.detector.languages)
        n_scripts = len({language_script(code) for code in self.detector.languages})
        self.log(f"Modele zaladowane: {n_langs} jezykow, {n_scripts} pism, "
                 f"{len(self.test_cases)} tekstow testowych")
        self.log(f"Algorytm modelu jezyka: {self.detector.algorithm}")

        self.root.after(0, lambda: self.status_label.config(
            text=f"{self.detector.algorithm} | {n_langs} jezykow | {n_scripts} pism "
                 f"| {len(self.test_cases)} tekstow testowych",
            fg='#2ecc71'))
        self.root.after(0, self.render_model_comparison)

    @property
    def models_loaded(self):
        return self.detector is not None

    # --- Zakladka MODELE -----------------------------------------------------

    def render_model_comparison(self):
        self.models_text.delete("1.0", tk.END)
        meta = self.detector.metadata if self.detector else {}
        comparison = meta.get('comparison')

        if not comparison:
            self.models_text.insert(tk.END, (
                "Brak zapisanego porownania algorytmow.\n\n"
                "Uruchom trening z porownaniem modeli:\n"
                "    python main.py train --models all\n\n"
                "Porownywane algorytmy: Naive Bayes (Multinomial i Complement),\n"
                "regresja logistyczna, SVM liniowy, Random Forest oraz siec neuronowa MLP.\n"
            ))
            return

        self.models_text.insert(tk.END, "=" * 92 + "\n")
        self.models_text.insert(tk.END, "  POROWNANIE ALGORYTMOW UCZENIA MASZYNOWEGO\n")
        self.models_text.insert(tk.END, "=" * 92 + "\n\n")
        self.models_text.insert(tk.END, f"  Data treningu:     {meta.get('trained_at', 'n/d')}\n")
        self.models_text.insert(tk.END, f"  Liczba jezykow:    {meta.get('n_classes', 'n/d')}\n")
        self.models_text.insert(tk.END, f"  Wybrany algorytm:  {meta.get('label', 'n/d')}\n\n")
        self.models_text.insert(tk.END, "-" * 92 + "\n")
        self.models_text.insert(tk.END,
                                f"  {'Model':<28}{'Accur.':>9}{'Precis.':>9}{'Recall':>9}"
                                f"{'F1':>9}{'F1 mac':>9}{'Top-3':>9}{'Trening':>10}\n")
        self.models_text.insert(tk.END, "-" * 92 + "\n")

        for row in comparison:
            self.models_text.insert(tk.END,
                                    f"  {row['label']:<28}{row['accuracy']:>9.4f}"
                                    f"{row['precision_weighted']:>9.4f}"
                                    f"{row['recall_weighted']:>9.4f}{row['f1_weighted']:>9.4f}"
                                    f"{row['f1_macro']:>9.4f}{row['top3_accuracy']:>9.4f}"
                                    f"{row['train_time']:>9.1f}s\n")

        self.models_text.insert(tk.END, "-" * 92 + "\n\n")
        self.models_text.insert(tk.END,
                                "  Miary wazone uwzgledniaja liczebnosc klas; miary makro traktuja\n"
                                "  kazdy jezyk jednakowo, wiec silniej karza slabe wyniki na jezykach\n"
                                "  o mniejszej liczbie probek.\n\n")
        self.models_text.insert(tk.END, "  Wykres: pliki/model_comparison.png\n")
        self.models_text.insert(tk.END, "  Tabela: pliki/reports/model_comparison.csv\n")

    # --- Detekcja ------------------------------------------------------------

    def clear_input(self):
        self.text_input.delete("1.0", tk.END)
        self.result_text.delete("1.0", tk.END)

    def detect_language(self):
        if not self.models_loaded:
            messagebox.showwarning("Uwaga", "Modele nie sa jeszcze zaladowane")
            return
        text = self.text_input.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Uwaga", "Wprowadz tekst do analizy")
            return

        self.log(f"Analiza tekstu: {text[:50]}...")
        self.detect_btn.config(state=tk.DISABLED, text="ANALIZUJE...")
        threading.Thread(target=lambda: self._detect_thread(text), daemon=True).start()

    def _detect_thread(self, text):
        analysis = self.detector.analyze(text)
        if analysis['language']:
            self.log(f"Wynik: {analysis['language']} ({analysis['confidence']:.1%}), "
                     f"pismo: {analysis['script']['script']}")
        else:
            self.log("Nie rozpoznano jezyka")
        self.root.after(0, lambda: self.display_results(text, analysis))

    def display_results(self, text, analysis):
        self.result_text.delete("1.0", tk.END)
        write = lambda s: self.result_text.insert(tk.END, s)  # noqa: E731

        write("=" * 74 + "\nWYNIK ANALIZY\n" + "=" * 74 + "\n\n")

        if not analysis['language']:
            write("NIE ROZPOZNANO JEZYKA\n\n")
            write("Tekst jest zbyt krotki lub nie zawiera liter zadnego\n"
                  "z obslugiwanych systemow pisma.\n")
            self.detect_btn.config(state=tk.NORMAL, text="ANALIZUJ")
            return

        top = analysis['predictions'][0]
        script = analysis['script']
        confidence = analysis['confidence']

        write(f"WYKRYTY JEZYK:  {analysis['language'].upper()} ({top['name']})\n")
        write(f"PEWNOSC:        {confidence:.1%}\n")
        write(f"RODZINA:        {top['family']}\n")
        write(f"PISMO:          {script['name']} [{script['script']}]  "
              f"{script['confidence']:.1%} ({script['source']})\n")
        write(f"ALGORYTM:       {analysis['algorithm']}\n\n")

        bar = "#" * int(confidence * 40) + "-" * (40 - int(confidence * 40))
        write(f"[{bar}] {confidence:.1%}\n\n")

        write("-" * 74 + f"\nTOP {len(analysis['predictions'])} PRZEWIDYWAN\n" + "-" * 74 + "\n")
        for i, item in enumerate(analysis['predictions'], 1):
            marks = "#" * int(item['confidence'] * 25)
            write(f"{i}. {item['language']:<5} {item['name'][:24]:<25} "
                  f"{item['confidence']:>6.1%}  {marks}\n")

        profile = script.get('profile') or {}
        if len(profile) > 1:
            write("\n" + "-" * 74 + "\nUDZIAL PISM W TEKSCIE\n" + "-" * 74 + "\n")
            for name, share in sorted(profile.items(), key=lambda kv: -kv[1]):
                write(f"  {script_name(name):<26} {share:>6.1%}\n")

        preview = text[:220] + "..." if len(text) > 220 else text
        write(f"\n{'-' * 74}\nTEKST ({len(text)} znakow):\n{preview}\n")
        self.detect_btn.config(state=tk.NORMAL, text="ANALIZUJ")

    # --- Teksty testowe ------------------------------------------------------

    def show_test_texts(self):
        if not self.test_cases:
            messagebox.showwarning("Uwaga", "Brak tekstow testowych (test_languages.txt)")
            return

        win = tk.Toplevel(self.root)
        win.title("Teksty Testowe")
        win.geometry("620x520")
        win.configure(bg=BG)

        tk.Label(win, text="Wybierz testowy tekst:", font=('Arial', 12, 'bold'),
                 bg=BG, fg='white').pack(pady=10)

        list_frame = tk.Frame(win, bg=BG)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 10))

        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        listbox = tk.Listbox(list_frame, font=('Arial', 10), bg=FIELD, fg='white',
                             selectbackground=ACCENT, yscrollcommand=scrollbar.set)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=listbox.yview)

        entries = sorted(
            self.test_cases,
            key=lambda case: (language_script(case[0]), language_name(case[0])),
        )
        for code, _ in entries:
            listbox.insert(
                tk.END,
                f"{language_name(code)} ({code.upper()}) - {script_name(language_script(code))}",
            )

        def on_select():
            selection = listbox.curselection()
            if selection:
                self.text_input.delete("1.0", tk.END)
                self.text_input.insert(tk.END, entries[selection[0]][1])
                win.destroy()

        frame = tk.Frame(win, bg=BG)
        frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        tk.Button(frame, text="WYBIERZ", font=('Arial', 10), bg=ACCENT, fg='white',
                  cursor='hand2', width=12, command=on_select).pack(side=tk.LEFT, padx=(0, 10))
        tk.Button(frame, text="ANULUJ", font=('Arial', 10), bg=PURPLE, fg='white',
                  cursor='hand2', width=12, command=win.destroy).pack(side=tk.LEFT)

    # --- Auto-test -----------------------------------------------------------

    def run_auto_test_gui(self):
        if not self.models_loaded or not self.test_cases:
            messagebox.showwarning("Uwaga", "Modele lub teksty testowe nie sa gotowe")
            return
        self.log(f"Uruchamianie auto-testu dla {len(self.test_cases)} przypadkow...")
        self.auto_btn.config(state=tk.DISABLED, text="TESTOWANIE...")
        self.auto_result_text.delete("1.0", tk.END)
        self.auto_result_text.insert(tk.END, "Uruchamianie auto-testu...\n\n")
        threading.Thread(target=self._run_auto_test_thread, daemon=True).start()

    def _run_auto_test_thread(self):
        supported = set(self.detector.languages)
        results, skipped = [], []

        for expected, text in self.test_cases:
            if expected not in supported:
                skipped.append(expected)
                continue
            analysis = self.detector.analyze(text, top_k=3)
            results.append({
                'expected': expected,
                'detected': analysis['language'] or '-',
                'confidence': analysis['confidence'],
                'script': analysis['script']['script'],
                'correct': analysis['language'] == expected,
            })

        correct = sum(row['correct'] for row in results)
        total = len(results) or 1
        self.log(f"Auto-test zakonczony: {correct}/{len(results)} "
                 f"({correct / total * 100:.1f}%)")
        self.root.after(0, lambda: self.display_auto_results(results, skipped))

    def display_auto_results(self, results, skipped):
        self.auto_result_text.delete("1.0", tk.END)
        write = lambda s: self.auto_result_text.insert(tk.END, s)  # noqa: E731

        if not results:
            write("Brak przypadkow testowych obslugiwanych przez model.\n")
            self.auto_btn.config(state=tk.NORMAL, text="URUCHOM AUTO-TEST")
            return

        correct = sum(row['correct'] for row in results)
        total = len(results)

        write("=" * 92 + "\n")
        write(f"  AUTO-TEST: {correct}/{total} POPRAWNYCH ({correct / total * 100:.1f}%)\n")
        write("=" * 92 + "\n\n")
        write(f"  {'Jezyk':<26}{'Oczek':<8}{'Wykryty':<9}{'Pewnosc':<10}"
              f"{'Pismo':<14}{'Status'}\n")
        write("  " + "-" * 88 + "\n")

        for row in sorted(results, key=lambda r: (r['correct'], r['expected'])):
            name = f"{language_name(row['expected'])[:24]}"
            write(f"  {name:<26}{row['expected']:<8}{row['detected']:<9}"
                  f"{row['confidence']:<9.1%} {row['script']:<14}"
                  f"{'OK' if row['correct'] else 'FAIL'}\n")

        per_script = defaultdict(lambda: [0, 0])
        for row in results:
            script = language_script(row['expected'])
            per_script[script][1] += 1
            per_script[script][0] += row['correct']

        write("\n  " + "=" * 88 + "\n")
        write("  SKUTECZNOSC WEDLUG SYSTEMU PISMA\n")
        write("  " + "-" * 88 + "\n")
        for script in sorted(per_script, key=lambda s: -per_script[s][1]):
            hits, count = per_script[script]
            bar = "#" * int(hits / count * 20)
            write(f"  {script_name(script):<26}{hits:>3}/{count:<4}"
                  f"({hits / count * 100:5.1f}%)  {bar}\n")

        failed = [row for row in results if not row['correct']]
        if failed:
            write(f"\n  NIEPOPRAWNE ({len(failed)}):\n")
            for row in failed:
                write(f"    - {row['expected']} ({language_name(row['expected'])}): "
                      f"wykryto '{row['detected']}' z pewnoscia {row['confidence']:.1%}\n")

        if skipped:
            write(f"\n  POMINIETO - brak jezyka w modelu ({len(skipped)}): "
                  f"{', '.join(sorted(skipped))}\n")

        write(f"\n  PODSUMOWANIE: {correct}/{total} ({correct / total * 100:.1f}%)\n")
        self.auto_btn.config(state=tk.NORMAL, text="URUCHOM AUTO-TEST")

    # --- Raporty -------------------------------------------------------------

    def generate_report_text(self):
        n_langs = len(self.detector.languages) if self.detector else 'n/d'
        report = [
            "=" * 74,
            "RAPORT - DETEKTOR JEZYKA I ALFABETU",
            "=" * 74,
            f"Data wygenerowania:  {datetime.now():%Y-%m-%d %H:%M:%S}",
            f"Algorytm:            {self.detector.algorithm if self.detector else 'n/d'}",
            f"Liczba jezykow:      {n_langs}",
            f"Tekstow testowych:   {len(self.test_cases)}",
            "=" * 74,
            "",
        ]

        auto_content = self.auto_result_text.get("1.0", tk.END).strip()
        if auto_content:
            report += ["--- WYNIKI AUTO-TESTU ---", auto_content, ""]

        models_content = self.models_text.get("1.0", tk.END).strip()
        if models_content:
            report += ["--- POROWNANIE ALGORYTMOW ---", models_content, ""]

        report += ["--- LOGI SYSTEMOWE ---", *self.log_lines, "", "=" * 74, "KONIEC RAPORTU"]
        return "\n".join(report)

    def _save_to_file(self, content, prefix):
        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            initialfile=f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}.txt",
            filetypes=[("Pliki tekstowe", "*.txt"), ("Wszystkie pliki", "*.*")],
        )
        if not filepath:
            return
        with open(filepath, 'w', encoding='utf-8') as handle:
            handle.write(content)
        self.log(f"Zapisano: {filepath}")
        messagebox.showinfo("OK", f"Zapisano:\n{filepath}")

    def save_report(self):
        if "AUTO-TEST" not in self.auto_result_text.get("1.0", tk.END):
            messagebox.showwarning("Uwaga", "Najpierw uruchom auto-test")
            return
        self._save_to_file(self.generate_report_text(), "raport")

    def save_logs(self):
        self._save_to_file("\n".join(self.log_lines), "logi")


def launch():
    root = tk.Tk()
    LanguageDetectorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    launch()
