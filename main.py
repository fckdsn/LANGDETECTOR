"""Interfejs wiersza polecen systemu detekcji jezyka i pisma.

Podkomendy:
    train      trening z porownaniem algorytmow uczenia maszynowego
    predict    rozpoznanie jezyka podanego tekstu (lub tryb interaktywny)
    autotest   ocena na zbiorze test_languages.txt
    report     tabela porownania algorytmow z ostatniego treningu
    languages  lista skonfigurowanych jezykow wedlug pisma
    models     lista dostepnych algorytmow
    gui        uruchomienie interfejsu graficznego
"""

import argparse
import os
import sys
from collections import defaultdict

from config import (
    LANGUAGES,
    MODELS_DIR,
    SAMPLE_OVERRIDES,
    SAMPLES_PER_LANGUAGE,
    language_name,
    language_script,
    languages_by_script,
    script_name,
)
from models import ALL_MODELS, DEFAULT_MODELS, MODEL_REGISTRY
from predict import TEST_FILE, read_test_cases

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _load_detector():
    from predict import LanguageDetector, ModelsNotFoundError
    try:
        return LanguageDetector()
    except ModelsNotFoundError as exc:
        print(f"\n{exc}")
        return None


# --- Podkomendy --------------------------------------------------------------

def cmd_train(args):
    from main_train import run_training

    languages = args.languages.split(',') if args.languages else None
    models = args.models.split(',') if args.models else None
    run_training(
        languages=languages,
        samples=args.samples,
        model_names=models,
        keep_candidates=args.keep_candidates,
        run_eda=not args.no_eda,
    )
    return 0


def cmd_predict(args):
    detector = _load_detector()
    if detector is None:
        return 1

    from predict import _print_analysis
    from predict import main as interactive

    print(f"Algorytm: {detector.algorithm} | jezykow: {len(detector.languages)}")

    if args.file:
        with open(args.file, 'r', encoding='utf-8') as handle:
            text = handle.read()
    else:
        text = ' '.join(args.text).strip()

    if not text:
        interactive()
        return 0

    _print_analysis(detector.analyze(text, top_k=args.top))
    return 0


def cmd_autotest(args):
    detector = _load_detector()
    if detector is None:
        return 1

    cases = read_test_cases(args.file)
    if not cases:
        print(f"Brak przypadkow testowych w pliku {args.file}")
        return 1

    print("=" * 78)
    print(f" AUTO-TEST: {len(cases)} przypadkow | algorytm: {detector.algorithm}")
    print("=" * 78)
    print(f"\n {'Oczek':<6} {'Wykryty':<8} {'Pewnosc':<10} {'Pismo':<12} {'Status':<7} Tekst")
    print("-" * 78)

    correct = 0
    per_script = defaultdict(lambda: [0, 0])
    unsupported = []

    for expected, text in cases:
        if expected not in detector.languages:
            unsupported.append(expected)
            continue

        analysis = detector.analyze(text, top_k=3)
        detected = analysis['language'] or '-'
        ok = detected == expected
        correct += ok

        script = language_script(expected)
        per_script[script][1] += 1
        per_script[script][0] += ok

        print(f" {expected:<6} {detected:<8} {analysis['confidence']:<9.1%} "
              f"{analysis['script']['script']:<12} {'OK' if ok else 'FAIL':<7} {text[:28]}...")

    tested = sum(count for _, count in per_script.values())
    print("-" * 78)
    if unsupported:
        print(f" Pominieto (jezyk nie ma modelu): {', '.join(unsupported)}")
    if not tested:
        print(" Brak przypadkow do oceny.")
        return 1

    print(f" WYNIK: {correct}/{tested} poprawnych ({correct / tested * 100:.1f}%)")
    print("\n Skutecznosc wedlug pisma:")
    for script in sorted(per_script, key=lambda s: -per_script[s][1]):
        hits, total = per_script[script]
        print(f"   {script_name(script):<24} {hits:>3}/{total:<3} ({hits / total * 100:5.1f}%)")
    print("=" * 78)
    return 0


def cmd_report(args):
    import joblib

    path = os.path.join(MODELS_DIR, 'model_metadata.pkl')
    if not os.path.exists(path):
        print("Brak metadanych treningu. Uruchom najpierw: python main.py train")
        return 1

    meta = joblib.load(path)
    print("=" * 78)
    print(" POROWNANIE ALGORYTMOW - OSTATNI TRENING")
    print("=" * 78)
    print(f" Data:              {meta.get('trained_at', 'n/d')}")
    print(f" Liczba jezykow:    {meta.get('n_classes', 'n/d')}")
    print(f" Wybrany algorytm:  {meta.get('label', 'n/d')}")
    print("-" * 78)
    print(f" {'Model':<28}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}"
          f"{'F1':>9}{'F1 makro':>10}")
    print("-" * 78)
    for row in meta.get('comparison', []):
        print(f" {row['label']:<28}{row['accuracy']:>10.4f}{row['precision_weighted']:>11.4f}"
              f"{row['recall_weighted']:>9.4f}{row['f1_weighted']:>9.4f}{row['f1_macro']:>10.4f}")
    print("=" * 78)
    return 0


def cmd_languages(args):
    grouped = languages_by_script()
    print("=" * 78)
    print(f" SKONFIGUROWANE JEZYKI: {len(LANGUAGES)} w {len(grouped)} systemach pisma")
    print("=" * 78)
    for script in sorted(grouped, key=lambda s: (-len(grouped[s]), s)):
        codes = grouped[script]
        print(f"\n {script_name(script)} [{script}] - {len(codes)} jezykow")
        for code in codes:
            budget = SAMPLE_OVERRIDES.get(code, SAMPLES_PER_LANGUAGE)
            print(f"   {code:<5} {language_name(code):<24} probek: {budget}")
    return 0


def cmd_models(args):
    print("=" * 78)
    print(" DOSTEPNE ALGORYTMY")
    print("=" * 78)
    for name in ALL_MODELS:
        spec = MODEL_REGISTRY[name]
        default = ' (w domyslnym porownaniu)' if name in DEFAULT_MODELS else ''
        print(f"\n {name}{default}")
        print(f"   {spec.label} | rodzina: {spec.family}")
        print(f"   {spec.notes}")
    print("\n Zestawy: default, fast, all")
    print(" Przyklad: python main.py train --models random_forest,linear_svm,mlp")
    return 0


def cmd_gui(args):
    from gui import launch
    launch()
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog='main.py',
        description='System klasyfikacji jezykow naturalnych i pism (uczenie maszynowe).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest='command')

    train = sub.add_parser('train', help='trening z porownaniem algorytmow')
    train.add_argument('--samples', type=int, default=None,
                       help='liczba probek na jezyk (domyslnie wg konfiguracji)')
    train.add_argument('--languages', default=None,
                       help='lista kodow jezykow rozdzielona przecinkami')
    train.add_argument('--models', default=None,
                       help="algorytmy do porownania albo zestaw: default, fast, all")
    train.add_argument('--keep-candidates', action='store_true',
                       help='zachowaj na dysku wszystkie porownywane modele')
    train.add_argument('--no-eda', action='store_true', help='pomin analize eksploracyjna')
    train.set_defaults(func=cmd_train)

    predict = sub.add_parser('predict', help='rozpoznaj jezyk tekstu')
    predict.add_argument('text', nargs='*', help='tekst do analizy')
    predict.add_argument('--file', default=None, help='wczytaj tekst z pliku')
    predict.add_argument('--top', type=int, default=5, help='liczba hipotez do pokazania')
    predict.set_defaults(func=cmd_predict)

    autotest = sub.add_parser('autotest', help=f'ocena na zbiorze {TEST_FILE}')
    autotest.add_argument('--file', default=TEST_FILE)
    autotest.set_defaults(func=cmd_autotest)

    sub.add_parser('report', help='tabela porownania algorytmow').set_defaults(func=cmd_report)
    sub.add_parser('languages', help='lista jezykow wedlug pisma').set_defaults(func=cmd_languages)
    sub.add_parser('models', help='lista algorytmow').set_defaults(func=cmd_models)
    sub.add_parser('gui', help='interfejs graficzny').set_defaults(func=cmd_gui)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, 'func', None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
