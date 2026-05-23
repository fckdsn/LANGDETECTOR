# LANGDETECTOR

Многоязычный детектор языка на основе TF-IDF (символьные и словные n-граммы) и Random Forest. Поддерживается **60+ языков** (латиница и кириллица): европейские, тюркские, славянские, финно-угорские и др.

Данные для обучения загружаются из корпуса [OSCAR](https://huggingface.co/datasets/oscar-corpus/community-oscar) через Hugging Face.

## Требования

- Python 3.10+
- Токен Hugging Face (для загрузки OSCAR при обучении)

```bash
pip install -r requirements.txt
```

Опционально для разработки:

```bash
pip install pytest ruff mypy
```

## Структура проекта

| Файл | Назначение |
|------|------------|
| `main_train.py` | Обучение модели (загрузка данных → EDA → train) |
| `train.py` | Логика обучения, метрики, сохранение артефактов |
| `data_loader.py` | Загрузка текстов из OSCAR |
| `preprocess.py` | Очистка текста, определение алфавита |
| `predict.py` | CLI-детектор (`LanguageDetector`) |
| `main.py` | Альтернативный CLI с обучением и auto-test |
| `gui.py` | Графический интерфейс (Tkinter) |
| `config.py` | Языки, пути, метаданные |
| `test_languages.txt` | Примеры для auto-test (`код\|текст`) |
| `pliki/models/` | Сохранённые модели (после обучения) |

## Быстрый старт

### 1. Токен Hugging Face

```powershell
$env:HF_TOKEN = "hf_xxxxxxxx"
```

Или задайте переменную в системе. Без токена загрузка OSCAR при обучении не сработает.

### 2. Обучение модели

Полный пайплайн: загрузка ~60 языков, EDA, обучение языковой и алфавитной моделей, графики в `pliki/`.

```bash
python main_train.py
```

Занимает **десятки минут** (зависит от сети и числа языков). Результат:

- `pliki/models/language_model.pkl`
- `pliki/models/language_encoder.pkl`
- `pliki/models/vectorizers.pkl`
- `pliki/models/alphabet_model.pkl`
- `pliki/eda_analysis.png`, confusion matrix, отчёт классификации

### 3. Интерактивное определение языка

**Консоль (модуль `predict.py`):**

```bash
python predict.py
```

Введите текст (минимум **30 символов** после очистки), для выхода — `exit`.

**Консоль (расширенный `main.py`):**

```bash
python main.py
```

Команды: `auto` — тест из `test_languages.txt`, `languages` — список классов, `exit`.

**GUI:**

```bash
python gui.py
```

Вкладки: детекция, auto-test, логи.

### 4. Auto-test по файлу

Формат `test_languages.txt` — одна строка на язык:

```
en|The quick brown fox jumps over the lazy dog...
de|Die schnelle Entwicklung der künstlichen Intelligenz...
```

Запуск из `main.py` (после обучения): введите `auto`.

## Использование в коде

```python
from predict import LanguageDetector

detector = LanguageDetector()
lang, confidence, top5 = detector.predict(
    "The rapid advancement of artificial intelligence has transformed many industries."
)
print(lang, confidence)  # en, ~0.95
for p in top5:
    print(p["language"], p["name"], p["confidence"])
```

Очистка текста отдельно:

```python
from preprocess import clean_text, detect_alphabet

text = clean_text("Hello! Visit https://example.com  ")
alphabet = detect_alphabet(text)  # latin | cyrillic | mixed
```

## Тесты

Запускайте из корня проекта (`LANGDETECTOR`):

```bash
python -m pytest tests/ -v
```

Также работает прямой запуск файла (путь к корню подставляется автоматически):

```bash
python tests/test_preprocess.py
python tests/test_predict.py
```

Не запускайте `conftest.py` напрямую — это служебный файл pytest.

Тесты `preprocess` не требуют обученных моделей. Тесты `predict` используют минимальную модель во временной директории.

## Линтеры

```bash
python -m ruff check .
python -m mypy .
```

## Проверка загрузки данных

```bash
python data_loader.py
```

## Дополнительные скрипты

```bash
python add_polish_data.py   # подготовка доп. польских текстов (нужны уже обученные vectorizers)
```

## Примечания

- Минимальная длина входа для `LanguageDetector.predict`: **30 символов** после `clean_text`.
- Модели в репозиторий обычно не коммитятся — обучите локально или положите `.pkl` в `pliki/models/`.
- Не храните `HF_TOKEN` в коде: используйте переменную окружения `HF_TOKEN` (см. `config.py`).

## Лицензия

Учебный / исследовательский проект. Данные OSCAR — см. условия на Hugging Face.
