# Multilingual Text Preprocessor

## Overview
This module provides a multilingual preprocessing pipeline for English, German,
and Russian text. It supports language-specific cleaning/tokenization and stores
processed outputs in SQLite.

## Project Structure
```
multilingual_preprocessor/
├── code/
│   ├── preprocess_ml.py      # Main multilingual preprocessing pipeline
│   ├── utils_ml.py           # File and SQLite helper functions
│   └── demo_ml.ipynb         # Notebook demo
├── data/
│   ├── sample_en.txt         # English sample input
│   ├── sample_de.txt         # German sample input
│   └── sample_ru.txt         # Russian sample input
├── requirements.txt
└── READ.ME.md
```

## Pipeline Details
Implemented in `code/preprocess_ml.py` using `MultilingualPreprocessor`:
- Language selection via `language` and `country_code`
- Alias support for language names (`deutsch`, Russian-language name alias)
- Unicode normalization with `NFKC`
- URL and mention removal
- Language-specific character filtering:
  - English: `a-z`, digits, whitespace
  - German: preserves `ä`, `ö`, `ü`, `ß`
  - Russian: preserves Cyrillic and digits
- Tokenization with spaCy (per language model) and NLTK fallback
- Stopword removal using NLTK stopwords for selected language
- English contraction merge helper
- Lemmatization with spaCy, or English NLTK fallback
- Optional stemming helpers for all three languages

## Storage Output
`process_file()` reads lines and writes rows to SQLite using `utils_ml.py`.

Stored row keys:
- `original`
- `cleaned`
- `tokens`
- `lemmas`

Database path pattern:
- `data/processed_results_<country_code>.db`

## Setup
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
python -m spacy download ru_core_news_sm
python -m nltk.downloader punkt stopwords wordnet
```

## Run
```bash
python code/preprocess_ml.py --in ../data/sample_en.txt --lang english --code en
python code/preprocess_ml.py --in ../data/sample_de.txt --lang german --code de
python code/preprocess_ml.py --in ../data/sample_ru.txt --lang russian --code ru
```

## Notes
- This preprocessor is file-oriented and writes processed rows to SQLite.
- Model selection in code depends on `country_code` (`en/de/ru`) when using non-English languages.