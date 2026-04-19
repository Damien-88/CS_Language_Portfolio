# Russian Text Preprocessor

## Overview
This module provides a Russian preprocessing pipeline for downstream NLP tasks.
It cleans text, tokenizes it, removes stopwords, lemmatizes tokens, and stores
processed outputs in SQLite.

## Project Structure
```
russian_preprocessor/
├── code/
│   ├── preprocess_ru.py      # Main preprocessing pipeline
│   ├── utils_ru.py           # File and SQLite helper functions
│   └── demo_ru.ipynb         # Notebook demo
├── data/
│   └── sample_ru.txt         # Sample input file
├── requirements.txt
└── READ.ME.md
```

## Pipeline Details
Implemented in `code/preprocess_ru.py`:
- Unicode normalization with `NFKC`
- Lowercasing
- URL and mention removal
- Character filtering that preserves Cyrillic letters (`а-я`, `ё`), digits, and whitespace
- Tokenization with spaCy (`ru_core_news_sm`) and NLTK fallback
- Stopword removal using NLTK Russian stopwords
- Lemmatization with spaCy, or passthrough fallback
- Optional stemming helper (`RussianStemmer`)

## Storage Output
`process_file()` reads lines and writes rows to SQLite using `utils_ru.py`.

Stored row keys:
- `original`
- `cleaned`
- `tokens`
- `lemmas`

Database path:
- `data/processed_results_ru.db`

## Setup
```bash
pip install -r requirements.txt
python -m spacy download ru_core_news_sm
python -m nltk.downloader punkt stopwords
```

## Run
```bash
python code/preprocess_ru.py --in ../data/sample_ru.txt
```

## Notes
- This preprocessor is file-oriented and writes processed rows to SQLite.
- spaCy is preferred for tokenization/lemmatization when available.