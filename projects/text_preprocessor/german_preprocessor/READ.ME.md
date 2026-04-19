# German Text Preprocessor

## Overview
This module provides a German preprocessing pipeline for downstream NLP tasks.
It cleans text, tokenizes it, removes stopwords, lemmatizes tokens, and stores
processed outputs in SQLite.

## Project Structure
```
german_preprocessor/
├── code/
│   ├── preprocess_de.py      # Main preprocessing pipeline
│   ├── utils_de.py           # File and SQLite helper functions
│   └── demo_de.ipynb         # Notebook demo
├── data/
│   └── sample_de.txt         # Sample input file
├── requirements.txt
└── READ.ME.md
```

## Pipeline Details
Implemented in `code/preprocess_de.py`:
- Unicode normalization with `NFKC`
- Lowercasing
- URL and mention removal
- Character filtering that preserves German letters (`a-z`, `ä`, `ö`, `ü`, `ß`)
- Tokenization with spaCy (`de_core_news_sm`) and NLTK fallback
- Stopword removal using NLTK German stopwords
- Lemmatization with spaCy, or passthrough fallback
- Optional stemming helper (`GermanStemmer`)

## Storage Output
`process_file()` reads lines and writes rows to SQLite using `utils_de.py`.

Stored row keys:
- `original`
- `cleaned`
- `tokens`
- `lemmas`

Database path:
- `data/processed_results_de.db`

## Setup
```bash
pip install -r requirements.txt
python -m spacy download de_core_news_sm
python -m nltk.downloader punkt stopwords
```

## Run
```bash
python code/preprocess_de.py --in ../data/sample_de.txt
```

## Notes
- This preprocessor is file-oriented and writes processed rows to SQLite.
- spaCy is preferred for tokenization/lemmatization when available.