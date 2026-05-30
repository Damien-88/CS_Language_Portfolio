# English Text Preprocessor

## Overview
This module provides an English preprocessing pipeline for downstream NLP tasks.
It cleans text, tokenizes it, removes stopwords, applies lemmatization, and stores
processed outputs in SQLite.

## Project Structure
```
english_preprocessor/
├── code/
│   ├── preprocess_en.py      # Main preprocessing pipeline
│   ├── utils.py              # File and SQLite helper functions
│   └── demo.ipynb            # Notebook demo
├── data/
│   └── sample_en.txt         # Sample input file
├── requirements.txt
└── READ.ME.md
```

## Pipeline Details
Implemented in `code/preprocess_en.py`:
- Unicode normalization with `NFKC`
- Lowercasing
- URL and mention removal
- Punctuation removal via regex
- Tokenization with spaCy (`en_core_web_sm`) and NLTK fallback
- Contraction merge logic for English tokens
- Stopword removal using NLTK English stopwords
- Lemmatization with spaCy, or NLTK WordNet fallback
- Optional stemming helper (`PorterStemmer`)

## Storage Output
`process_file()` reads lines from text input and stores rows in SQLite using `utils.py`.

Stored row keys:
- `original`
- `cleaned`
- `tokens`
- `lemmas`

Database path:
- `data/processed_results.db`

## Setup
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt stopwords wordnet
```

## Run
```bash
python code/preprocess_en.py --in ../data/sample_en.txt
```

## Notes
- This preprocessor is file-oriented and writes processed rows to SQLite.
- spaCy is preferred for tokenization/lemmatization when available.