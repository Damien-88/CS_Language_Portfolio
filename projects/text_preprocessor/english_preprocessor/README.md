# English Text Preprocessor  

## Overview
This project implements a full **English text preprocessing pipeline** suitable for NLP and intelligence-focused tasks. The pipeline cleans, tokenizes, normalizes, and lemmatizes text, while handling common contractions and removing stopwords. Processed results are stored in a SQLite database for downstream analysis.

---

## Features

- **Text Cleaning**
  - Lowercases text
  - Removes URLs, mentions, and punctuation
  - Normalizes Unicode characters
  - Collapses extra whitespace

- **Tokenization**
  - Uses **spaCy** (preferred) or **NLTK** as fallback

- **Contraction Handling**
  - Merges common contractions, e.g.:
    - `"that's"` → `"thatis"`
    - `"can't"` → `"cant"`

- **Stopword Removal**
  - Removes English stopwords using NLTK

- **Lemmatization**
  - Uses spaCy or NLTK WordNet lemmatizer
  - Produces base forms of words for analysis

- **Database Integration**
  - Stores processed results in **SQLite**
  - Columns:
    - `id` — Unique ID
    - `original_text` — Original line from input
    - `cleaned_text` — Cleaned text
    - `tokens` — List of tokens (space-separated)
    - `lemmas` — List of lemmatized tokens (space-separated)

---

## Getting Started

### Requirements

```bash
pip install -r requirements.txt