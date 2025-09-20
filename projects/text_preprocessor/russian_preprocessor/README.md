# Russian Text Preprocessor  

## Overview
This project implements a full **Russian text preprocessing pipeline** suitable for NLP and intelligence-focused tasks. The pipeline cleans, tokenizes, normalizes, and lemmatizes text, while handling common contractions and removing stopwords. Processed results are stored in a SQLite database for downstream analysis.

---

## Features

- **Text Cleaning**
  - Lowercases text
  - Removes URLs, mentions, and punctuation
  - Normalizes Unicode characters
  - Collapses extra whitespace

- **Tokenization**
  - Uses **spaCy** (preferred) or **NLTK** as fallback

- **Stopword Removal**
  - Removes Russian stopwords using NLTK

- **Lemmatization**
  - Uses spaCy lemmatizer
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
This project requires Python 3.10+ and the following packages:

```bash
pip install -r requirements.txt

python -m spacy download ru_core_news_sm

## Example Output

| original_text            | cleaned_text         | tokens                        | lemmas                         |
|--------------------------|----------------------|-------------------------------|--------------------------------|
| "Это новая книга."      | "это новая книга"     |   ["это", "новая", "книга"]   |    ["это", "новый", "книга"]   |

## Project Structure
text_preprocessor/
├── russian_preprocessor/
│ ├── code/
│ │ ├── preprocess_ru.py # Main Russian text preprocessing script
│ │ └── utils.py         # File I/O and database helper functions
│ ├── data/
│ │ └── sample_ru.txt    # Sample Russian text file for testing
│ ├── demo_ru.ipynb      # Jupyter notebook demonstrating preprocessing
│ ├── README.md
│ └── requirements.txt