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
This project requires Python 3.10+ and the following packages:

```bash
pip install -r requirements.txt

python -m spacy download en_core_web_sm

## Example Output

| original_text                 | cleaned_text                   | tokens                              | lemmas                          |
|-------------------------------|--------------------------------|-------------------------------------|---------------------------------|
| "That's totally amazing!"     | "that s totally amazing"       | ["that", "s", "totally", "amazing"] | ["that", "be", "totally", "amazing"] |

## Project Structure
text_preprocessor/
├── english_preprocessor/
│ ├── code/
│ │ ├── preprocess_en.py # Main text preprocessing script
│ │ └── utils.py         # File I/O and database helper functions
│ ├── data/
│ │ └── sample_en.txt    # Sample text file for testing
│ ├── demo.ipynb      # Jupyter notebook demonstrating preprocessing
│ ├── README.md
│ └── requirements.txt