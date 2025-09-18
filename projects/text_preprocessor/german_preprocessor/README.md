# German Text Preprocessor  

## Overview
This project implements a full **German text preprocessing pipeline** suitable for NLP and intelligence-focused tasks. The pipeline cleans, tokenizes, normalizes, and lemmatizes text, while handling common contractions and removing stopwords. Processed results are stored in a SQLite database for downstream analysis.

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
  - Removes German stopwords using NLTK

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

python -m spacy download de_core_news_sm

## Example Output

| original_text            | cleaned_text         | tokens                        | lemmas                         |
|--------------------------|----------------------|-------------------------------|--------------------------------|
| "Das ist ein Buch."      | "das ist ein buch"   | ["das", "ist", "ein", "buch"] | ["das", "sein", "ein", "buch"] |

## Project Structure
text_preprocessor/
├── german_preprocessor/
│ ├── code/
│ │ ├── preprocess_de.py # Main German text preprocessing script
│ │ └── utils.py         # File I/O and database helper functions
│ ├── data/
│ │ └── sample_de.txt    # Sample German text file for testing
│ ├── demo_de.ipynb      # Jupyter notebook demonstrating preprocessing
│ ├── README.md
│ └── requirements.txt