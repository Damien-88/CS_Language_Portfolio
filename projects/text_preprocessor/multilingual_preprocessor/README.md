# Multilingual Text Preprocessor  

## Overview
This project implements a full **multilingual text preprocessing pipeline** for English, German, and Russian, suitable for NLP and intelligence-focused tasks. The pipeline cleans, tokenizes, normalizes, and lemmatizes text, while handling common contractions and removing stopwords. Processed results are stored in a SQLite database for downstream analysis.

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
  - Removes stopwords for specified language using NLTK

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

python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
python -m spacy download ru_core_news_sm

## Example Output
# English
| original_text                 | cleaned_text                   | tokens                              | lemmas                          |
|-------------------------------|--------------------------------|-------------------------------------|---------------------------------|
| "That's totally amazing!"     | "that s totally amazing"       | ["that", "s", "totally", "amazing"] | ["that", "be", "totally", "amazing"] |

# German
| original_text            | cleaned_text         | tokens                        | lemmas                         |
|--------------------------|----------------------|-------------------------------|--------------------------------|
| "Das ist ein Buch."      | "das ist ein buch"   | ["das", "ist", "ein", "buch"] | ["das", "sein", "ein", "buch"] |

# Russian

| original_text            | cleaned_text         | tokens                        | lemmas                         |
|--------------------------|----------------------|-------------------------------|--------------------------------|
| "Это новая книга."      | "это новая книга"     |   ["это", "новая", "книга"]   |    ["это", "новый", "книга"]   |

## Project Structure
text_preprocessor/
├── multilingual_preprocessor/
│ ├── code/
│ │ ├── preprocess_ml.py # Main Multilingual text preprocessing script
│ │ └── utils_ml.py         # File I/O and database helper functions
│ ├── data/
│ │ └── sample_en.txt    # Sample English text file for testing
│ │ └── sample_de.txt    # Sample German text file for testing
│ │ └── sample_ru.txt    # Sample Russian text file for testing
│ ├── demo_ml.ipynb      # Jupyter notebook demonstrating preprocessing
│ ├── README.md
│ └── requirements.txt