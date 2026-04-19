# Russian Sentiment Analysis Classifier

## Overview
This project implements a machine learning-based **sentiment analysis classifier for Russian text**.  
It predicts whether a given sentence expresses **positive**, **negative**, or **neutral** sentiment.

The system is built using classical NLP techniques (TF-IDF + Logistic Regression with probability calibration) and integrates a Russian-specific preprocessing pipeline designed for Cyrillic text and morphological normalization.

---

## Project Structure

```
russian_classifier/
├── code/
│   ├── train_model.py          # Train and save model + vectorizer
│   ├── predict_model.py        # Load model and run inference
│   ├── preprocess_ru.py        # Russian NLP preprocessing pipeline
│   ├── utils.py                # Backup utilities (not used in runtime)
│   └── demo_ru.ipynb           # End-to-end walkthrough notebook
├── data/
│   ├── train_ru.csv            # Training data
│   ├── test_ru.csv             # Test data
│   ├── sample_ru.txt           # Sample input texts
│   ├── sentiment_model_ru.pkl  # Saved trained model (generated)
│   └── vectorizer_ru.pkl       # Saved TF-IDF vectorizer (generated)
├── requirements.txt
└── README.md
```

---

## Features

### Text Preprocessing (`preprocess_ru.py`)
- Unicode normalization (NFKC) and lowercasing
- URL and mention (`@user`) removal
- Character filtering — keeps Cyrillic (`а-яё`), Latin (`a-z`), digits, whitespace, and apostrophes
- Tokenization — spaCy (`ru_core_news_sm`) with NLTK `word_tokenize` (Russian) fallback, then whitespace split
- Stopword removal (Russian NLTK corpus), with negations preserved: `не`, `нет`, `ни`, `никогда`, `никто`, `ничто`, `нигде`, `никуда`, `никак`, `нисколько`
- Lemmatization — spaCy with token passthrough fallback

### Feature Engineering (`train_model.py`)
- TF-IDF vectorization with:
  - `max_features=10000`
  - `ngram_range=(1, 2)` — unigrams and bigrams
  - `min_df=2`, `max_df=0.9`
  - `sublinear_tf=True`
  - `token_pattern=r"(?u)\b\w+\b"` — includes single-character Cyrillic tokens

### Machine Learning Model (`train_model.py`)
- **Logistic Regression** (`C=1.5`, `class_weight="balanced"`, `max_iter=2000`)
- Wrapped in **`CalibratedClassifierCV`** (Platt scaling, `cv=3`) for reliable probability estimates
- Stratified 80/20 train/test split (`random_state=42`)

### Prediction (`predict_model.py`)
- Single text prediction: `predict_sentiment(text)`
- Full probability distribution: `predict_proba(text)`
- Batch prediction with per-item detail: `predict_batch_detailed(text_list)`
- Confidence thresholding: predictions fall back to `neutral` when `max_prob < 0.45` or the margin between the top two classes is `< 0.12`

### Evaluation
- Accuracy, precision, recall, F1-score (per class)
- Confusion matrix

---

## Dataset Format

Training data should be a UTF-8 CSV with string sentiment labels:

```csv
text,sentiment
"Мне очень нравится этот продукт!",positive
"Это ужасно.",negative
"Фильм был нормальным.",neutral
```

Labels: `positive`, `negative`, `neutral`

---

## Setup

### Install dependencies
```bash
pip install -r requirements.txt
```

### Download NLP resources
```bash
python -m spacy download ru_core_news_sm
python -m nltk.downloader stopwords punkt
```

---

## Usage

### Train the model
```bash
python code/train_model.py
```
Saves `sentiment_model_ru.pkl` and `vectorizer_ru.pkl` to `data/`.

### Run predictions
```python
from predict_model import predict_sentiment, predict_proba, predict_batch_detailed

predict_sentiment("Мне очень нравится этот продукт!")
# → "positive"

predict_proba("Мне очень нравится этот продукт!")
# → {"negative": 0.04, "neutral": 0.10, "positive": 0.86}

predict_batch_detailed(["Отлично!", "Ужасно.", "Всё нормально."])
# → [{"text": ..., "prediction": ..., "probabilities": {...}}, ...]
```

### Interactive demo
Open and run `code/demo_ru.ipynb` for a full end-to-end walkthrough including dataset visualizations, preprocessing demo, single and batch predictions, and a confusion matrix diagnostic.