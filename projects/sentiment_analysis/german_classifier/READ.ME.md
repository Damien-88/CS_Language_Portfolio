# German Sentiment Analysis Classifier

## Overview
This project implements a machine learning-based **sentiment analysis classifier for German text**.  
It predicts whether a given sentence expresses **positive**, **negative**, or **neutral** sentiment.

The system is built using classical NLP techniques (TF-IDF + Logistic Regression with probability calibration) and integrates a German-specific preprocessing pipeline designed for linguistic structure and token normalization.

---

## Project Structure

```
german_classifier/
├── code/
│   ├── train_model.py          # Train and save model + vectorizer
│   ├── predict_model.py        # Load model and run inference
│   ├── preprocess_de.py        # German NLP preprocessing pipeline
│   ├── utils.py                # Backup utilities (not used in runtime)
│   └── demo_de.ipynb           # End-to-end walkthrough notebook
├── data/
│   ├── train_de.csv            # Training data
│   ├── test_de.csv             # Test data
│   ├── sample_de.txt           # Sample input texts
│   ├── sentiment_model_de.pkl  # Saved trained model (generated)
│   └── vectorizer_de.pkl       # Saved TF-IDF vectorizer (generated)
├── requirements.txt
└── README.md
```

---

## Features

### Text Preprocessing (`preprocess_de.py`)
- Unicode normalization (NFKC) and lowercasing
- URL and mention (`@user`) removal
- Special character removal — preserves German letters (`ä`, `ö`, `ü`, `ß`) and apostrophes
- Tokenization — spaCy (`de_core_news_sm`) with NLTK `word_tokenize` (German) fallback, then whitespace split
- Stopword removal (German NLTK corpus), with negations preserved: `nicht`, `kein`, `keine`, `keinen`, `keinem`, `keiner`, `keines`, `nie`, `niemals`, `weder`, `noch`
- Lemmatization — spaCy with token passthrough fallback

### Feature Engineering (`train_model.py`)
- TF-IDF vectorization with:
  - `max_features=10000`
  - `ngram_range=(1, 2)` — unigrams and bigrams
  - `min_df=2`, `max_df=0.9`
  - `sublinear_tf=True`

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
"Ich liebe dieses Produkt!",positive
"Das ist schrecklich.",negative
"Der Film war okay.",neutral
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
python -m spacy download de_core_news_sm
python -m nltk.downloader stopwords punkt
```

---

## Usage

### Train the model
```bash
python code/train_model.py
```
Saves `sentiment_model_de.pkl` and `vectorizer_de.pkl` to `data/`.

### Run predictions
```python
from predict_model import predict_sentiment, predict_proba, predict_batch_detailed

predict_sentiment("Ich habe diesen Film wirklich genossen!")
# → "positive"

predict_proba("Ich habe diesen Film wirklich genossen!")
# → {"negative": 0.04, "neutral": 0.11, "positive": 0.85}

predict_batch_detailed(["Fantastisch!", "Schrecklich.", "Es ist okay."])
# → [{"text": ..., "prediction": ..., "probabilities": {...}}, ...]
```

### Interactive demo
Open and run `code/demo_de.ipynb` for a full end-to-end walkthrough including dataset visualizations, preprocessing demo, single and batch predictions, and a confusion matrix diagnostic.
