# Multilingual Sentiment Analysis Classifier

## Overview
This project implements a machine learning-based **sentiment analysis classifier for multilingual text** (English, German, and Russian).  
It predicts whether a sentence expresses **positive**, **negative**, or **neutral** sentiment.

The system uses classical NLP techniques (TF-IDF + Logistic Regression with probability calibration) and a language-aware preprocessing pipeline that applies the correct normalization/tokenization strategy per language.

---

## Project Structure

```
multilingual_classifier/
├── code/
│   ├── train_model_ml.py        # Train and save multilingual model + vectorizer
│   ├── predict_model_ml.py      # Load model and run multilingual inference
│   ├── preprocess_ml.py         # Multilingual NLP preprocessing pipeline
│   ├── utils_ml.py              # Backup utilities (not used in runtime)
│   └── demo_ml.ipynb            # End-to-end multilingual walkthrough notebook
├── data/
│   ├── train_ml.csv             # Multilingual training data
│   ├── test_ml.csv              # Multilingual test data
│   ├── sample_ml.txt            # Sample input texts
│   ├── sentiment_model_ml.pkl   # Saved trained model (generated)
│   └── vectorizer_ml.pkl        # Saved TF-IDF vectorizer (generated)
├── requirements.txt
└── README.md
```

---

## Features

### Text Preprocessing (`preprocess_ml.py`)
- Language-aware pipeline via `MultilingualPreprocessor` class
- Unicode normalization (NFKC) and lowercasing
- URL and mention (`@user`) removal
- Language-specific character filtering:
	- English: `a-z`, digits, whitespace, apostrophes
	- German: preserves `ä`, `ö`, `ü`, `ß`
	- Russian: preserves Cyrillic (`а-яё`)
- Tokenization — spaCy per language model with NLTK fallback
- Stopword removal (language corpus), with negations preserved:
	- English: `no`, `not`, `never`, `nor`, `none`, ...
	- German: `nicht`, `kein`, `keine`, `nie`, `niemals`, `weder`, `noch`, ...
	- Russian: `не`, `нет`, `ни`, `никогда`, `никто`, `ничто`, ...
- Lemmatization — spaCy with English NLTK `WordNetLemmatizer` fallback

### Feature Engineering (`train_model_ml.py`)
- TF-IDF vectorization with:
	- `max_features=10000`
	- `ngram_range=(1, 2)` — unigrams and bigrams
	- `min_df=2`, `max_df=0.9`
	- `sublinear_tf=True`
	- `token_pattern=r"(?u)\b\w+\b"` — includes single-character tokens

### Machine Learning Model (`train_model_ml.py`)
- Class-based trainer: `MultilingualSentimentTrainer`
- **Logistic Regression** (`C=1.5`, `class_weight="balanced"`, `max_iter=2000`)
- Wrapped in **`CalibratedClassifierCV`** (Platt scaling, `cv=3`) for reliable probability estimates
- Stratified 80/20 train/test split (`random_state=42`)
- Handles optional `language` column in training data:
	- If present, each row is preprocessed with its language
	- If absent, uses default language fallback (`english`)

### Prediction (`predict_model_ml.py`)
- Class-based predictor: `MultilingualSentimentPredictor`
- Single text prediction: `predict_sentiment(text, language="english")`
- Full probability distribution: `predict_proba(text, language="english")`
- Batch prediction with per-item detail: `predict_batch_detailed(text_list, language="english")`
- Confidence thresholding: predictions fall back to `neutral` when `max_prob < 0.45` or top-two class margin is `< 0.12`
- Supports language aliases (`en`, `de`, `ru`, `deutsch`, `русский`, etc.)

### Evaluation
- Accuracy, precision, recall, F1-score (per class)
- Confusion matrix (in demo diagnostics)

---

## Dataset Format

Training data should be a UTF-8 CSV with sentiment labels as strings:

```csv
text,sentiment
"I love this product!",positive
"Das ist schrecklich.",negative
"Это нормально.",neutral
```

Optional multilingual schema (recommended for explicit language routing during training):

```csv
text,sentiment,language
"I love this product!",positive,english
"Das ist schrecklich.",negative,german
"Это нормально.",neutral,russian
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
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
python -m spacy download ru_core_news_sm
python -m nltk.downloader stopwords punkt wordnet
```

---

## Usage

### Train the model
```bash
python code/train_model_ml.py
```
Saves `sentiment_model_ml.pkl` and `vectorizer_ml.pkl` to `data/`.

### Run predictions
```python
from predict_model_ml import predict_sentiment, predict_proba, predict_batch_detailed

predict_sentiment("Ich bin sehr zufrieden damit.", language="german")
# -> "positive"

predict_proba("Обновление добавило новые баги.", language="russian")
# -> {"negative": 0.81, "neutral": 0.13, "positive": 0.06}

predict_batch_detailed([
		"This is great!",
		"Das ist enttäuschend.",
		"Это нормально."
], language="english")
# -> [{"text": ..., "prediction": ..., "probabilities": {...}}, ...]
```

For mixed-language batches, call prediction per item with each item's language.

### Interactive demo
Open and run `code/demo_ml.ipynb` for an end-to-end walkthrough including:
- multilingual dataset overview
- preprocessing demos for English/German/Russian
- single and batch predictions with language-aware inference
- confusion matrix diagnostic
