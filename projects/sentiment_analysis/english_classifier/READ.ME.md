# English Sentiment Analysis Classifier

## Overview
This project implements a machine learning-based **sentiment analysis classifier for English text**.  
It predicts whether a given sentence expresses **positive**, **negative**, or **neutral** sentiment.

The system is built using classical NLP techniques (TF-IDF + Logistic Regression with probability calibration) and integrates the English preprocessing pipeline.

---

## Project Structure

```
english_classifier/
├── code/
│   ├── train_model.py          # Train and save model + vectorizer
│   ├── predict_model.py        # Load model and run inference
│   ├── preprocess_en.py        # English NLP preprocessing pipeline
│   ├── utils.py                # Backup utilities (not used in runtime)
│   └── demo_en.ipynb           # End-to-end walkthrough notebook
├── data/
│   ├── train_en.csv            # Training data
│   ├── test_en.csv             # Test data
│   ├── sample_en.txt           # Sample input texts
│   ├── sentiment_model.pkl     # Saved trained model (generated)
│   └── vectorizer_en.pkl       # Saved TF-IDF vectorizer (generated)
├── requirements.txt
└── README.md
```

---

## Features

### Text Preprocessing (`preprocess_en.py`)
- Unicode normalization (NFKC) and lowercasing
- URL and mention (`@user`) removal
- Tokenization — spaCy (`en_core_web_sm`) with NLTK `word_tokenize` fallback
- Stopword removal (English), with negations preserved: `no`, `not`, `never`, `nor`, `none`, `nobody`, `nothing`, `neither`, `nowhere`, `hardly`, `scarcely`, `barely`
- Lemmatization — spaCy with NLTK `WordNetLemmatizer` fallback

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
"I love this product!",positive
"This is terrible.",negative
"The movie was okay.",neutral
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
python -m nltk.downloader stopwords punkt wordnet
```

---

## Usage

### Train the model
```bash
python code/train_model.py
```
Saves `sentiment_model.pkl` and `vectorizer_en.pkl` to `data/`.

### Run predictions
```python
from predict_model import predict_sentiment, predict_proba, predict_batch_detailed

predict_sentiment("I really enjoyed this!")
# → "positive"

predict_proba("I really enjoyed this!")
# → {"negative": 0.05, "neutral": 0.12, "positive": 0.83}

predict_batch_detailed(["Great!", "Terrible.", "It's fine."])
# → [{"text": ..., "prediction": ..., "probabilities": {...}}, ...]
```

### Interactive demo
Open and run `code/demo_en.ipynb` for a full end-to-end walkthrough including dataset visualizations, preprocessing demo, single and batch predictions, and a confusion matrix diagnostic.
