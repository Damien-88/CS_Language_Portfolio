# English Sentiment Analysis Classifier

## Overview
This project implements a machine learning-based **sentiment analysis classifier for English text**.  
It predicts whether a given sentence expresses **positive**, **negative**, or **neutral** sentiment.

The system is built using classical NLP techniques (TF-IDF + Logistic Regression) and integrates your existing English preprocessing pipeline.



## Features

### Text Preprocessing
- Lowercasing and Unicode normalization
- Tokenization (spaCy or NLTK fallback)
- Stopword removal
- Lemmatization
- Contraction handling (e.g., "don't → do not")

### Feature Engineering
- TF-IDF vectorization of cleaned text
- Configurable feature size for scalability

### Machine Learning Model
- Logistic Regression classifier (baseline)
- Easily extendable to Naive Bayes or SVM

### Evaluation
- Accuracy
- Precision / Recall / F1-score
- Confusion matrix

### Prediction
- Predict sentiment for new, unseen text inputs

---

## Dataset Format

Training data should be in CSV format:

```csv
text,sentiment
"I love this product!",positive
"This is terrible.",negative
"The movie was okay.",neutral