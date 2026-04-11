# German Sentiment Analysis Classifier

## Overview
This project implements a machine learning-based **sentiment analysis classifier for German text**.  
It predicts whether a given sentence expresses **positive**, **negative**, or **neutral** sentiment.

The system is built using classical NLP techniques (**TF-IDF + Logistic Regression**) and integrates a German-specific preprocessing pipeline designed for linguistic structure and token normalization.


## Features

### Text Preprocessing
- Lowercasing and Unicode normalization  
- Tokenization (spaCy German model or fallback methods)  
- Stopword removal (German stopword list)  
- Lemmatization (spaCy / fallback support)  
- Text cleaning (removal of punctuation, URLs, and noise)


### Feature Engineering
- TF-IDF vectorization of cleaned German text  
- Support for unigrams and bigrams  
- Configurable feature space for experimentation and scaling  


### Machine Learning Model
- Logistic Regression classifier (baseline model)
- Multi-class sentiment classification:
  - Positive
  - Negative
  - Neutral
- Easily extendable to:
  - Linear SVM
  - Naive Bayes
  - Transformer-based models (future upgrade)


### Evaluation
- Accuracy
- Precision / Recall / F1-score
- Confusion matrix analysis
- Class-wise performance tracking


### Prediction
- Single-text sentiment prediction
- Batch prediction support
- Probability scoring for interpretability



## Dataset Format

Training data must be in CSV format:

```csv
text,sentiment
"Ich liebe dieses Produkt!",positive
"Das ist schrecklich.",negative
"Der Film war okay.",neutral