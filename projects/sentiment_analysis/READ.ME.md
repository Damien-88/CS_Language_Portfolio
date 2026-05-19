# Sentiment Analysis Project

## Overview
This project focuses on building and comparing sentiment analysis classifiers across multiple language scopes:
- English
- German
- Russian
- Multilingual (English + German + Russian)

Each classifier predicts one of three sentiment classes:
- negative
- neutral
- positive

The primary goal is to create language-specific and multilingual pipelines that share a consistent machine learning foundation while using preprocessing tailored to each language.

---

## Linguistic Motivation
Sentiment classification behaves differently across languages due to structural linguistic variation:

- English: relatively low morphological complexity, with sentiment often expressed through explicit adjectives and adverbs.
- German: compound words and inflection can compress sentiment-bearing information into single tokens, affecting feature sparsity in TF-IDF space.
- Russian: rich inflectional morphology encodes sentiment cues through case, aspect, and verb conjugation, increasing variability in surface forms.
- Multilingual setting: combining languages introduces vocabulary overlap noise and structural heterogeneity, making preprocessing alignment critical.

This project investigates how a consistent ML model performs under these differing linguistic conditions.

---

## Classifier Collection

### English Classifier
Path: `english_classifier/`
- Uses English preprocessing and lexical normalization.
- Trained with TF-IDF features and calibrated Logistic Regression.
- Includes training, prediction, and notebook demo workflows.

### German Classifier
Path: `german_classifier/`
- Uses German-specific preprocessing (including umlaut-safe cleaning and German stopwords).
- Trained with TF-IDF features and calibrated Logistic Regression.
- Includes training, prediction, and notebook demo workflows.

### Russian Classifier
Path: `russian_classifier/`
- Uses Russian-specific preprocessing (Cyrillic-aware cleaning and Russian stopwords).
- Trained with TF-IDF features and calibrated Logistic Regression.
- Includes training, prediction, and notebook demo workflows.

### Multilingual Classifier
Path: `multilingual_classifier/`
- Uses a language-aware preprocessing pipeline with support for English, German, and Russian.
- Supports per-language preprocessing during training and inference.
- Trained with TF-IDF features and calibrated Logistic Regression.
- Includes class-based trainer/predictor modules and multilingual demo notebook workflows.

---

## Shared Modeling Approach
Across the classifiers, the modeling approach is intentionally consistent:
- Feature extraction: TF-IDF (`max_features=10000`, unigrams + bigrams)
- Model: Logistic Regression (`class_weight="balanced"`, `max_iter=2000`, `C=1.5`)
- Probability calibration: `CalibratedClassifierCV(method="sigmoid", cv=3)`
- Labels: `negative`, `neutral`, `positive`

This consistency makes performance and behavior comparisons across language pipelines easier.

---

## Cross-Language Analysis
Because all classifiers share the same modeling backbone, this project enables controlled comparison across languages.
Key comparative dimensions include:
- Feature sparsity effects: Morphologically rich languages (German, Russian) tend to produce higher lexical variability than English.
- Tokenization sensitivity: Preprocessing differences have a larger impact on German and Russian due to compounding and inflection.
- Multilingual interference: The combined model may introduce vocabulary collision and reduced class separability across languages.
- Model robustness: Comparing performance across languages highlights whether performance gaps are driven by linguistic structure or dataset characteristics.

This setup allows the project to function as a simplified cross-linguistic NLP experiment framework.

---

## Repository Purpose
This directory serves as a language portfolio for sentiment analysis system creation.
It demonstrates:
- End-to-end classifier development per language
- Preprocessing strategy differences by language
- A progression from single-language systems to a multilingual classifier
- Practical training, prediction, and notebook diagnostics workflows

---

## Error Analysis Perspective

Across languages, common failure modes include:
- Negation handling errors
    - “not good” (EN), “nicht gut” (DE), and Russian negation structures may be inconsistently captured depending on tokenization.
- Morphological ambiguity
    - Russian inflection and German compounding can obscure sentiment-bearing root forms.
- Neutral class confusion
    - Short or context-poor sentences often collapse into neutral due to insufficient lexical signal.
- Cross-language noise in multilingual model
    - Shared vector space introduces ambiguity between similar surface tokens with different sentiment distributions across languages.

These errors are primarily driven by linguistic structure rather than model architecture.

---

## Typical Workflow
1. Prepare or update language dataset CSV files.
2. Run the language-specific (or multilingual) training script.
3. Save model and vectorizer artifacts in each classifier's `data/` directory.
4. Run prediction modules for single, probability, or batch inference.
5. Use demo notebooks for visual diagnostics and quick evaluation.

---

## Notes
- Utility modules are maintained as backup helpers; active runtime paths use each classifier's main preprocessing and prediction modules.
- The multilingual classifier is designed to align training-time and prediction-time preprocessing behavior for each language.