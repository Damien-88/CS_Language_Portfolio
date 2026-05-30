# Sentiment Analysis

## Project Overview
This project presents a multilingual sentiment analysis system engineered for controlled cross-language comparison across English, German, Russian, and a combined multilingual setting. The implementation emphasizes consistency in model architecture while allowing language-specific preprocessing behavior where linguistic structure requires it.

The core deliverable is a set of production-style classifier pipelines that predict three sentiment classes: negative, neutral, and positive.

## Problem Statement
Sentiment modeling quality is strongly constrained by linguistic variability across languages. Two recurring failure modes are central in this project:

1. Lexical and morphological sparsity
German compounding and Russian inflection increase surface-form diversity, making feature distributions sparser than in lower-morphology settings such as English.

2. Cross-language representation mismatch
In multilingual training, shared vector spaces can mix structurally different lexical signals, causing feature collisions and weaker class separability.

In practical ML workflows, these issues can reduce:
- classification stability across domains,
- robustness to negation and context-poor inputs,
- interpretability of model behavior across languages.

## Linguistic Motivation
Sentiment is not encoded uniformly across languages. The same polarity signal can be realized through very different morphosyntactic patterns.

### Key cross-linguistic properties
1. English
Sentiment cues are often explicit at the lexical level and comparatively stable under light normalization.

2. German
Compounding and inflection can compress sentiment-bearing material into fewer but denser tokens, amplifying feature sparsity.

3. Russian
Inflectional morphology and syntactic flexibility increase surface variability, requiring stronger normalization to preserve sentiment signal.

4. Multilingual mixture
Joint training introduces vocabulary overlap noise and language interference effects that do not appear in monolingual pipelines.

## Methods and Models
The project uses a unified modeling backbone with language-aware preprocessing, enabling direct comparison between monolingual and multilingual configurations.

### 1. Language-specific preprocessing pipelines
- Dedicated preprocessors for English, German, and Russian normalize text according to language-specific constraints.
- A multilingual pipeline applies explicit language routing during training and inference.

### 2. Feature extraction
- TF-IDF vectorization with unigrams and bigrams.
- Controlled vocabulary size with max_features set to 10000.

### 3. Classification and calibration
- Logistic Regression with class balancing and high-iteration convergence settings.
- CalibratedClassifierCV with sigmoid calibration for improved probability reliability.

### 4. Label schema consistency
All pipelines share a fixed target label set: negative, neutral, positive. This maintains metric comparability across language tracks.

## Error Analysis
The system surfaces language-dependent and cross-language error patterns that are critical for interview and production discussions.

### Negation sensitivity
Short negation patterns such as not good and nicht gut can be inconsistently represented after preprocessing, especially in noisy text.

### Morphological ambiguity
German and Russian forms may hide sentiment-bearing lemmas behind inflection or compounding, reducing feature clarity.

### Neutral class overprediction
Low-context inputs and short utterances frequently collapse into neutral due to weak polarity evidence.

### Multilingual interference
Shared feature space in the multilingual model can blur sentiment boundaries between languages with different lexical distributions.

## Cross-Language Observations
Model behavior is shaped as much by linguistic structure as by classifier choice.

1. English
Lower morphological variance typically yields cleaner lexical sentiment features and stronger baseline separability.

2. German and Russian
Higher morphological complexity increases token variability, making preprocessing quality a dominant performance factor.

3. Multilingual setting
A common architecture is feasible, but language-aware preprocessing alignment is essential to control interference and preserve class signal.

This project therefore functions as a practical framework for studying how linguistic structure affects classical ML sentiment systems under a shared architecture.

## Practical Applications
- Multilingual customer feedback triage and monitoring.
- Product and brand sentiment tracking across regions.
- Baseline benchmarking for cross-lingual NLP model comparison.
- Explainable diagnostics for language-specific preprocessing impact.

## Repository Artifacts
- english_classifier/: training, preprocessing, prediction, and demo assets for English.
- german_classifier/: German sentiment pipeline with language-specific normalization.
- russian_classifier/: Russian sentiment pipeline with Cyrillic-aware preprocessing.
- multilingual_classifier/: joint-language sentiment pipeline with explicit language routing.
- Sentiment_Analysis.docx: interview-oriented technical study guide.

## Reproducibility Notes
Recommended environment:
- Python 3.10+
- scikit-learn for vectorization, classification, and calibration
- pandas for dataset handling
- notebook stack for exploratory diagnostics

Run each classifier's training and prediction scripts within its subdirectory to ensure artifact paths resolve correctly.

## Conclusion
This project demonstrates that consistent model architecture does not guarantee consistent cross-language behavior. By combining shared ML foundations with language-aware preprocessing, the system provides a strong, interpretable baseline for multilingual sentiment analysis and a practical platform for error analysis in linguistically diverse settings.