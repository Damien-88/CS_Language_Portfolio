# Text Preprocessor Project

## Overview
This project contains reusable preprocessing pipelines for:
- English
- German
- Russian
- Multilingual (English, German, Russian)

The goal is to normalize raw text into cleaned/tokenized/lemmatized outputs that
can be used directly by downstream NLP workflows.

---

## Preprocessor Collection

### English Preprocessor
Path: `english_preprocessor/`
- Main script: `code/preprocess_en.py`
- Helpers: `code/utils.py`
- Output DB: `data/processed_results.db`

### German Preprocessor
Path: `german_preprocessor/`
- Main script: `code/preprocess_de.py`
- Helpers: `code/utils_de.py`
- Output DB: `data/processed_results_de.db`

### Russian Preprocessor
Path: `russian_preprocessor/`
- Main script: `code/preprocess_ru.py`
- Helpers: `code/utils_ru.py`
- Output DB: `data/processed_results_ru.db`

### Multilingual Preprocessor
Path: `multilingual_preprocessor/`
- Main script: `code/preprocess_ml.py`
- Helpers: `code/utils_ml.py`
- Language selection is explicit via command arguments (`--lang`, `--code`)
- Output DB pattern: `data/processed_results_<country_code>.db`

---

## Shared Preprocessing Goals
Across the preprocessors, the core workflow is aligned:
- Unicode normalization and lowercasing
- URL and mention removal
- Language-aware punctuation/symbol filtering
- Tokenization
- Stopword removal
- Lemmatization (with stemming helpers where applicable)

This consistency allows downstream models to consume normalized, vectorizer-ready text while still respecting language differences.

---

## Repository Purpose
This directory serves as a language portfolio for text preprocessing pipeline creation.
It demonstrates:
- Language-specific preprocessing implementations
- Practical multilingual preprocessing design
- Reusable helper utilities for text/file handling
- Notebook-based demonstrations and diagnostics

---

## Typical Workflow
1. Provide raw text input (file or sample text).
2. Run the language-specific or multilingual preprocessing script.
3. Generate cleaned tokens/lemmas or normalized output text.
4. Persist results to SQLite.
5. Use demo notebooks for step-by-step inspection of preprocessing behavior.

---

## Notes
- Each language folder has its own script, helper module, data samples, and notebook demo.
- The multilingual preprocessor centralizes cross-language processing, but language choice is passed explicitly at runtime.