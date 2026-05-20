# Russian Morphology Analyzer

## Overview

This project implements a **Russian morphological analysis system** designed to computationally model the rich inflectional structure of the Russian language.

The analyzer identifies and extracts:
- Lemmas (base forms)
- Part-of-speech tags
- Grammatical case
- Gender
- Number
- Verb tense and aspect
- Morphological feature patterns

The project combines:
- rule-based linguistic analysis
- statistical NLP methods
- morphology-aware preprocessing pipelines

to study how Russian grammatical structure can be represented computationally for downstream NLP tasks.

---

## Linguistic Motivation

Russian is a **morphology-rich language** with extensive inflectional variation.

Unlike English, grammatical information is frequently encoded through:
- suffixes
- inflectional endings
- conjugation patterns
- aspectual verb systems

This creates several computational challenges:
- large surface-form variability
- lexical sparsity
- morphological ambiguity
- context-sensitive interpretation

For example:

| Surface Form | Lemma | Features |
|---|---|---|
| книгами | книга | instrumental plural |
| читаю | читать | 1st person singular present |
| красивого | красивый | masculine/neuter genitive singular |

This project investigates how morphology-aware systems can:
- recover normalized forms
- identify grammatical structure
- reduce ambiguity
- improve downstream linguistic processing

---

## Project Structure

```text
russian_morphology_analyzer/
├── analyzer.py
├── code/
│   ├── __init__.py
│   ├── analyzer.py
│   ├── morphology_rules.py
│   ├── lemmatizer.py
│   ├── tokenizer.py
│   ├── feature_extractor.py
│   ├── evaluate.py
│   ├── utils.py
│   └── demo_ru.ipynb
│
├── data/
│   ├── sample_sentences_ru.txt
│   ├── morphology_dictionary.json
│   ├── irregular_forms.json
│   ├── evaluation_dataset.csv
│   └── outputs/
│
├── requirements.txt
└── README.md
```

---

## System Components

### Tokenization (`tokenizer.py`)
- Russian-aware tokenization
- Unicode normalization
- punctuation handling
- Cyrillic-safe preprocessing

---

### Lemmatization (`lemmatizer.py`)
Converts inflected words into base dictionary forms.

Example:
```text
книгами → книга
читал → читать
лучшего → лучший
```

Methods:
- suffix stripping
- morphological rule matching
- dictionary lookup fallback

---

### Morphological Rules (`morphology_rules.py`)
Encodes linguistic rules for:
- noun declension
- adjective agreement
- verb conjugation
- grammatical case identification
- aspect detection

The rule system models:
- nominative
- genitive
- dative
- accusative
- instrumental
- prepositional

as well as:
- singular/plural variation
- gender agreement
- tense/aspect distinctions

---

### Feature Extraction (`feature_extractor.py`)
Extracts structured morphological information:

Example:
```json
{
  "token": "книгами",
  "lemma": "книга",
  "part_of_speech": "noun",
  "case": "instrumental",
  "number": "plural",
  "gender": "feminine"
}
```

---

### Analysis Pipeline (`analyzer.py`)
Combines:
1. tokenization
2. normalization
3. lemmatization
4. rule matching
5. feature extraction

Supports:
- single-word analysis
- sentence parsing
- batch processing

---

## Example Usage

### Analyze a single word

```python
from analyzer import analyze_word

result = analyze_word("книгами")

print(result)
```

Output:
```json
{
  "token": "книгами",
  "lemma": "книга",
  "part_of_speech": "noun",
  "case": "instrumental",
  "number": "plural",
  "gender": "feminine"
}
```

---

### Analyze a sentence

```python
from analyzer import analyze_sentence

text = "Я читаю интересные книги"

results = analyze_sentence(text)

for item in results:
    print(item)
```

---

## Methods & Models

This project uses a hybrid computational linguistics approach:

### Rule-Based Components
- declension pattern matching
- suffix analysis
- grammatical heuristics
- exception dictionaries

### Statistical / NLP Components
- token frequency analysis
- ambiguity reduction heuristics
- morphological candidate ranking

### NLP Libraries
- pymorphy3
- pymorphy3 dictionaries (Russian)
- regex-based preprocessing
- custom morphology rules
- ipywidgets (optional notebook interactivity)

The demo notebook also includes optional visualization cells that use `matplotlib`.

---

## Demo Notebook

`code/demo_ru.ipynb` includes:
- single-word, sentence, and batch analysis
- evaluation on labeled dataset
- robustness checks (environment + sanity assertions)
- standard morphology visualizations:
  - per-feature accuracy
  - POS confusion matrix
  - analysis-source coverage

---

## Evaluation Metrics

System performance is evaluated using:

- Lemmatization accuracy
- Morphological tagging accuracy
- Case identification accuracy
- POS tagging consistency
- Rule coverage rate

Evaluation datasets include:
- manually annotated Russian sentences
- morphology benchmark examples
- irregular inflection test cases

---

## Error Analysis (Linguistic + ML)

### Morphological Ambiguity
Some Russian forms map to multiple grammatical interpretations.

Example:
```text
стола
```

Possible interpretations:
- genitive singular
- accusative singular

---

### Irregular Inflection
Certain verbs and nouns do not follow standard declension patterns.

Example:
```text
человек → люди
```

---

### Context-Dependent Interpretation
Correct parsing may depend on sentence-level syntax rather than isolated tokens.

---

### Lexical Sparsity
Rich inflection increases vocabulary variability, reducing statistical consistency across forms.

---

## Key Results

The analyzer successfully:
- identifies major Russian grammatical cases
- normalizes inflected forms into lemmas
- extracts interpretable linguistic features
- models morphology-aware token structure

The project demonstrates how rule-based and statistical methods can be combined for morphology-rich NLP systems.

---

## Limitations

- Limited contextual disambiguation
- Rule-based systems may fail on rare irregular forms
- Dependency-level syntax is not fully modeled
- Performance depends on morphology dictionary coverage

---

## Future Work

Potential extensions include:
- transformer-based morphological tagging
- dependency-aware morphology analysis
- probabilistic disambiguation models
- cross-lingual morphology comparison (Russian vs German)
- integration into downstream NLP pipelines

---

## Relevance to NLP / Computational Linguistics

This project demonstrates:
- computational modeling of inflectional morphology
- morphology-aware NLP preprocessing
- rule-based linguistic system design
- hybrid statistical + linguistic analysis methods
- foundations for multilingual syntax and semantic systems

It serves as a core Phase 2 project in the broader multilingual computational linguistics portfolio.