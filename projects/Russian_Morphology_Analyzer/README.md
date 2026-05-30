# Russian Morphology Analyzer

## Project Overview
This project presents a linguistically grounded Russian morphological analysis system designed to model rich inflectional structure in production-style NLP workflows. The implementation combines rule-based analysis, dictionary resources, and statistical ambiguity handling to produce interpretable morphological outputs.

The core deliverable is a modular analyzer pipeline that extracts lemmas and grammatical features for words, sentences, and batch inputs.

## Problem Statement
Russian inflectional complexity creates recurrent challenges for standard token-first NLP pipelines. Two core failure modes are especially relevant:

1. Surface-form variability and lexical sparsity
Russian lexical items can produce many inflected forms, fragmenting evidence across paradigms and reducing statistical consistency in downstream models.

2. Morphological ambiguity under limited context
Single surface forms can correspond to multiple grammatical analyses, and context-independent parsing may overgenerate candidate interpretations.

In practical systems, these issues can reduce:
- lemmatization reliability,
- grammatical tagging accuracy,
- robustness in downstream tasks such as parsing, retrieval, and MT alignment.

## Linguistic Motivation
Russian grammatical information is encoded extensively in morphology rather than fixed word order or isolated function words.

### Key morphosyntactic properties
1. Inflectional density
Nouns, adjectives, and verbs carry rich grammatical information through endings and stem alternations.

2. Case and agreement systems
Robust analysis requires explicit handling of case, number, gender, and agreement interactions.

3. Aspect and conjugation behavior
Verb interpretation often depends on tense, aspect, and conjugational patterns that are not trivially recoverable from surface forms.

4. Ambiguity under token-level analysis
Forms such as стола can map to multiple grammatical interpretations without sentential cues.

## Methods and Models
The project implements a hybrid architecture that combines symbolic linguistic constraints with statistical heuristics.

### 1. Tokenization and normalization
- Cyrillic-safe preprocessing with Unicode-aware handling.
- Punctuation and boundary normalization for reliable downstream rule application.

### 2. Lemmatization and lexical recovery
- Suffix and rule-driven candidate generation.
- Dictionary-backed fallback for irregular and non-trivial paradigms.

### 3. Morphological rule application
- Declension and conjugation rule matching.
- Feature inference for case, number, gender, tense, aspect, and part of speech.

### 4. Candidate ranking and extraction
- Heuristic/statistical ranking for ambiguous analyses.
- Structured output generation for word-level and sentence-level workflows.

## Error Analysis
The system is designed to expose difficult morphology cases and provide interpretable failure categories.

### Morphological ambiguity
Certain forms map to multiple valid analyses. Without broader context, ambiguity may persist across equally plausible candidates.

### Irregular inflection
Irregular paradigms and suppletive forms can reduce rule coverage and increase dictionary dependency.

### Context-sensitive disambiguation limits
Token-level analysis may miss sentence-level syntactic constraints needed for final interpretation.

### Coverage dependence
Performance is sensitive to morphology dictionary breadth and exception-list quality.

## Cross-Language Observations
Russian morphology demonstrates a different complexity profile from lower-inflection languages and complements comparative work in German and multilingual projects.

1. Russian vs English
Russian encodes substantially more grammatical information morphologically, requiring deeper feature extraction before modeling.

2. Russian vs German
Both are morphology-rich, but Russian declensional and aspectual behavior creates distinct ambiguity patterns compared to German compounding-focused complexity.

3. Multilingual relevance
Morphology-aware normalization provides a foundation for consistent cross-lingual preprocessing and improved model alignment.

## Practical Applications
- Morphology-aware preprocessing for Russian NLP pipelines.
- Improved lemmatization and feature extraction for search and IR.
- Linguistically interpretable tagging for educational and diagnostic tools.
- Better upstream normalization for sentiment, classification, and translation systems.

## Repository Artifacts
- analyzer.py: top-level analysis entry points.
- code/analyzer.py: orchestrated analysis pipeline.
- code/tokenizer.py: Russian-aware tokenization and normalization.
- code/lemmatizer.py: lemma recovery logic.
- code/morphology_rules.py: grammatical rule set and feature inference.
- code/feature_extractor.py: structured feature output formatting.
- code/evaluate.py: evaluation pipeline.
- code/demo_ru.ipynb: interactive demonstration and diagnostics.
- data/: dictionary resources, evaluation data, and output artifacts.

## Reproducibility Notes
Recommended environment:
- Python 3.10+
- pymorphy3 and Russian dictionary resources
- regex-compatible text preprocessing stack
- notebook tooling for exploratory diagnostics and visualization

Install dependencies from requirements and run evaluation scripts to reproduce baseline metrics.

## Conclusion
This project demonstrates that Russian morphology analysis benefits from a hybrid strategy combining explicit linguistic rules, lexical resources, and statistical ranking. The resulting system offers interpretable structure, practical robustness, and direct utility for downstream NLP tasks in morphology-rich language settings.