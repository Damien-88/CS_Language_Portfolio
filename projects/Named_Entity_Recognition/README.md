# Named Entity Recognition

## Overview

This project develops a **multilingual Named Entity Recognition (NER) system** for English, German, and Russian.

The system identifies and classifies entities such as:

* **PERSON** — people and names
* **ORG** — organizations and institutions
* **LOC** — locations
* **GPE** — geopolitical entities
* **DATE** — dates and temporal expressions
* **EVENT** — named events
* **PRODUCT** — products and named artifacts

The project treats NER as both an NLP engineering problem and a **computational linguistics research problem**.

The central goal is to investigate how multilingual transformer-based NER systems represent and extract entities across languages with different:

* morphological systems
* word-order patterns
* capitalization conventions
* inflectional behavior
* compound formation
* named-entity conventions

The project therefore combines model-based entity extraction with **cross-linguistic error analysis and entity normalization**.

---

# Research Motivation

Named entities are important intermediate representations for many NLP systems.

They connect unstructured language to structured information and are particularly important for:

* information extraction
* semantic search
* question answering
* knowledge graphs
* document classification
* information retrieval
* multilingual information access

This project follows the progression of the portfolio's earlier work.

Previous projects examined:

```text
Text preprocessing
        ↓
Sentiment classification
        ↓
Morphological analysis
        ↓
Syntactic structure
        ↓
Semantic representation
        ↓
Cross-lingual semantic retrieval
        ↓
Named entity recognition
        ↓
Knowledge representation
```

NER provides a natural bridge between **semantic retrieval** and the later **Cross-Lingual Knowledge Graph** project.

The system should therefore not only answer:

> "Which words refer to entities?"

but also investigate:

> "How does language-specific structure affect entity recognition and representation?"

---

# Research Questions

The project investigates several controlled questions.

### Q1 — Multilingual Entity Recognition

Can transformer-based NER models reliably identify common entity types across English, German, and Russian?

### Q2 — Cross-Linguistic Differences

How do linguistic differences between English, German, and Russian affect entity recognition?

Examples include:

* capitalization
* grammatical case
* inflection
* compound nouns
* word order
* transliteration
* multi-word entities

### Q3 — Entity Boundary Detection

How frequently do models identify the correct entity boundaries?

For example:

```text
New York University
^^^^^^^^^^^^^^^^^^
```

should be treated as one organization rather than several independent tokens.

### Q4 — Entity Type Confusion

Which entity categories are most frequently confused?

Examples:

```text
ORG ↔ GPE
LOC ↔ GPE
PERSON ↔ ORG
EVENT ↔ ORG
```

### Q5 — Entity Normalization

Can extracted entities be normalized into consistent representations across languages?

For example:

```text
United States
Vereinigte Staaten
Соединённые Штаты
```

may refer to the same underlying entity.

This question provides a direct connection to later entity linking and knowledge-graph construction.

---

# System Architecture

The initial architecture is:

```text
                 Input Text
                     │
                     ▼
             Language Detection
                     │
                     ▼
             Tokenization
                     │
                     ▼
          Multilingual NER Model
                     │
                     ▼
             Entity Extraction
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
    Entity Classification   Entity Spans
          │                     │
          └──────────┬──────────┘
                     ▼
             Entity Normalization
                     │
                     ▼
          Cross-Lingual Analysis
                     │
                     ▼
              Error Analysis
```

The system should keep extraction, normalization, evaluation, and analysis as separate components.

This makes it possible to compare model behavior without coupling the research experiments to a single implementation.

---

# Core Components

## 1. Model Loader

`ner/model_loader.py`

Responsible for loading the selected transformer-based NER model.

Responsibilities:

* model initialization
* tokenizer initialization
* model configuration
* device selection
* inference configuration

The model should support multilingual input where practical.

Potential model families include:

* multilingual BERT
* XLM-RoBERTa
* multilingual transformer NER models

The exact model should be selected during implementation and documented with its language coverage and limitations.

---

## 2. Entity Extractor

`ner/entity_extractor.py`

Provides a consistent interface for NER inference.

Conceptually:

```text
text
 ↓
tokenization
 ↓
model inference
 ↓
token-level predictions
 ↓
entity span reconstruction
 ↓
structured entities
```

A returned entity should contain information such as:

```text
text
entity_type
start
end
score
```

Example:

```text
{
    "text": "Angela Merkel",
    "entity_type": "PERSON",
    "start": 0,
    "end": 13,
    "score": 0.97
}
```

The implementation should preserve character offsets whenever possible because entity boundaries are important for later evaluation and analysis.

---

# 3. Entity Normalizer

`ner/entity_normalizer.py`

The normalizer converts extracted entity mentions into more consistent representations.

Normalization may include:

* whitespace normalization
* Unicode normalization
* case normalization where appropriate
* punctuation handling
* transliteration analysis
* language-specific normalization

The normalizer should **not automatically assume that two similar strings refer to the same real-world entity**.

For example:

```text
Washington
Washington State
Washington, D.C.
George Washington
```

cannot safely be merged solely through string similarity.

The initial project therefore focuses on **mention normalization**, while full entity linking is left for later work.

---

# 4. Error Analysis

`analysis/entity_errors.py`

NER errors should be categorized linguistically rather than simply counted.

Initial categories include:

### Boundary Error

The model identifies the wrong span.

```text
Expected:
[New York University]

Predicted:
[New York]
```

### Entity Type Error

The entity span is approximately correct, but its category is wrong.

```text
Expected: ORG
Predicted: GPE
```

### Missed Entity

A real entity is not detected.

```text
Expected:
[Angela Merkel] — PERSON

Predicted:
nothing
```

### Spurious Entity

The model predicts an entity where none exists.

### Morphological Variation

Inflection changes the surface form of an entity.

This is particularly relevant for Russian and German.

### Compound Structure

German compound nouns may contain entity-related information that is difficult to separate from surrounding lexical material.

### Capitalization Variation

Capitalization provides different amounts of information across languages.

### Cross-Lingual Variation

Equivalent entities may appear with different:

* spellings
* grammatical forms
* transliterations
* word order
* abbreviations

---

# 5. Linguistic Analysis

`analysis/linguistic_analysis.py`

This component connects model errors to linguistic phenomena.

The analysis should investigate questions such as:

* Does inflection affect entity recognition?
* Are multi-word entities more difficult than single-word entities?
* Are German compounds problematic?
* Does capitalization provide a useful signal?
* Are Russian case forms associated with missed entities?
* Does entity word order vary between languages?
* Are transliterated entities recognized consistently?

The goal is not simply to report:

```text
F1 = 0.82
```

but to explain **why particular errors occur**.

---

# Experiments

The project should contain controlled experiments corresponding to the research questions.

## Experiment 1 — Multilingual NER

**Question:** How does the NER system perform across English, German, and Russian?

Measure:

* precision
* recall
* F1
* entity-level performance
* language-level performance

Where appropriate, report results separately for entity types.

---

## Experiment 2 — Cross-Lingual Entity Recognition

**Question:** How consistently are equivalent entities recognized across languages?

Use multilingual examples containing equivalent or closely related entity mentions.

For example:

```text
English:
Angela Merkel visited Berlin.

German:
Angela Merkel besuchte Berlin.

Russian:
Ангела Меркель посетила Берлин.
```

Compare:

* detected entities
* entity types
* boundaries
* confidence
* surface-form differences

---

## Experiment 3 — Entity Boundary Analysis

**Question:** How does entity complexity affect boundary detection?

Compare categories such as:

```text
single-token entities
multi-token entities
nested-looking expressions
entities containing punctuation
entities containing abbreviations
```

The experiment should identify which structures produce boundary errors.

---

## Experiment 4 — Morphological and Orthographic Effects

**Question:** How do language-specific surface forms affect NER?

Construct controlled examples involving:

* Russian case variation
* German compounds
* capitalization variation
* inflected forms
* punctuation
* transliteration

Compare recognition before and after controlled changes.

The objective is to isolate linguistic effects rather than simply report corpus-level performance.

---

## Experiment 5 — Entity Normalization

**Question:** Can entity mentions from different languages be represented consistently?

Compare multilingual mentions that refer to the same underlying concept or entity.

For example:

```text
United Nations
Vereinte Nationen
Организация Объединённых Наций
```

The experiment should distinguish:

1. surface-form normalization
2. cross-lingual equivalence
3. true entity identity

The third category is intentionally more difficult and provides groundwork for the future knowledge-graph project.

---

# Evaluation

NER evaluation should use entity-level metrics where possible.

Primary metrics:

* Precision
* Recall
* F1

Additional analysis:

* per-language F1
* per-entity-type F1
* boundary accuracy
* confusion patterns
* missed entities
* spurious entities

The project should avoid relying exclusively on token-level accuracy because correct entity recognition depends on both:

1. identifying the correct span
2. assigning the correct entity type

---

# Error Analysis Framework

A central output of this project is a structured error analysis.

Each analyzed error should contain information such as:

```text
language
input_text
expected_entity
predicted_entity
expected_type
predicted_type
error_category
linguistic_explanation
```

Example:

```text
Language: Russian

Input:
Президент посетил Москву.

Expected:
Москва — LOC

Prediction:
Москва — GPE

Error:
Entity type confusion

Linguistic interpretation:
The surface form is correctly identified as an entity,
but the model distinguishes geographic location and
geopolitical entity inconsistently.
```

The exact interpretation should be based on observed model behavior rather than assumptions.

---

# Demonstrations

The notebooks provide reproducible demonstrations of the system.

## `ner_demo.ipynb`

Basic end-to-end NER:

```text
text
→ model
→ entities
→ entity types
→ confidence scores
```

---

## `multilingual_ner.ipynb`

Run equivalent examples in:

* English
* German
* Russian

Compare extracted entities and entity types.

---

## `entity_error_analysis.ipynb`

Inspect model failures and classify them using the project's error-analysis framework.

---

## `cross_lingual_entities.ipynb`

Explore equivalent entity mentions across languages and investigate normalization behavior.

---

# Project Structure

```text
Named_Entity_Recognition/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── ner/
│   ├── __init__.py
│   ├── model_loader.py
│   ├── entity_extractor.py
│   └── entity_normalizer.py
│
├── analysis/
│   ├── __init__.py
│   ├── entity_errors.py
│   └── linguistic_analysis.py
│
├── experiments/
│   ├── __init__.py
│   ├── multilingual_ner.py
│   ├── cross_lingual_ner.py
│   └── entity_normalization.py
│
├── demos/
│   ├── ner_demo.ipynb
│   ├── multilingual_ner.ipynb
│   ├── entity_error_analysis.ipynb
│   └── cross_lingual_entities.ipynb
│
├── tests/
│   ├── test_entity_extractor.py
│   ├── test_entity_normalizer.py
│   └── test_ner_experiments.py
│
├── config.py
├── README.md
└── requirements.txt
```

---

# Technologies

Primary technologies:

* Python 3.10+
* PyTorch
* Hugging Face Transformers
* Hugging Face Tokenizers
* pandas
* NumPy
* scikit-learn
* matplotlib
* Jupyter

Potential NLP libraries:

* spaCy
* Stanza

The final dependency set should reflect the actual implementation rather than adding libraries that are not used.

---

# Testing Strategy

Tests should cover the system independently from the notebooks.

Initial test areas:

### Entity Extraction

* entities are returned in the expected structure
* entity types are preserved
* character offsets are valid
* empty input is handled
* multiple entities can be extracted

### Normalization

* Unicode normalization
* whitespace normalization
* punctuation handling
* preservation of meaningful distinctions

### Experiments

* evaluation metrics
* language-specific inputs
* batch processing
* error categorization
* normalization comparisons

Tests should use deterministic fixtures or mocked model outputs where possible.

The objective is to prevent model availability or external downloads from making the core test suite unreliable.

---

# Reproducibility

Experiments should document:

* model name
* model version where relevant
* language
* input data
* evaluation methodology
* random seed where applicable
* hardware/device where relevant

Results should be reproducible from the repository whenever external model downloads are available.

---

# Linguistic Scope

The three primary languages are:

| Language | Linguistic relevance                                                  |
| -------- | --------------------------------------------------------------------- |
| English  | Relatively limited inflection; strong capitalization cues             |
| German   | Compounding, capitalization, inflection, multi-word structures        |
| Russian  | Rich morphology, case variation, Cyrillic script, flexible word order |

These differences make the three languages useful for controlled cross-linguistic analysis.

The project should avoid treating English as the default linguistic reference point. Each language should be analyzed according to its own structural properties.

---

# Expected Outcomes

The project should produce:

1. A reusable multilingual NER inference component.
2. Structured entity extraction with character spans.
3. Entity normalization utilities.
4. Quantitative multilingual NER evaluation.
5. Cross-lingual entity comparisons.
6. Linguistically motivated error categories.
7. Reproducible experiments and notebooks.
8. Unit tests for the core components.
9. Evidence about how morphology, orthography, compounds, and cross-lingual variation affect NER.

The primary research outcome is an understanding of **where multilingual NER succeeds and fails across English, German, and Russian, and how those failures relate to linguistic structure**.

---

# Limitations

This project does not attempt to solve every aspect of entity understanding.

In particular, the initial implementation does not aim to provide:

* complete entity linking
* authoritative knowledge-base resolution
* comprehensive coreference resolution
* exhaustive nested-entity recognition
* production-scale information extraction
* perfect multilingual coverage

Entity linking and structured knowledge representation are intentionally reserved for the later **Cross-Lingual Knowledge Graph** project.

---

# Future Work

Potential extensions include:

* entity linking
* Wikidata integration
* multilingual knowledge graphs
* relation extraction
* coreference resolution
* nested entity recognition
* domain-specific NER
* entity-aware semantic search
* integration with the portfolio's cross-lingual retrieval system

These extensions lead toward the Phase 5 goal of constructing a **cross-lingual knowledge representation and query system**.

---

# Relevance to the Portfolio

This project extends the portfolio from semantic retrieval into structured language understanding.

The progression is:

```text
Multilingual text
      ↓
Morphological structure
      ↓
Syntactic structure
      ↓
Semantic representation
      ↓
Semantic retrieval
      ↓
Named entities
      ↓
Entity normalization
      ↓
Knowledge representation
```

The project therefore serves as a bridge between **multilingual NLP** and **knowledge representation**.

---

# Project Status

**Phase:** 4 — Advanced Language Intelligence

**Status:** Scaffolding / Initial Implementation

Planned progression:

* [X] Project scaffolding
* [X] README specification
* [ ] NER model loader
* [ ] Entity extraction
* [ ] Entity normalization
* [ ] Multilingual evaluation
* [ ] Cross-lingual experiments
* [ ] Linguistic error analysis
* [ ] Demonstration notebooks
* [ ] Unit tests
* [ ] README validation
* [ ] Final research summary