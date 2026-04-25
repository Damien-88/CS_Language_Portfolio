# CS_Language_Portfolio

Portfolio showcasing projects in Computer Science, Computational Linguistics, and Human Language Technology (English, Russian, German).


# HLT / Computational Linguistics Portfolio (CS + ML + RU/DE/EN)

This portfolio explores multilingual NLP systems using a **project-centric, modular architecture**.

Each project is designed with a consistent internal structure:

- `english/` → English-language implementation
- `russian/` → Russian-language implementation
- `german/` → German-language implementation
- `multilingual/` → unified cross-lingual integration layer

This reflects a computational linguistics methodology:

> language-specific modeling → comparative linguistic analysis → multilingual fusion systems


# Portfolio Roadmap


## Phase 1: Foundations

- [Text Preprocessor](projects/text_preprocessor/)
- [Sentiment Analysis](projects/sentiment_analysis/)

**Inside each project:**
- `english/`
- `russian/`
- `german/`
- `multilingual/`

**Focus:**
- Tokenization, normalization, stemming/lemmatization
- Language-specific preprocessing differences
- Morphological variation handling
- Baseline sentiment classification per language


## Phase 2: Linguistic Modeling

- [Morphological & Syntax Analysis System](projects/morphological_syntax_analysis/)
- [Compound & Morphology Study System](projects/compound_morphology_analysis/)

**Inside each project:**
- `english/`
- `russian/`
- `german/`
- `multilingual/`

**Focus:**
- English: syntactic parsing + baseline morphological structure
- Russian: case system, aspect, inflectional morphology
- German: compound decomposition + declension patterns

**Comparative Linguistics Layer:**
- cross-language morphological complexity comparison
- linguistic error pattern analysis
- rule-based vs statistical system performance comparison


## Phase 3: Applied NLP + Semantic Search

- [Machine Translation System](projects/machine_translation/)
- [Cross-Lingual Semantic Search Engine](projects/cross_lingual_search/)

**Inside each project:**
- `english/`
- `russian/`
- `german/`
- `multilingual/`

**Focus:**
- Transformer-based translation (EN ↔ RU ↔ DE)
- Multilingual embedding models (Sentence-BERT / LaBSE)
- Cross-lingual semantic retrieval
- Representation alignment across languages


## Phase 4: Advanced Language Intelligence

- [Named Entity Recognition System](projects/named_entity_recognition/)
- [Speech-to-Text Pipeline](projects/speech_to_text/)
- [Dependency Parsing & Syntax Analysis System](projects/dependency_parsing/)

**Inside each project:**
- `english/`
- `russian/`
- `german/`
- `multilingual/`

**Focus:**
- multilingual entity extraction (NER)
- syntactic variation analysis across languages
- ASR systems (wav2vec2 / Whisper)
- phonetic vs morphological error analysis


## Phase 5: Flagship Capstone — Multilingual Intelligence Systems

- [Multilingual Misinformation Detection System](projects/misinformation_detection/)
- [Cross-Lingual Knowledge Graph Platform](projects/cross_lingual_knowledge_graph/)

**Inside each project:**
- `english/`
- `russian/`
- `german/`
- `multilingual/`

**Focus:**
- language-specific misinformation patterns
- multilingual entity extraction and normalization
- graph-based knowledge representation (Neo4j)
- cross-lingual query system (English interface → RU/DE sources)


# Skills Highlighted

## Programming & Systems
- Python
- Linux
- Git / GitHub
- Docker

## NLP & Machine Learning
- Hugging Face Transformers
- spaCy / Stanza
- NLTK
- PyTorch / TensorFlow
- Sentence-BERT / LaBSE

## Data & Search Systems
- SQL
- Elasticsearch
- FAISS / vector databases

## Knowledge Representation
- Neo4j (Graph Databases)
- Entity linking
- Cross-lingual alignment systems

## Languages
- English (baseline NLP system)
- Russian (morphologically rich system)
- German (compound + syntax-heavy system)

## Standard Project Structure

Each project is organized as follows:

projects/<project_name>/

├── english/           # English implementation
├── russian/           # Russian implementation
├── german/            # German implementation
├── multilingual/      # Cross-lingual integration layer

├── shared/
│   ├── evaluation/    # Metrics, scoring, benchmarks
│   ├── utils/         # Shared preprocessing & helpers

└── README.md          # Project documentation


# Each Project README Includes:

- Problem Statement  
- Language Scope (EN / RU / DE / Multilingual)  
- Linguistic Motivation  
- Methods & Models  
- Evaluation Metrics  
- Error Analysis (language-specific comparisons)  
- Cross-Language Observations  
- Key Results  
- Limitations & Future Work  


# Portfolio Goal

This portfolio demonstrates the ability to:

- Build modular multilingual NLP systems
- Model linguistic differences computationally
- Compare language-specific NLP behavior scientifically
- Integrate systems into unified cross-lingual architectures
- Bridge computational linguistics and applied NLP engineering