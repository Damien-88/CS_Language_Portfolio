# German Compound Decomposition

## Project Overview
This project presents a linguistically grounded and engineering-oriented system for decomposing German compound nouns into interpretable morphemic units. The implementation combines rule-aware recursive segmentation with statistical path scoring to handle ambiguity in morphologically rich input.

The core deliverable is a production-ready Python class, GermanCompoundDecomposer, designed for downstream NLP pipelines that require robust preprocessing for German lexical compounding.

## Problem Statement
Standard tokenization strategies often underperform on German due to the productivity of compounding. Two common failure modes are especially relevant in modern NLP systems:

1. Lexical sparsity
German permits highly productive noun formation, generating many low-frequency and one-off surface forms. Even large corpora provide sparse evidence for long-tail compounds.

2. Out-of-vocabulary behavior
Basic subword or whitespace-driven tokenizers can fragment compounds into segments that are orthographically plausible but semantically unhelpful. This increases OOV-style errors at the lexical-semantic level, even when byte-pair segmentation technically covers the string.

In practical transformer settings, naive tokenization can obscure constituent meaning and degrade:
- retrieval recall for semantically related forms,
- translation alignment for compositional nouns,
- interpretability in feature attribution and error tracing.

## Linguistic Motivation
German noun compounds are prototypical Determinativkomposita: the rightmost constituent acts as the semantic and syntactic head, while left constituents incrementally restrict or specify meaning.

### Key morphosyntactic properties
1. Right-headedness
Parsing decisions should prioritize right-to-left structure building, since head identification is central to semantic interpretation.

2. Inter-morphemic linking elements
Compounds may include Fugenlaute such as s, es, n, en, and er at morpheme boundaries. These elements complicate boundary detection because they are not always independent lexical stems.

3. Surface alternations
Constituents may appear with orthographic or stem-level variation, including umlaut alternations and reduced forms. A decomposition system must treat these as linguistically licensed realizations, not noise.

## Methods and Models
The project implements a hybrid decomposition strategy that combines symbolic structure constraints with lightweight probabilistic scoring.

### 1. Recursive right-headed parsing
- Candidate splits are generated recursively with right-headed preference.
- Each candidate path preserves component order and type annotations.
- Fugenlaut-aware boundary logic is applied during split expansion.

### 2. Lexical validation layer
- Components are validated against dictionary lemmas and optionally spaCy lexical cues.
- Recovery variants support common German alternations to improve recall on non-canonical surface forms.

### 3. Statistical path scoring
- A unigram-based frequency model scores candidate decomposition tracks.
- If no explicit corpus frequency file is provided, a local fallback frequency dictionary supplies baseline priors for common roots.
- Candidate ranking prioritizes high-probability lemma paths over arbitrary orthographic fragmentations.

### 4. Confidence normalization
Final confidence values are normalized to the interval from 0.0 to 1.0, enabling consistent interpretation across short and long compounds.

## Error Analysis
The system is intentionally transparent about difficult cases and residual risks.

### Over-splitting and derivational confounds
Forms such as Verhalten can be over-segmented by purely structural heuristics when lexical evidence is weak. This is a classic precision-recall tradeoff: permissive splitting improves coverage but can introduce false morpheme boundaries.

### Structural ambiguity
Some strings admit multiple analyses with distinct semantics. A representative example is Staubecken, where competing interpretations can be generated from superficially plausible split points. Statistical path scoring mitigates this by preferring decomposition paths whose base lemmata are more probable under the frequency model.

### Rare domain terms
Specialized legal, biomedical, or technical compounds may require domain-adapted lexicons or frequency priors to avoid under-segmentation.

## Cross-Language Observations
German and English differ in where compositional complexity is encoded.

1. German
Compositional structure is frequently realized inside single orthographic words through nominal compounding. Syntax is partially shifted into morphology.

2. English
Equivalent content is more often distributed across multiword phrasal constructions, where syntactic boundaries remain explicit.

For multilingual NLP, this mismatch matters: alignment systems must map English phrase-level units to German intra-word morphology. Compound decomposition therefore functions as a cross-lingual normalization bridge, improving both interpretability and model transfer.

## Practical Applications
- Machine Translation: improved constituent alignment and lexical choice.
- Information Retrieval: better indexing of head and modifier semantics.
- Terminology Mining: clearer extraction of domain concepts from long compounds.
- Explainable NLP: component-level diagnostics for error investigation.

## Repository Artifacts
- german_compound_decomposer.py: production implementation of decomposition logic and scoring.
- test_decomposer.py: pytest suite covering core linguistic archetypes and edge cases.
- decomposer_demo.ipynb: interactive portfolio walkthrough with examples and ambiguity visualization.

## Reproducibility Notes
Recommended environment:
- Python 3.10+
- Optional spaCy model for lexical validation enhancement
- pandas and matplotlib for notebook analytics and visualization

Install core dependencies as needed in your environment and run tests with pytest to validate behavior.

## Conclusion
This project demonstrates that linguistically informed decomposition, when coupled with lightweight probabilistic ranking, can materially improve German text preprocessing quality. The resulting pipeline is suitable both as a research artifact in computational linguistics and as a practical component in production NLP stacks for morphologically rich languages.