# Evaluation Module - Metrics & Linguistic Analysis

## Overview

A **computational linguistics-focused** evaluation framework combining:
- **Automatic Metrics**: BLEU, chrF scores
- **Linguistic Analysis**: Error patterns, morphological issues, compound handling, word order shifts
- **Structured Reports**: Human-readable summaries with recommendations

**Design**: Portfolio-grade research tool, not production-level benchmarking.


## Core Components

### 1. Automatic Metrics

#### BLEU Score (`BLEUMetric`)

Standard BLEU implementation with n-gram precision and brevity penalty.

```python
from evaluation.metrics import BLEUMetric

bleu_metric = BLEUMetric(max_n=4)

# Single sentence
score = bleu_metric.score(
    hypothesis="Das ist gut",
    references=["Das ist gut", "Das ist schön"]
)
print(score)  # BLEU = 1.0000, precisions = [1.0, 1.0, 1.0, 1.0], ...

# Corpus-level
corpus_score = bleu_metric.corpus_score(
    hypotheses=["Das ist gut", "Das ist schlecht"],
    references_list=[
        ["Das ist gut"],
        ["Das ist schlecht"]
    ]
)
```

**Properties**:
- `score`: Overall BLEU (0-1)
- `precisions`: List of n-gram precisions (1-gram to 4-gram)
- `bp`: Brevity penalty (0-1)
- `ratio`: Length ratio (hypothesis / reference)

#### chrF Score (`ChrFMetric`)

Character n-gram F-score (language-agnostic, good for morphologically-rich languages).

```python
from evaluation.metrics import ChrFMetric

chrf_metric = ChrFMetric(order=6, beta=3.0)

score = chrf_metric.score(
    hypothesis="Das ist ganz wunderbar",
    reference="Das ist wunderbar"
)
print(score)  # chrF = 0.8234, precision = 0.8523, recall = 0.7845
```

**Why chrF**:
- Character-level: captures morphological variations
- Language-agnostic: works well for German, Russian, etc.
- Fewer false negatives: better for paraphrases


### 2. Linguistic Analysis & Reports

#### Error Pattern Extraction

Automatically identifies recurring error patterns from sentence-level analyses.

```python
from evaluation.linguistic_report import LinguisticReportGenerator
from analysis.error_analyzer import TranslationErrorAnalyzer

# Run error analysis on corpus
analyzer = TranslationErrorAnalyzer()
batch_analysis = analyzer.analyze_batch(
    sources=source_texts,
    targets=hypotheses,
    source_lang="en",
    target_lang="de"
)

# Generate linguistic report
report_gen = LinguisticReportGenerator(("en", "de"))
report = report_gen.generate(batch_analysis)
```

#### Linguistic Evaluation Report

Comprehensive report with patterns across 5 linguistic categories:

```
|      Category     |          Description         |              Examples              |
|-------------------|------------------------------|------------------------------------|
| **Morphological** |   Case/gender/number loss    | German adjective-noun disagreement |
|   **Compounds**   |    German compound issues    | "Schmetterling" → "Schmetter ling" |
|   **Agreement**   | Subject-verb, noun-adjective |        Russian case mismatch       |
|   **Word Order**  |      SVO/SOV reordering      |        Verb position shifts        |
|    **Semantic**   |        Meaning changes       |   Negation flips, semantic drift   |
```

**Report Output**:

```python
report.linguistic_summary  # Human-readable text summary
report.recommendations     # Actionable improvement suggestions

# Error patterns (top 5 per category)
report.morphological_patterns
report.compound_patterns
report.agreement_patterns
report.word_order_patterns
report.semantic_patterns
```

Each pattern includes:
- `pattern`: Description
- `count`: Number of occurrences
- `confidence`: 0-1 (frequency / sentence_count)
- `examples`: Up to 3 (source, target) pairs


### 3. Evaluation Pipeline

#### Full Evaluation

Combines metrics and linguistic analysis in one step.

```python
from evaluation.evaluation_pipeline import EvaluationPipeline

pipeline = EvaluationPipeline(
    bleu_max_n=4,
    include_chrf=True,
    include_linguistic=True,
)

result = pipeline.evaluate(
    hypotheses=translations,
    references_list=references,
    source_lang="en",
    target_lang="de",
    sources=source_texts,  # Required for linguistic analysis
)

# Access results
print(f"BLEU: {result.bleu.score:.4f}")
print(f"chrF: {result.chrf.score:.4f}")
print(f"Linguistic Report: {result.linguistic_report.linguistic_summary}")
```

#### Sentence-Level Evaluation

Evaluate individual sentences with full breakdown.

```python
from evaluation.evaluation_pipeline import SentenceEvaluator

evaluator = SentenceEvaluator()

result = evaluator.evaluate_sentence(
    source="The system is broken.",
    hypothesis="Das System ist kaputt.",
    references=["Das System ist defekt.", "Das System funktioniert nicht."],
    source_lang="en",
    target_lang="de",
)

# Access individual metrics
print(f"BLEU: {result['metrics']['bleu']['score']:.4f}")
print(f"Quality: {result['overall_quality']}")
for error in result['linguistic_analysis']['errors']:
    print(f"  - {error['category']}: {error['explanation']}")
```


## Usage Examples

### Example 1: Quick Corpus Evaluation

```python
from evaluation.evaluation_pipeline import EvaluationPipeline

# Your translations
hypotheses = [...]   # List of translated texts
references_list = [[ref1, ref2, ...], ...]  # Multiple references per hypothesis
sources = [...]      # Original source texts

pipeline = EvaluationPipeline(include_linguistic=True)
result = pipeline.evaluate(
    hypotheses=hypotheses,
    references_list=references_list,
    sources=sources,
    source_lang="en",
    target_lang="de"
)

# Print formatted report
pipeline.print_report(result)

# Export to JSON
with open("results.json", "w") as f:
    f.write(result.to_json())
```

### Example 2: Focus on Morphological Issues

```python
from analysis.error_analyzer import TranslationErrorAnalyzer

analyzer = TranslationErrorAnalyzer()

for source, hypothesis, reference in data:
    analysis = analyzer.analyze(
        source=source,
        target=hypothesis,
        reference=reference,
        source_lang="de",  # German (morphologically complex)
        target_lang="en"
    )
    
    morph_errors = [
        e for e in analysis.errors
        if e.category.value == "morphological_loss"
    ]
    
    if morph_errors:
        print(f"Source: {source}")
        print(f"Errors: {[e.explanation for e in morph_errors]}")
```

### Example 3: Compound-Aware Evaluation

```python
from evaluation.evaluation_pipeline import EvaluationPipeline

# With German
pipeline = EvaluationPipeline(include_linguistic=True)
result = pipeline.evaluate(
    hypotheses=de_translations,
    references_list=de_references,
    sources=en_sources,
    source_lang="en",
    target_lang="de"
)

# Check compound patterns
report = result.linguistic_report
for pattern in report.compound_patterns:
    print(f"{pattern.pattern} ({pattern.count}x)")
```


## Output Formats

### Console Report

```
================================================================================
TRANSLATION EVALUATION REPORT
================================================================================

AUTOMATIC METRICS
--------------------------------------------------------------------------------
BLEU:  BLEU = 0.2847, precisions = [0.6000, 0.4000, 0.0000, 0.0000], BP = 1.0000, ratio = 1.0000
chrF:  chrF = 0.4230, precision = 0.4530, recall = 0.4015

LINGUISTIC ANALYSIS
--------------------------------------------------------------------------------
Translation Quality Summary (EN → DE)
--------------------------------------

Overall Quality Distribution (n=5 sentences):
  Excellent:   0 (  0.0%)
  Good:        2 ( 40.0%)
  Fair:        2 ( 40.0%)
  Poor:        1 ( 20.0%)

Error Statistics:
  Total errors: 8
  Avg errors per sentence: 1.60

  Error type distribution:
    lexical_mismatch:         2 ( 25.0%)
    word_order_shift:         3 ( 37.5%)
    morphological_loss:       2 ( 25.0%)
    semantic_drift:           1 ( 12.5%)

Key Linguistic Patterns:

  Word Order Patterns:
    • Word order reordering (3x, confidence: 0.60)

  Morphological Issues (German/Russian case, gender, number):
    • Feature loss: ... (2x, confidence: 0.40)

RECOMMENDATIONS FOR IMPROVEMENT
--------------------------------------------------------------------------------
1. Significant word order reordering detected (3x). This may reflect language
   typology (SVO→SOV). Verify this is linguistically appropriate for the
   language pair.
```

### JSON Export

```json
{
  "metrics": {
    "bleu": {
      "score": 0.2847,
      "precisions": [0.6, 0.4, 0.0, 0.0],
      "bp": 1.0,
      "ratio": 1.0
    },
    "chrf": {
      "score": 0.423,
      "precision": 0.453,
      "recall": 0.402
    }
  },
  "linguistic_analysis": {
    "metadata": {
      "source_lang": "en",
      "target_lang": "de",
      "sentence_count": 5
    },
    "error_summary": {
      "total_errors": 8,
      "avg_errors_per_sentence": 1.6,
      "distribution": {
        "lexical_mismatch": 2,
        "word_order_shift": 3,
        "morphological_loss": 2,
        "semantic_drift": 1
      }
    },
    "linguistic_patterns": {
      "morphological": [...],
      "compounds": [...],
      "agreement": [...],
      "word_order": [...],
      "semantic": [...]
    },
    "summary": {
      "linguistic_summary": "...",
      "recommendations": [...]
    }
  }
}
```


## Key Design Decisions

```
|          Decision         |                            Rationale                            |
|---------------------------|-----------------------------------------------------------------|
|      **BLEU + chrF**      | BLEU for standard benchmarking; chrF for morphological richness |
|   **Pattern extraction**  |      Identify systematic issues, not just aggregate scores      |
| **Linguistic categories** |    Rooted in computational linguistics, not arbitrary metrics   |
|    **Heuristic-based**    |     Explainable error detection vs. black-box ML classifiers    |
|  **Multiple references**  |      BLEU can use N references; linguistic analysis uses 1      |
|  **Optional linguistic**  |          Fast metrics-only mode for quick benchmarking          |
```

## Integration with Other Modules

### With TranslationErrorAnalyzer

```python
from analysis.error_analyzer import TranslationErrorAnalyzer
from evaluation.linguistic_report import LinguisticReportGenerator

analyzer = TranslationErrorAnalyzer()
batch_analysis = analyzer.analyze_batch(sources, hypotheses, source_lang, target_lang)

report_gen = LinguisticReportGenerator((source_lang, target_lang))
linguistic_report = report_gen.generate(batch_analysis)
```

### With CompoundProcessor

Compound errors are automatically detected and included in the linguistic report.

```python
# If compound analysis is active, compound_patterns will be populated
report = linguistic_report
for pattern in report.compound_patterns:
    print(f"Compound issue: {pattern.pattern}")
```


## Computational Linguistics Perspective

This evaluation framework reflects linguistic thinking:

1. **Morphological Awareness**: Special attention to case/gender/number (German, Russian)
2. **Compound Handling**: Dedicated analysis for German compounds
3. **Word Order Typology**: Recognizes SVO/SOV differences
4. **Agreement Analysis**: Subject-verb and noun-adjective agreement
5. **Semantic Preservation**: Negation, meaning drift detection

**Not just metrics** — explainable error taxonomy grounded in linguistic theory.


## Running Evaluation Examples

```bash
# Full evaluation examples
python evaluation_examples.py

# Specific example
# (Modify evaluation_examples.py to run individual examples)
```


## API Reference

### BLEUMetric

```python
metric = BLEUMetric(max_n=4, smooth=False)
score = metric.score(hypothesis, references)          # Single
score = metric.corpus_score(hypotheses, references_list)  # Corpus
```

### ChrFMetric

```python
metric = ChrFMetric(order=6, beta=3.0)
score = metric.score(hypothesis, reference)           # Single
score = metric.corpus_score(hypotheses, references)   # Corpus
```

### EvaluationPipeline

```python
pipeline = EvaluationPipeline(
    bleu_max_n=4,
    include_chrf=True,
    include_linguistic=True
)
result = pipeline.evaluate(hypotheses, references_list, ...)
pipeline.print_report(result)
```

### SentenceEvaluator

```python
evaluator = SentenceEvaluator()
result = evaluator.evaluate_sentence(source, hypothesis, references, ...)
```


## Limitations

- **Tokenization**: Space-based (no subword)
- **Alignment**: Positional (not learned)
- **BLEU smoothing**: Simple add-one (not corpus-level smoothing)
- **Linguistic analysis**: Heuristic-based (not ML-based)

**By design**: Interpretable, lightweight, suitable for research.


## Future Extensions

- [ ] Sentence-level BLEU variance (can a model achieve good sentences consistently?)
- [ ] BERTScore / embedding-based semantic similarity
- [ ] POS-aware morphological analysis (using spaCy)
- [ ] Dependency-based word order analysis
- [ ] Domain-specific error categories