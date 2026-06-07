# Linguistic Error Analysis Module

## Overview

A **computational linguistics-focused** error analysis layer for machine translation that goes beyond standard metrics (BLEU, METEOR). This module provides explainable, structured diagnostics of translation errors at morphological, syntactic, and semantic levels.

**Design Philosophy**: Rule-light but explainable. Heuristic-based analysis suitable for research and portfolio demonstration.


## Core Components

### 1. Error Categories

Each detected error is classified into one of 9 linguistic categories:

```
|       Category       |                     Description                   |                 Example                |
|----------------------|---------------------------------------------------|----------------------------------------|
|  `LEXICAL_MISMATCH`  |          Wrong word choice or vocabulary          |       "quick" → "langsam" (slow)       |
| `MORPHOLOGICAL_LOSS` |  Lost grammatical features (case, gender, number) |     German noun loses case marking     |
|   `AGREEMENT_ERROR`  |  Subject-verb or adjective-noun agreement failure | "großen Haus" (wrong gender agreement) |
| `COMPOUND_BREAKDOWN` | German compound incorrectly decomposed/translated |   "Schmetterling" → "Schmetter ling"   |
|  `WORD_ORDER_SHIFT`  |          Significant syntactic reordering         |            SVO → SOV changes           |
|   `SEMANTIC_DRIFT`   |            Meaning changed or distorted           |  Negation flip: "not" → (no negation)  |
|  `OOV_UNTRANSLATED`  |      Out-of-vocabulary or untranslated words      |   "rapidement" left in French output   |
|  `VOICE_ASPECT_LOSS` |       Active/passive voice or aspect changed      |       Imperfect → perfect aspect       |
|    `PRONOUN_ERROR`   |       Incorrect or missing pronoun agreement      |    Case mismatch in Russian pronouns   |
```

### 2. Error Span Structure

Each error is represented as an `ErrorSpan`:

```python
@dataclass
class ErrorSpan:
    category: ErrorCategory          # Error type
    source_span: str                 # Problem in source
    target_span: str                 # Problem in translation
    reference_span: Optional[str]    # Reference for comparison
    severity: str                    # "low", "medium", "high"
    explanation: str                 # Human-readable explanation
    confidence: float                # 0.0-1.0 (analyst confidence)
```

### 3. Analysis Results

Each sentence analysis produces a `SentenceLinguisticAnalysis`:

```python
@dataclass
class SentenceLinguisticAnalysis:
    source_text: str
    target_text: str
    reference_text: Optional[str]
    errors: List[ErrorSpan]          # All detected errors
    error_categories: dict           # Error type counts
    overall_quality: str             # "excellent", "good", "fair", "poor"
    linguistic_notes: str            # Additional observations
```


## Usage Examples

### Single Sentence Analysis

```python
from analysis.error_analyzer import TranslationErrorAnalyzer

analyzer = TranslationErrorAnalyzer()

analysis = analyzer.analyze(
    source="The company will not increase prices.",
    target="Die Firma wird Preise erhöhen.",  # Missing negation
    source_lang="en",
    target_lang="de"
)

print(analysis.overall_quality)  # "fair" or "poor"
for error in analysis.errors:
    print(f"{error.category.value}: {error.explanation}")
```

### Batch Analysis

```python
sources = ["Hello", "Good morning", "Never seen this before"]
targets = ["Hallo", "Guten Morgen", "Nie das gesehen bevor"]

batch = analyzer.analyze_batch(
    sources=sources,
    targets=targets,
    source_lang="en",
    target_lang="de"
)

# Get error distribution across all sentences
dist = batch.error_distribution()
# Output: {"lexical_mismatch": 2, "word_order_shift": 1, ...}
```

### JSON Output

```python
analysis = analyzer.analyze(source, target, source_lang, target_lang)

# Structured output for downstream processing
json_str = analysis.to_json()
# Includes: errors, categories, quality rating, explanations
```


## Sub-Analyzers

### LexicalAnalyzer

Detects vocabulary and lexical errors:
- **OOV/Untranslated Words**: Words that appear unchanged in translation
- **Negation Loss**: Critical negation words ("not", "kein", "не") missing

```python
analyzer.lexical.detect_oov_untranslated(source_tokens, target_tokens)
analyzer.lexical.detect_negation_loss(source, target)
```

### MorphologicalComparator

Analyzes grammatical features across languages:
- **Case/Gender/Number Loss**: German/Russian features not preserved in target
- **Agreement Errors**: Subject-verb or adjective-noun mismatches

Language-specific analyzers:
- `GermanMorphologyAnalyzer`: Case detection, capitalization, articles
- `RussianMorphologyAnalyzer`: Case endings, aspect marking
- `EnglishMorphologyAnalyzer`: Simpler; detects determiners, verbs

```python
comparator = MorphologicalComparator()
losses = comparator.detect_morphological_loss(
    source_tokens, target_tokens, "de", "en"
)
errors = comparator.detect_agreement_errors(target_tokens, "de")
```

### WordOrderAnalyzer

Detects syntactic reordering:
- **Word Order Similarity**: Measures token order preservation (0-1 scale)
- **SVO/SOV Changes**: Structural reordering

```python
similarity = analyzer.word_order.compute_word_order_similarity(
    source_tokens, target_tokens
)
# Low similarity (<0.6) → significant reordering
```

### SemanticAnalyzer

Detects meaning changes:
- **Semantic Similarity**: N-gram overlap between source and target
- **Negation Flip**: Meaning inverted via negation change

```python
similarity = analyzer.semantic.compute_semantic_similarity(source, target)
flipped = analyzer.semantic.detect_negation_flip(source, target)
```

### GermanCompoundAnalyzer

Detects compound-related issues (stub; integrates with decomposer):

```python
breakdowns = analyzer.compound.detect_compound_breakdown(
    source_tokens, target_tokens
)
# Returns: [(compound, target_expansion), ...]
```


## Integration with GermanCompoundDecomposer

The `CompoundProcessor` is the integration point for your German compound decomposer:

### Without Decomposer (Stub Mode)

```python
from analysis.compound_processor import CompoundProcessor

processor = CompoundProcessor(decomposer=None)

# Works with limited functionality
analysis = processor.analyze_compound_preservation(
    source_de="Das Schmetterlingshaus",
    target_de="Das Schmetterling Haus"
)
```

### With Your Decomposer

```python
from your_module import GermanCompoundDecomposer
from analysis.compound_processor import CompoundProcessor

decomposer = GermanCompoundDecomposer()
processor = CompoundProcessor(decomposer=decomposer)

# Full compound analysis
analysis = processor.analyze_compound_preservation(source_de, target_de)
# Returns: {
#     "source_compound": "Schmetterlingshaus",
#     "source_decomposition": ["Schmetterling", "Haus"],
#     "target_compounds": [...],
#     "alignment_score": 0.85,
#     "compound_preserved": True
# }

# Create ErrorSpan if issues detected
error = processor.create_error_span(source_de, target_de, reference_de)
```


## Severity Levels

Each error has a severity rating:

- **HIGH**: Critical errors affecting meaning (negation loss, semantic flip)
- **MEDIUM**: Grammatical issues but meaning mostly preserved (agreement errors)
- **LOW**: Minor issues or word order changes that don't affect understandability


## Confidence Scores

Confidence (0.0-1.0) indicates analyzer confidence in its error detection:

- **0.9-1.0**: High confidence (obvious pattern)
- **0.7-0.9**: Medium confidence (heuristic-based)
- **0.5-0.7**: Low confidence (ambiguous or rule-light)


## Quality Ratings

Overall sentence translation quality is rated as:

| Rating | Criteria |
|--------|----------|
| **Excellent** | No errors detected |
| **Good** | 1 medium-severity error |
| **Fair** | 1+ high-severity errors OR 3+ medium errors |
| **Poor** | 2+ high-severity errors |


## Computational Linguistics Perspective

This analysis reflects linguistics thinking:

1. **Morphological Awareness**: Special treatment for German/Russian grammatical features
2. **Word Order Typology**: Recognizes SVO/SOV differences
3. **Semantic Compositionality**: Analyzes how meaning is preserved
4. **Lexical Semantics**: Tracks negation and semantic drift
5. **Compound Decomposition**: Integrates morphological analysis (your decomposer)

**Not just engineering metrics** — explainable error taxonomy grounded in linguistic theory.


## JSON Output Format

```json
{
  "source": "The software is working.",
  "target": "Die Software funktioniert.",
  "source_lang": "en",
  "target_lang": "de",
  "errors": [
    {
      "category": "morphological_loss",
      "source_span": "software",
      "target_span": "Software",
      "severity": "low",
      "explanation": "Gender feature not marked in target",
      "confidence": 0.65
    }
  ],
  "error_summary": {
    "morphological_loss": 1
  },
  "overall_quality": "good",
  "linguistic_notes": ""
}
```


## Batch Analysis Output

```json
{
  "source_lang": "en",
  "target_lang": "de",
  "sentence_count": 50,
  "error_distribution": {
    "morphological_loss": 12,
    "word_order_shift": 5,
    "semantic_drift": 3,
    "lexical_mismatch": 8
  },
  "quality_distribution": {
    "excellent": 15,
    "good": 20,
    "fair": 10,
    "poor": 5
  },
  "analyses": [...]
}
```


## Design Notes

### Rule-Light Philosophy

The analyzer uses **heuristics** rather than heavy NLP models:

- Simple tokenization (space-based)
- Morphological feature inference from word patterns
- N-gram semantic similarity (not embedding-based)
- Negation detection via keyword matching

This keeps it **fast**, **explainable**, and **lightweight** — suitable for portfolio work.

### Extensibility

To add new error detection:

```python
def detect_your_error(self, source_tokens, target_tokens):
    # Implement your logic
    return errors

# Add to _detect_* methods in TranslationErrorAnalyzer
```

### Language Support

Currently handles: **English, German, Russian**

Adding a new language:
1. Extend `MorphologyAnalyzer` base class
2. Add to `ANALYZER_MAP`
3. Implement language-specific rules


## Limitations

- **Tokenization**: Space-based (no subword handling)
- **Alignment**: Simple positional (not learned alignment)
- **Morphology**: Pattern-based heuristics (not statistical)
- **Semantic**: N-gram overlap (not semantic embeddings)

**By design**: Interpretable, not maximally accurate. Suitable for research/portfolio demonstration.


## Example Output

See `analysis_examples.py` for:
- Single sentence analysis
- Batch analysis with distributions
- Morphological error detection
- Semantic drift examples
- Word order shift detection
- Compound analysis
- JSON serialization
- CompoundProcessor integration

Run:
```bash
python analysis_examples.py
```