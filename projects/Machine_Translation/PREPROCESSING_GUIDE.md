# Optional Morphological Preprocessing Module

## Overview

An **optional, experimental** preprocessing layer that decomposes German compounds before translation.

**Design Philosophy**: 
- Modular and decoupled from core translation
- Turn on/off via config flag
- Graceful degradation if decomposer unavailable
- Non-destructive (original text always accessible)
- Dependency injection (no tight coupling)

**Question Being Tested**: Does morphological decomposition of compounds improve German→English translation quality?


## Architecture

### Components

1. **MorphologicalPreprocessor** (Abstract)
   - Base class for morphological preprocessing steps
   - Language-agnostic interface

2. **GermanCompoundPreprocessor**
   - Specific implementation for German compounds
   - Takes optional external decomposer
   - Marks compound parts with `|` separator

3. **PreprocessorFactory**
   - Creates appropriate preprocessor by language
   - Handles feature flags
   - Dependency injection

4. **PreprocessingConfig**
   - Flag: `enable_morphological` (on/off)
   - Decomposer module specification
   - Preserve original text setting

5. **PreprocessedText**
   - Result object with:
     - Original text
     - Preprocessed text
     - Decompositions (token → components)
     - Metadata

### Integration Points

```
Input Text
    ↓
[MorphologicalPreprocessor] (German compounds)
    ↓
[Standard Language Preprocessor] (normalization, case)
    ↓
[Tokenizer]
    ↓
[Translation Model]
    ↓
[Postprocessor]
    ↓
Output Translation + Metadata
```


## Configuration

### Enable/Disable via Config

```python
from config import TranslationConfig, PreprocessingConfig

# Disabled (default)
config = TranslationConfig.default()
config.preprocessing = PreprocessingConfig(enable_morphological=False)

# Enabled
config = TranslationConfig.default()
config.preprocessing = PreprocessingConfig(enable_morphological=True)
```

### Pipeline Setup

```python
from pipeline.translator_pipeline import TranslatorPipeline
from preprocessing.morphological_preprocessing import GermanCompoundPreprocessor

# With decomposer
decomposer = your_decomposer  # GermanCompoundDecomposer instance
pipeline = TranslatorPipeline(
    config=config,
    decomposer=decomposer
)

# Without decomposer (graceful degradation)
pipeline = TranslatorPipeline(config=config)  # Works fine, just no decomposition
```


## Usage

### Basic Translation

```python
pipeline = TranslatorPipeline(config=config, decomposer=decomposer)

result = pipeline.translate(
    "Das Schmetterlingshaus ist schön.",
    source_lang="de",
    target_lang="en"
)

print(result.text)  # "The butterfly house is beautiful."

# Check if preprocessing was applied
if "decompositions" in result.metadata:
    print(result.metadata["decompositions"])
    # {'schmetterlingshaus': ['schmetterling', 'haus']}
```

### Direct Preprocessor Usage

```python
from preprocessing.morphological_preprocessing import GermanCompoundPreprocessor

preprocessor = GermanCompoundPreprocessor(
    decomposer=your_decomposer,
    enable=True
)

result = preprocessor.preprocess("Das Schmetterlingshaus ist schön.")

print(result.original)      # Original text
print(result.preprocessed)  # "Das Schmetterling|Haus ist schön."
print(result.decompositions)  # {'schmetterlingshaus': ['schmetterling', 'haus']}
```

### Batch Processing

```python
results = pipeline.translate_batch(
    texts=["Das Schmetterlingshaus...", "Der Blauwal..."],
    source_lang="de",
    target_lang="en"
)

for result in results:
    if "decompositions" in result.metadata:
        print(f"Compounds: {result.metadata['decompositions']}")
```


## Decomposer Integration

### Interface Requirement

The decomposer must implement:

```python
class YourDecomposer:
    def decompose(self, word: str) -> List[str]:
        """Decompose a German word into morphemes."""
        return ["component1", "component2", ...]
```

### Injection Points

```python
# At pipeline creation
pipeline = TranslatorPipeline(config=config, decomposer=your_decomposer)

# At runtime (dynamic replacement)
pipeline.set_decomposer(new_decomposer)

# To preprocessor directly
preprocessor = GermanCompoundPreprocessor(decomposer=your_decomposer, enable=True)
```

### Graceful Degradation

If decomposer unavailable:
```python
pipeline = TranslatorPipeline(config=config)  # decomposer=None

# Works fine - just operates without decomposition
result = pipeline.translate(text, "de", "en")
# Preprocessing passes through unchanged
```


## Experimental Features

### Comparison Experiment

Compare translation quality with and without preprocessing:

```python
from experiments.decomposition_experiment import MorphologicalPreprocessingExperiment

experiment = MorphologicalPreprocessingExperiment(decomposer=your_decomposer)

# Single sentence
comparison = experiment.compare_single(
    source="Das Schmetterlingshaus ist groß.",
    reference="The butterfly house is big.",
    source_lang="de",
    target_lang="en"
)

print(f"Raw BLEU: {comparison.bleu_raw:.4f}")
print(f"Preprocessed BLEU: {comparison.bleu_preprocessed:.4f}")
print(f"Improvement: {comparison.improvement:+.2f}%")

# Batch comparison
comparisons = experiment.compare_batch(sources, references, "de", "en")
experiment.print_comparison_report(comparisons)
experiment.export_results(comparisons, "results.json")
```

### Output Format

```
================================================================================
MORPHOLOGICAL PREPROCESSING EXPERIMENT REPORT
================================================================================

Sentences analyzed: 3

AGGREGATE RESULTS
--------------------------------------------------------------------------------
Avg BLEU (Raw):         0.3421
Avg BLEU (Preprocessed): 0.3598
Avg Improvement:        +5.17%

IMPACT DISTRIBUTION
--------------------------------------------------------------------------------
Improved: 2 sentences (66.7%)
Same:     0 sentences ( 0.0%)
Worse:    1 sentences (33.3%)

DETAILED RESULTS
--------------------------------------------------------------------------------
[1] Source: Das Schmetterlingshaus ist sehr groß.
    Raw BLEU:        0.3421 | Preprocessed BLEU: 0.4521 | Change: +32.15%
    Raw Translation:         The butterfly house is very large.
    Preprocessed Translation: The butterfly house is large.
    Decompositions: schmetterlingshaus→['schmetterling', 'haus']
```


## Metadata Tracking

When preprocessing is applied, metadata includes:

```python
result.metadata = {
    "morphological_preprocessing": True,
    "decomposer_available": True,
    "raw_input_text": "Das Schmetterlingshaus...",
    "decomposed_input_text": "Das Schmetterling|Haus...",
    "decompositions": {
        "schmetterlingshaus": ["schmetterling", "haus"],
        ...
    },
    "decomposition_count": 1,
}
```


## Design Decisions

```
|          Decision          |                     Rationale                      |
|----------------------------|----------------------------------------------------|
| **Optional/Config-Driven** | No forcing - experimental feature, can be disabled |
|  **Dependency Injection**  |   Decomposer passed in, not hard-coded; testable   |
|  **Graceful Degradation**  |     Works without decomposer (passthrough mode)    |
|     **Non-Destructive**    |     Original text always available in metadata     |
|         **Modular**        |     Separate preprocessor classes per language     |
|      **Marker-Based**      |   Use `\|` to mark decomposed parts (reversible)   |
|      **Metadata Rich**     |      Track what preprocessing happened and how     |
```

## Extensibility

### Add New Language Preprocessing

```python
from preprocessing.morphological_preprocessing import MorphologicalPreprocessor, PreprocessorFactory

class RussianMorphPreprocessor(MorphologicalPreprocessor):
    def preprocess(self, text: str) -> PreprocessedText:
        # Russian-specific decomposition logic
        pass

# Register in factory
PreprocessorFactory.create(
    language="ru",
    decomposer=russian_decomposer,
    enable_morphological=True
)
```

### Custom Compound Markers

```python
preprocessor = GermanCompoundPreprocessor(
    decomposer=decomposer,
    enable=True
)
preprocessor.compound_marker = "+"  # Change from "|" to "+"
# Result: "Das Schmetterling+Haus..."
```


## Performance Considerations

- **Memory**: Minimal overhead (stores decompositions in dict)
- **Speed**: Preprocessing is fast (just string manipulation + external decomposer call)
- **Model Impact**: Input length may increase slightly (token count higher with separated compounds)


## Testing & Validation

Run example experiments:

```bash
python preprocessing_examples.py
```

Key test scenarios:
1. ✓ Preprocessing disabled (baseline)
2. ✓ Preprocessing enabled, no decomposer (graceful degradation)
3. ✓ Preprocessing enabled with mock decomposer
4. ✓ Direct preprocessor usage
5. ✓ Config-driven toggle
6. ✓ Comparison experiment
7. ✓ Dynamic decomposer injection


## Integration with Error Analysis

Compound decomposition errors are automatically detected:

```python
from analysis.compound_processor import CompoundProcessor

processor = CompoundProcessor(decomposer=your_decomposer)
error = processor.create_error_span(source_de, target_de)

# Compound errors flagged separately from other morphological errors
```


## Limitations & Future Work

**Current Limitations**:
- Space-based tokenization (no subword)
- Decomposition applied at word level only
- Single marker style (configurable but not context-aware)

**Future Extensions**:
- Subword-level decomposition
- Context-aware merging of results
- Bidirectional (recomposition) for generation tasks
- Multi-language support (Russian aspect, etc.)


## Quick Reference

```
|          Feature          |                       Command                       |
|---------------------------|-----------------------------------------------------|
|  **Enable preprocessing** |   `PreprocessingConfig(enable_morphological=True)`  |
| **Disable preprocessing** |  `PreprocessingConfig(enable_morphological=False)`  |
|   **Inject decomposer**   | `TranslatorPipeline(config=config, decomposer=dec)` |
|   **Replace decomposer**  |          `pipeline.set_decomposer(new_dec)`         |
|      **Direct usage**     |    `GermanCompoundPreprocessor(dec, enable=True)`   |
| **Comparison experiment** |     `MorphologicalPreprocessingExperiment(dec)`     |
|    **Check if applied**   |      `if "decompositions" in result.metadata:`      |
|  **View decompositions**  |         `result.metadata["decompositions"]`         |
```

## Philosophy

This module demonstrates **experimental research integration** into a production translation pipeline:
- ✓ Easily turned on/off
- ✓ Doesn't break existing code
- ✓ Measurable impact (via BLEU comparison)
- ✓ Transparent (metadata shows what happened)
- ✓ Extensible (new decomposers, languages)

**Not a hack** — a well-designed optional feature for testing morphological hypotheses.