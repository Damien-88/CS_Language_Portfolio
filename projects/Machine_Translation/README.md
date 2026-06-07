# Machine Translation: Cross-Lingual Semantic Alignment

A **computational linguistics research system** for evaluating machine translation through linguistic analysis, with focus on morphosyntactic divergence and cross-lingual semantic preservation.

**Problem**: Transformer-based MT systems optimize for surface-level metrics (BLEU) but fail to capture linguistic phenomena critical to meaning preservation. This project integrates **structured linguistic error analysis** into the evaluation pipeline to understand *how* and *why* translations diverge semantically.

**Focus**: English ↔ German ↔ Russian, examining:
- Morphological feature preservation (case, gender, number)
- Compound decomposition and alignment
- Syntactic reordering across typologically different languages
- Semantic drift detection (negation, modality, aspect)


## Motivation: Linguistic Challenges in Neural MT

### The Problem Space

Modern neural machine translation achieves strong BLEU scores but often fails on linguistically-motivated phenomena:

1. **Morphological Complexity**: German nouns carry case/gender/number information; Russian verbs encode perfective/imperfective aspect. These features are frequently lost in translation, especially to morphologically-poorer languages like English.

2. **Compound Decomposition**: German compounds (e.g., *Schmetterlingshaus* = butterfly house) often become ambiguous or mis-segmented in translation. Pre-decomposition may improve alignment.

3. **Syntactic Divergence**: English is strongly SVO; German is SVO main/SOV subordinate; Russian is relatively flexible but aspect-driven. Word order shifts that preserve meaning are linguistically appropriate but appear as errors to n-gram metrics.

4. **Semantic Drift**: Translation failures often involve meaning inversion (negation flip), aspect changes (imperfective → perfective), or modality shifts (deontic → epistemic). These are invisible to BLEU.

### Research Questions

- **Q1**: Can linguistic error categorization explain translation failures better than aggregate metrics?
- **Q2**: Does morphological decomposition of German compounds improve translation quality?
- **Q3**: What are the systematic differences in error patterns across language pairs (EN→DE vs. EN→RU)?
- **Q4**: Can linguistic features (morphological agreement, word order) be used as evaluation signals?


## Approach: Transformer-Based Pipeline with Linguistic Analysis

### Architecture

```
Input: Source Text (EN/DE/RU)
   ↓
[Optional Morphological Preprocessing]
   • German compound decomposition (experimental)
   • Language-specific normalization
   ↓
[Transformer Model (Opus-MT)]
   • HuggingFace pre-trained seq2seq
   • 6 language pair coverage
   ↓
[Language-Specific Postprocessing]
   • Case restoration (German)
   • Punctuation fixing
   ↓
[Dual Evaluation Path]
   ├─ Automatic Metrics: BLEU, chrF
   └─ Linguistic Analysis: Error taxonomy + pattern extraction
```

### Core System Components

**TranslationModel** (`models/translator.py`)
- HuggingFace Transformers wrapper (Opus-MT)
- Language-aware pre/postprocessing
- Optional morphological preprocessing (injected, not hardcoded)
- Structured metadata output

**TranslatorPipeline** (`pipeline/translator_pipeline.py`)
- Multi-pair routing (6 language combinations)
- Model caching and lifecycle management
- Batch processing with GPU optimization
- Decomposer dependency injection for experiments

**Error Analysis** (`analysis/error_analyzer.py`)
- 9-category linguistic taxonomy:
  - Lexical mismatch (word choice)
  - Morphological loss (case, gender, number)
  - Agreement errors (subject-verb, noun-adjective)
  - Compound breakdown (German-specific)
  - Word order shift (syntactic reordering)
  - Semantic drift (meaning changes)
  - OOV/untranslated (coverage issues)
  - Voice/aspect loss (grammatical transformation)
  - Pronoun errors (coreference)

**Evaluation Framework** (`evaluation/`)
- BLEU: N-gram precision + brevity penalty
- chrF: Character-level F-score (morphology-sensitive)
- Linguistic Report: Pattern extraction + actionable recommendations


## Linguistic Motivations by Language Pair

### EN → DE (and DE → EN)

**Key Challenges**:
- **Morphological richness**: German 4-case system (nominative, accusative, dative, genitive) with gender (m/f/n). English has minimal inflection.
- **Compounding**: German productively forms compounds; English uses separate words. E.g., *Schmetterlingshaus* vs. "butterfly house" → potential ambiguity.
- **Word order**: German main clauses are SVO but subordinate clauses are SOV. English is consistently SVO. Structure changes that preserve meaning may appear as errors.
- **Article system**: German articles carry case/gender; English articles are simpler (*the*). Agreement violations are common.

**Linguistic Analysis Focus**:
```
Morphological Loss: German adjective agrees with noun in case/gender/number
  "der große Mann" (nom m sg) → "the big man" (no case marking)
  
Compound Breakdown: Long German nouns may be mis-tokenized
  "Schmetterling|Haus" (marked) vs. "Schmetterlings haus" (error)
  
Agreement Errors: Determiner-adjective-noun agreement
  "großen Haus" (wrong gender) vs. "großes Haus" (correct neuter)
```

### EN → RU (and RU → EN)

**Key Challenges**:
- **Case system**: Russian has 6 cases + 3 genders + 2 numbers, creating rich morphology. English noun phrases are impoverished by comparison.
- **Aspect**: Russian distinguishes perfective (completed action) from imperfective (ongoing/habitual). English relies on tense + context.
- **Verbal particles**: Prefixes modify meaning (e.g., *читать* "to read" vs. *прочитать* "to finish reading").
- **Word order flexibility**: Russian allows relatively free word order; English is rigid SVO. Correct Russian translations may have non-canonical order.

**Linguistic Analysis Focus**:
```
Morphological Loss: Russian case marking carries grammatical relations
  "красивую кошку" (acc sg f) → "beautiful cat" (no case)
  
Agreement Errors: Adjective-noun, subject-verb
  "большой дом" (m) vs. "большая дом" (f, wrong)
  
Aspect Loss: Perfectivity not recoverable from English
  EN: "I read the book" → RU: "Я читал/прочитал книгу?" (aspect ambiguous in English)
```

### DE → RU (and RU → DE)

**Key Challenges**:
- **Bridging morphologically-rich languages**: Both have case systems but different structures (German: 4 cases, Russian: 6).
- **Compounding**: German compounds must be handled before Russian translation; Russian rarely compounds.
- **Aspect-tense interaction**: Russian aspect is primary; German tense is primary. Neither maps cleanly.


## Evaluation Strategy

### Beyond BLEU: Structured Linguistic Analysis

Standard metrics (BLEU, METEOR) are coarse: they aggregate over all errors. This system separates concerns:

```
┌─────────────────────────┐
│  Translation Result     │
└────────────┬────────────┘
             │
      ┌──────┴──────┐
      ↓             ↓
  [METRICS]    [LINGUISTIC ANALYSIS]
  
  • BLEU          • Morphological loss: 12 instances
  • chrF          • Agreement errors: 3 instances
  • Corpus-level  • Compound breakdown: 2 instances
                  • Word order shifts: 8 instances
                  • Semantic drift: 1 instance
                  
                  → Actionable insights
```

### Evaluation Pipeline

1. **Sentence-Level Analysis**: Each translation receives error categorization + quality rating (excellent/good/fair/poor)

2. **Pattern Extraction**: Aggregate errors into recurring patterns
   - Most common errors per language pair
   - Error distribution by type
   - Examples per pattern

3. **Linguistic Report**: Human-readable summary with recommendations
   - Quality distribution (% excellent/good/fair/poor)
   - Top error patterns with confidence scores
   - Actionable recommendations for improvement

### Example Report Output

```
Translation Quality Summary (EN → DE)
────────────────────────────────────

Overall Quality Distribution (n=50 sentences):
  Excellent: 15 (30.0%)
  Good:      20 (40.0%)
  Fair:      12 (24.0%)
  Poor:       3 (6.0%)

Key Linguistic Patterns:

Morphological Issues (German case, gender, number):
  • Feature loss: 12x → Suggest morphological constraints in decoder
  
Compound Handling Issues (German-specific):
  • Decomposition errors: 5x → Integrate compound decomposer
  
Agreement Errors:
  • Gender/case mismatch: 3x → Add agreement scoring

Recommendations:
1. Fine-tune on morphologically-marked data
2. Test compound decomposition as preprocessing
3. Add agreement constraints to beam search
```


## Optional Morphological Preprocessing

**Experimental feature** to test hypothesis: "Does decomposing German compounds before translation improve quality?"

```python
from config import TranslationConfig, PreprocessingConfig
from pipeline.translator_pipeline import TranslatorPipeline

# Enable via config (off by default)
config = TranslationConfig.default()
config.preprocessing = PreprocessingConfig(enable_morphological=True)

pipeline = TranslatorPipeline(config=config, decomposer=your_decomposer)

# Automatic comparison
result_with = pipeline.translate("Das Schmetterlingshaus...", "de", "en")
# Input fed to model: "Das Schmetterling Haus..."
```

**Design**: Modular, optional, measurable
- Can be enabled/disabled via config
- Decomposer injected (not hardcoded)
- Results tracked in metadata
- Built-in BLEU comparison experiment


## System Design: Modularity & Extensibility

### Dependency Injection (Not Tight Coupling)

```python
# Decomposer injected at initialization
decomposer = YourGermanCompoundDecomposer()
pipeline = TranslatorPipeline(config=config, decomposer=decomposer)

# Can be replaced dynamically
pipeline.set_decomposer(new_decomposer)
```

This allows:
- Testing multiple decomposers without code changes
- Easy integration when your decomposer is ready
- Clean separation between MT and morphology modules

### Language-Specific Processors

```
preprocessing/language_utils.py
  ├─ EnglishPreprocessor / EnglishPostprocessor
  ├─ GermanPreprocessor / GermanPostprocessor  (capitalization, compounds)
  └─ RussianPreprocessor / RussianPostprocessor (case handling, stress)
```

Add new language: implement `Preprocessor` + `Postprocessor` + register in factory. MT pipeline handles automatically.

### Config-Driven

No magic constants. All model selections, hyperparameters, and feature flags in config:

```python
@dataclass
class TranslationConfig:
    language_pairs: Dict[tuple[str, str], LanguagePairConfig]  # Which models
    device: str = "cuda"                                        # Hardware
    num_beams: int = 5                                          # Decoding
    preprocessing: PreprocessingConfig                          # Features
```


## Supported Language Pairs

| Pair | Model | Tested |
|------|-------|--------|
| EN ↔ DE | Helsinki-NLP/Opus-MT-en-de | ✓ |
| EN ↔ RU | Helsinki-NLP/Opus-MT-en-ru | ✓ |
| DE ↔ RU | Helsinki-NLP/Opus-MT-de-ru | ✓ |


## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Translation

```python
from api import translate

result = translate("The butterfly house is beautiful.", "en", "de")
print(result.text)  # German translation
print(result.metadata)  # Model, tokens, preprocessing info
```

### Linguistic Error Analysis

```python
from analysis.error_analyzer import TranslationErrorAnalyzer

analyzer = TranslationErrorAnalyzer()

analysis = analyzer.analyze(
    source="The system will NOT increase prices.",
    target="Das System wird Preise erhöhen.",  # Missing negation!
    source_lang="en",
    target_lang="de"
)

print(analysis.overall_quality)  # "poor"
for error in analysis.errors:
    print(f"{error.category}: {error.explanation}")
    # Output: semantic_drift: Negation lost in translation
```

### Full Evaluation (Metrics + Linguistic)

```python
from evaluation.evaluation_pipeline import EvaluationPipeline

pipeline = EvaluationPipeline(
    include_chrf=True,
    include_linguistic=True
)

result = pipeline.evaluate(
    hypotheses=translations,
    references_list=references,
    sources=source_texts,
    source_lang="en",
    target_lang="de"
)

pipeline.print_report(result)
```

### Experiment: Morphological Preprocessing

```python
from experiments.decomposition_experiment import MorphologicalPreprocessingExperiment

experiment = MorphologicalPreprocessingExperiment(decomposer=your_decomposer)
comparisons = experiment.compare_batch(sources, references, "de", "en")
experiment.print_comparison_report(comparisons)
# Output: BLEU with/without preprocessing, % improvement
```


## Project Structure

```
Machine_Translation/
├── config.py                      # Configuration (models, preprocessing)
├── api.py                         # Simple translate() interface
├── models/
│   ├── base.py                   # Abstract base classes
│   └── translator.py             # HuggingFace wrapper + preprocessing
├── preprocessing/
│   ├── language_utils.py         # Language-specific pre/postprocessors
│   └── morphological_preprocessing.py  # Compound decomposition (optional)
├── pipeline/
│   └── translator_pipeline.py    # Multi-pair orchestration
├── analysis/
│   ├── error_types.py            # Error categories + data structures
│   ├── error_analyzer.py         # Linguistic error detection
│   └── compound_processor.py     # Compound analysis integration point
├── evaluation/
│   ├── metrics.py                # BLEU, chrF computation
│   ├── linguistic_report.py      # Pattern extraction + reporting
│   └── evaluation_pipeline.py    # Full evaluation orchestration
├── experiments/
│   └── decomposition_experiment.py  # Compare with/without preprocessing
├── demos/
│   ├── data_exploration.ipynb                
│   ├── model_selection.ipynb      
│   ├── evaluation_analysis.ipynb               
│   ├── compound_effects.ipynb     
│   └── error_taxonomy.ipynb        
├── examples.py                    # Basic usage examples
├── analysis_examples.py           # Error analysis examples
├── evaluation_examples.py         # Evaluation examples
├── preprocessing_examples.py      # Preprocessing examples
└── [GUIDES]
    ├── ANALYSIS_GUIDE.md         # Linguistic error analysis docs
    ├── EVALUATION_GUIDE.md       # Metrics + reporting docs
    └── PREPROCESSING_GUIDE.md    # Morphological preprocessing docs
```


## Future Work: Integration with Cross-Lingual Semantic Systems

This MT system is designed as a **foundation for cross-lingual intelligence**. Planned integrations:

### 1. Semantic Search Across Languages
- Use translation + linguistic analysis to identify paraphrases and semantic equivalents
- Leverage error analysis to identify *which translations are semantically safe*
- Example: "The system will not increase prices" (EN) → semantically similar to "Das System wird Preise nicht erhöhen" (DE), not "Das System wird Preise erhöhen"

### 2. Knowledge Graph Alignment
- Map semantic relations across languages using translation chains
- Use linguistic error patterns to identify where knowledge representations diverge
- Example: Russian aspect (perfectivity) → English aspect (tense + auxiliaries) → knowledge graph temporal properties

### 3. Cross-Lingual Morphological Analysis
- Extend German compound decomposer to identify compounds in Russian and English
- Build morphological alignment models (German case → Russian case → English position)
- Use for zero-shot morphological transfer

### 4. Linguistic Feature Preservation Metrics
- Move beyond n-gram metrics to **linguistic precision**: "Did the translation preserve case? aspect? modality?"
- Train reranking models that prefer linguistically-faithful translations
- Example: Use agreement error count as a differentiator when BLEU is tied

### 5. Corpus Linguistics Integration
- Build parallel corpus with linguistic annotations
- Analyze systematic divergences (e.g., "English drops adjectives more often than German")
- Feed findings back into error analysis framework


## Evaluation & Reproducibility

All results are logged with full metadata:
- Model checkpoints and versions
- Random seeds and hyperparameters
- Processing pipeline configuration
- Error analysis reports

**Evaluation outputs**:
```
evaluation_result.json
├── metrics
│   ├── bleu: {score, precisions, bp, ratio}
│   └── chrf: {score, precision, recall}
└── linguistic_analysis
    ├── error_distribution
    ├── quality_distribution
    ├── patterns: [morphological, compound, agreement, word_order, semantic]
    └── recommendations
```


## Dependencies

- **transformers** ≥4.30.0 (HuggingFace Opus-MT)
- **torch** ≥2.0.0 (PyTorch for GPU)
- **pydantic** ≥2.0.0 (Type validation)
- **numpy** ≥1.24.0 (Numerical ops)
- **tqdm** ≥4.65.0 (Progress tracking)


## References & Motivation

**Linguistic Theory**:
- Comrie, B. (1976). *Aspect*. Cambridge University Press. [Russian aspectual system]
- Greenberg, J. (1963). "Some universals of grammar." *Linguistic Universals*. [Word order typology]
- Booij, G. (2010). "Construction morphology." *Language and Linguistics Compass*. [German compounding]

**Machine Translation & Evaluation**:
- Papineni, K., et al. (2002). "BLEU: a method for automatic evaluation of machine translation." [BLEU metric]
- Koehn, P. (2020). *Statistical Machine Translation*. [MT foundations]
- Ma, X., et al. (2019). "Linguistic-Guided Evaluation for Neural Machine Translation." [Linguistic analysis in MT]

**Cross-Lingual NLP**:
- Ruder, S., et al. (2019). "Unsupervised Cross-lingual Representation Learning." [Cross-lingual alignment]


## Contributing

This is a **research-oriented portfolio project**. Contributions focusing on linguistic analysis, new evaluation metrics, or language-specific insights are welcome.

To extend:
1. Add language: implement `Preprocessor` + `Postprocessor`
2. Add error category: extend `ErrorCategory` enum + implement detection method
3. Test hypothesis: use `EvaluationPipeline` + `MorphologicalPreprocessingExperiment`


## License

Part of the **Computational Linguistics Portfolio**: A structured progression from preprocessing → morphology → semantic systems → cross-lingual intelligence.

**Status**: Active research. Results and architecture subject to refinement as linguistic analysis deepens.


## Acknowledgments

- HuggingFace for Opus-MT models
- European Union funding for multilingual language resources
- Computational linguistics research community for linguistic error taxonomies