# Machine Translation System Architecture

## Overview

A research-oriented, modular machine translation system supporting English ↔ German ↔ Russian with integrated linguistic error analysis and compound-aware evaluation.

**Design philosophy**: Transformer-based models (HuggingFace Opus-MT) → structured evaluation pipeline → deep linguistic analysis → composition with existing GermanCompoundDecomposer.


## 1. Folder Structure

```
Machine_Translation/
├── README.md                          # Setup, usage, results summary
├── requirements.txt
├── config/
│   ├── models.yaml                   # Model selections & parameters
│   ├── evaluation.yaml               # Metric configurations
│   └── languages.yaml                # Language-specific rules
├── data/
│   ├── raw/                          # Original datasets
│   │   ├── en_de/
│   │   ├── en_ru/
│   │   └── de_ru/
│   ├── processed/                    # Normalized, tokenized corpora
│   ├── splits/                       # Train/val/test assignments
│   └── loaders.py                    # Unified data interface
├── models/
│   ├── translator.py                 # Main translation interface
│   ├── base.py                       # Abstract base classes
│   └── checkpoints/                  # Model weights (gitignored)
├── preprocessing/
│   ├── tokenizer.py                  # Multilingual tokenization
│   ├── detokenizer.py                # Token → text reconstruction
│   └── language_utils.py             # DE/RU/EN-specific rules
├── evaluation/
│   ├── metrics.py                    # BLEU, METEOR, chrF, sentence-level
│   ├── linguistic_analysis.py        # Morphology, syntax, semantics
│   └── compound_analysis.py          # German compound errors
├── analysis/
│   ├── error_analyzer.py             # Error taxonomy & patterns
│   ├── morphology_checker.py         # Morphological validation
│   ├── compound_processor.py         # ← YOUR DECOMPOSER INTEGRATION
│   └── visualizer.py                 # Error heatmaps, comparisons
├── pipeline/
│   ├── translator_pipeline.py        # Input → Output orchestration
│   └── evaluation_pipeline.py        # Evaluation workflow
├── experiments/
│   ├── baseline_translation.py       # Baseline runs
│   ├── compound_aware_translation.py # Compound-enhanced pipeline
│   └── results/
│       ├── metrics.json
│       └── error_reports.json
├── scripts/
│   ├── download_models.py            # Fetch HuggingFace
│   ├── prepare_data.py               # Create splits
│   ├── translate_batch.py            # CLI translation
│   ├── evaluate_system.py            # Full eval suite
│   └── analyze_errors.py             # Error deep-dive
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_selection.ipynb
│   ├── 03_evaluation_analysis.ipynb
│   ├── 04_compound_effects.ipynb
│   └── 05_error_taxonomy.ipynb
└── tests/
    ├── test_models.py
    ├── test_preprocessing.py
    ├── test_evaluation.py
    └── test_pipeline.py
```


## 2. Module Breakdown & Responsibilities

### Core Translation (`models/`)

**translator.py**
```python
class MultillingualTranslator:
    """Unified interface for all EN/DE/RU pairs."""
    
    LANGUAGE_PAIRS = {
        'en→de': 'Helsinki-NLP/Opus-MT-en-de',
        'de→en': 'Helsinki-NLP/Opus-MT-de-en',
        'en→ru': 'Helsinki-NLP/Opus-MT-en-ru',
        'ru→en': 'Helsinki-NLP/Opus-MT-ru-en',
        'de→ru': 'Helsinki-NLP/Opus-MT-de-ru',
        'ru→de': 'Helsinki-NLP/Opus-MT-ru-de',
    }
    
    def translate(self, text: str, src_lang: str, tgt_lang: str) -> Dict
    def translate_batch(self, texts: List[str], src_lang, tgt_lang) -> List[Dict]
    def _get_preprocessor(self, lang: str) -> Preprocessor
    def _get_postprocessor(self, lang: str) -> Postprocessor
```

**base.py** - Abstract interfaces for extensibility
- `Translator` (base class for model wrappers)
- `Preprocessor` (language-aware normalization)
- `Postprocessor` (language-aware detokenization)
- `LanguageConfig` (morphological rules, special tokens)


### Data & Preprocessing (`data/`, `preprocessing/`)

**data/loaders.py**
```python
class ParallelCorpusLoader:
    """Load and split language pairs."""
    
    def load_language_pair(src_lang: str, tgt_lang: str) -> ParallelCorpus
    def create_splits(corpus, train=0.8, val=0.1, test=0.1)
    def get_batch_generator(split: str, batch_size: int)
```

**preprocessing/tokenizer.py**
```python
class MultilingualTokenizer:
    """Language-aware tokenization preserving meaningful units."""
    
    def tokenize(self, text: str, lang: str) -> List[str]
    def handle_compounds(text: str, lang: str)  # Preserve German compounds pre-translation
    def normalize(text: str, lang: str)  # Case, punctuation, diacritics
```

**preprocessing/language_utils.py**
- `GermanRules`: Capitalization, compound boundary markers, umlauts
- `RussianRules`: Case system, aspect marking, stress notation
- `EnglishRules`: Contractions, determiners, word-boundary handling


### Evaluation (`evaluation/`)

**metrics.py** - Standard + custom metrics
```python
class TranslationMetrics:
    def bleu(predictions, references) -> float
    def sentence_bleu(pred, ref) -> float
    def meteor(predictions, references) -> float
    def chrf(predictions, references) -> float
    def weighted_score(bleu, meteor, chrf, weights) -> float
```

**linguistic_analysis.py** - Linguistic error classification
```python
class LinguisticAnalyzer:
    def classify_error(src, pred, ref) -> ErrorType
    # ErrorType: MORPHOLOGY, SYNTAX, SEMANTIC, COMPOUNDING, OOV, UNTRANSLATED
    
    def morphological_check(text: str, lang: str) -> MorphologyReport
    # Validates agreement, case, tense
    
    def syntactic_check(text: str, lang: str) -> SyntaxReport
    # Checks word order, dependency structures
```

**compound_analysis.py** - Compound-specific metrics
```python
class CompoundMetrics:
    def compound_preservation_rate(src_de, pred_de) -> float
    def compound_decomposition_accuracy(pred_de, ref_de) -> float
    def compound_error_rate(results) -> float
```


### Analysis (`analysis/`)

**error_analyzer.py** - Pattern extraction
```python
class ErrorAnalyzer:
    def categorize_errors(translation_results) -> ErrorTaxonomy
    def error_distribution() -> Dict[ErrorType, int]
    def generate_error_report() -> Report
    # Report includes: top error types, worst-performing pairs, examples
```

**compound_processor.py** ⭐ **YOUR INTEGRATION POINT**
```python
from path.to.your.GermanCompoundDecomposer import GermanCompoundDecomposer

class CompoundProcessor:
    """Integrates your decomposer into the evaluation pipeline."""
    
    def __init__(self, decomposer: GermanCompoundDecomposer):
        self.decomposer = decomposer
    
    def analyze_compound_translation(self, src_de: str, pred_de: str, ref_de: str):
        """
        Analyze if compounds were correctly translated.
        Returns: {
            'source_compounds': [(word, decomposition), ...],
            'predicted_compounds': [...],
            'correct_decompositions': int,
            'compound_bleu': float,
            'error_analysis': {...}
        }
        """
        source_compounds = self.decomposer.decompose(src_de)
        pred_compounds = self.decomposer.decompose(pred_de)
        ref_compounds = self.decomposer.decompose(ref_de)
        
        return self._compute_metrics(source_compounds, pred_compounds, ref_compounds)
    
    def enrich_error_report(self, error_report: Dict) -> Dict:
        """Add compound-specific insights to error analysis."""
        # Distinguish between compound errors vs. other morphological errors
        return error_report
```


## 3. Pipeline Flow

### Translation Pipeline

```
INPUT TEXT (EN/DE/RU)
       ↓
[LANGUAGE DETECTION]
       ↓
[PREPROCESSING]
   ├─ Tokenization (language-aware)
   ├─ Normalization (case, punct, diacritics)
   ├─ Compound marking (if source is DE)
   └─ Special token handling
       ↓
[MODEL ROUTING]
   ├─ Select language pair model
   ├─ Load from HuggingFace or checkpoint
   └─ Check if compound-aware variant available
       ↓
[TRANSLATION]
   ├─ Forward pass through transformer
   ├─ Beam search decoding
   └─ Attention scores (optional, for analysis)
       ↓
[POSTPROCESSING]
   ├─ Detokenization
   ├─ Language-specific fixes (compound recombination, case normalization)
   ├─ Punctuation restoration
   └─ Validation (length, special chars)
       ↓
OUTPUT: TranslationResult {
    'text': str,
    'confidence': float,
    'source_lang': str,
    'target_lang': str,
    'metadata': {
        'model': str,
        'tokens': int,
        'compounds_detected': int (if DE)
    }
}
```

### Evaluation Pipeline

```
REFERENCE CORPUS (test set)
       ↓
[BATCH TRANSLATION]
   └─ Translate all test samples
       ↓
[METRIC COMPUTATION]
   ├─ BLEU, METEOR, chrF (corpus + sentence level)
   ├─ Linguistic analysis (morphology, syntax)
   └─ Compound analysis (if source/target is DE)
       ↓
[ERROR CATEGORIZATION]
   ├─ Classify each error: MORPH / SYNTAX / SEMANTIC / COMPOUND / OOV
   ├─ Extract patterns (e.g., "Russian genitive always fails with...")
   └─ Compute error distribution
       ↓
EVALUATION REPORT {
    'metrics': {
        'bleu': float,
        'meteor': float,
        'chrf': float
    },
    'linguistic_analysis': {
        'morphology_errors': int,
        'syntax_errors': int,
        ...
    },
    'compound_metrics': {  # Only if DE involved
        'preservation_rate': float,
        'decomposition_accuracy': float
    },
    'error_examples': [...]
}
```


## 4. Multilingual Support Architecture

**Language routing** (`models/translator.py`):
```python
LANGUAGE_PAIR_MODELS = {
    ('en', 'de'): 'Helsinki-NLP/Opus-MT-en-de',
    ('de', 'en'): 'Helsinki-NLP/Opus-MT-de-en',
    ('en', 'ru'): 'Helsinki-NLP/Opus-MT-en-ru',
    ('ru', 'en'): 'Helsinki-NLP/Opus-MT-ru-en',
    ('de', 'ru'): 'Helsinki-NLP/Opus-MT-de-ru',
    ('ru', 'de'): 'Helsinki-NLP/Opus-MT-ru-de',
}

LANGUAGE_CONFIGS = {
    'en': LanguageConfig(
        preprocessor=EnglishPreprocessor(),
        postprocessor=EnglishPostprocessor(),
        rules=EnglishRules(),
    ),
    'de': LanguageConfig(
        preprocessor=GermanPreprocessor(),
        postprocessor=GermanPostprocessor(),
        rules=GermanRules(),  # Compound handling, capitalization
        compound_decomposer=GermanCompoundDecomposer(),
    ),
    'ru': LanguageConfig(
        preprocessor=RussianPreprocessor(),
        postprocessor=RussianPostprocessor(),
        rules=RussianRules(),  # Case system, aspect
    ),
}

def translate(self, text, src_lang, tgt_lang):
    key = (src_lang, tgt_lang)
    if key not in LANGUAGE_PAIR_MODELS:
        raise ValueError(f"Unsupported pair: {src_lang}→{tgt_lang}")
    
    model_name = LANGUAGE_PAIR_MODELS[key]
    src_config = LANGUAGE_CONFIGS[src_lang]
    tgt_config = LANGUAGE_CONFIGS[tgt_lang]
    
    # Preprocess with source-specific rules
    preprocessed = src_config.preprocessor.preprocess(text)
    
    # Translate
    translated = self.model[key].translate(preprocessed)
    
    # Postprocess with target-specific rules
    output = tgt_config.postprocessor.postprocess(translated)
    
    return output
```

**Benefits**:
- ✓ Clean separation of language-specific logic
- ✓ Easy to add new languages (just add config)
- ✓ Compounds are a first-class concern for DE
- ✓ Extensible for compound-aware models later


## 5. Integration with GermanCompoundDecomposer

**Direct integration point: `analysis/compound_processor.py`**

```python
# In evaluation_pipeline.py

if 'de' in (source_lang, target_lang):
    compound_analyzer = CompoundProcessor(
        decomposer=GermanCompoundDecomposer()
    )
    
    # Enrich evaluation with compound metrics
    eval_results['compound_metrics'] = compound_analyzer.analyze_compound_translation(
        src_de=source_text,
        pred_de=prediction,
        ref_de=reference
    )
    
    # Add compound insights to error report
    eval_results['error_report'] = compound_analyzer.enrich_error_report(
        eval_results['error_report']
    )
```

**What this enables**:
1. **Compound-aware metrics**: Separate compound errors from other morphological errors
2. **Translation analysis**: Did the model preserve compound structure? Decompose correctly?
3. **Research insights**: "Compounds are 15% harder to translate than simple words" 
4. **Future fine-tuning**: Train models with compound-aware loss functions


## 6. Configuration Management

**config/models.yaml**
```yaml
models:
  en-de:
    name: Helsinki-NLP/Opus-MT-en-de
    cache_dir: ./models/checkpoints
    device: cuda
  de-en:
    name: Helsinki-NLP/Opus-MT-de-en
  # ... other pairs

translation:
  beam_size: 5
  max_length: 512
  num_beams: 5
  early_stopping: true
```

**config/evaluation.yaml**
```yaml
metrics:
  bleu:
    smooth_method: exp
    effective_order: true
  meteor:
    language: auto
  chrf:
    order: 6
    beta: 3

linguistic_analysis:
  enable_morphology: true
  enable_syntax: true
  enable_compounding: true  # Auto-enabled if DE involved
```

**config/languages.yaml**
```yaml
languages:
  en:
    lowercase: false
    preserve_case: true
    special_tokens: []
  de:
    lowercase: false
    preserve_case: true
    preserve_compounds: true
    compound_marker: "##"
  ru:
    lowercase: false
    preserve_case: true
    case_system: true
    preserve_aspect: true
```


## 7. Extension Points for Your Workflow

### Adding a New Language Pair (e.g., EN→FR)
1. Add model to `LANGUAGE_PAIR_MODELS`
2. Create `FrenchConfig` in `LANGUAGE_CONFIGS`
3. Implement `FrenchPreprocessor` + `FrenchPostprocessor` + `FrenchRules`
4. Done—translator and evaluation pipelines work automatically

### Plugging in Compound-Aware Fine-Tuning
```python
# In preprocessing/tokenizer.py
def add_compound_markers(text: str, decomposer: GermanCompoundDecomposer):
    """Mark compound boundaries for special loss function."""
    # "Schmetterling" → "Schmetter##ling"
    # Train with compound-aware loss to improve handling

# scripts/train_translator.py
trainer = CompoundAwareTrainer(
    model=base_model,
    decomposer=GermanCompoundDecomposer(),
    loss_fn=CompoundAwareLoss(),
)
```

### Expanding Error Analysis
- Add new error types to `LinguisticAnalyzer.classify_error()`
- Plug in external morphological analyzers (spaCy, pymorphy2 for RU)
- Track error propagation through pipelines (compound error → downstream syntactic error)


## 8. Key Design Decisions

```
|                 Decision                 |                                    Rationale                                     |
|------------------------------------------|----------------------------------------------------------------------------------|
| **Modular preprocessing/postprocessing** | Language-specific rules change independently; compounds warrant special handling |
|     **Compound processor as plugin**     |        Keeps MT system clean; allows your decomposer to evolve separately        |
|          **Multilingual router**         |      Single unified interface vs. separate scripts; easier to compare models     |
|       **Structured error taxonomy**      |                Move beyond BLEU; understand *why* translations fail              |
|     **Config files over hardcoding**     |  Reproducibility; easy to swap models, tune hyperparameters, document decisions  |
|     **Separate evaluation pipeline**     |      Full hypothesis sweep without re-translating; enables batch experiments     |
```

## 9. Expected Outputs

### Pipeline Outputs
- ✓ Translated text with confidence scores
- ✓ Metadata (model, tokens, language pair)
- ✓ Error classifications (morphology, syntax, compound)
- ✓ Linguistic analysis (morphological agreement, verb tense)

### Evaluation Reports
- ✓ BLEU / METEOR / chrF (corpus + sentence level)
- ✓ Error distribution heatmap (error type × language pair)
- ✓ Compound-specific metrics (preservation rate, decomposition accuracy)
- ✓ Example translations with annotations (color-coded errors)
- ✓ Reproducible experiment logs (config, random seeds, timestamps)


## Implementation Sequence

1. **Week 1**: Folder structure + data loaders + basic translator
2. **Week 2**: Preprocessing/postprocessing for all 3 languages
3. **Week 3**: Evaluation metrics + linguistic analysis framework
4. **Week 4**: Error analyzer + compound processor integration
5. **Week 5**: Pipelines + CLI scripts + notebooks
6. **Week 6**: Experiments + documentation + hyperparameter tuning

This architecture supports your portfolio progression:
- **Preprocessing** → tokenization + language-aware rules
- **Morphology** → linguistic analysis of errors + compound decomposition
- **Semantics** → translation captures meaning (METEOR, sentence-level BLEU)
- **Cross-lingual** → RU↔DE routes through decomposition analysis