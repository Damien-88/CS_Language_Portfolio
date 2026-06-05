# Machine Translation System

A production-oriented, modular machine translation system supporting English ↔ German ↔ Russian with transformer-based inference and integrated linguistic error analysis.

## Features

- **Multilingual Support**: English ↔ German ↔ Russian (6 language pairs)
- **HuggingFace Transformers**: Opus-MT models for high-quality translation
- **Batch Processing**: Efficient translation of multiple texts
- **Config-Driven**: Model selection and hyperparameters via configuration
- **Type-Safe**: Full type hints and dataclasses for clean architecture
- **Modular Design**: Extensible preprocessing, postprocessing, and evaluation
- **Portfolio-Grade**: Clean separation of concerns suitable for production

## Supported Language Pairs

- English → German (EN→DE)
- German → English (DE→EN)
- English → Russian (EN→RU)
- Russian → English (RU→EN)
- German → Russian (DE→RU) — *stub support available*
- Russian → German (RU→DE) — *stub support available*

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Simple API

```python
from api import translate

# Translate a single sentence
result = translate("Hello, how are you?", "en", "de")
print(result.text)  # Output: "Hallo, wie geht es dir?"
```

### Batch Translation

```python
from api import translate_batch

texts = [
    "Good morning",
    "Beautiful day",
]
results = translate_batch(texts, "en", "de")

for result in results:
    print(result.text)
```

### Using the Pipeline Directly

```python
from pipeline.translator_pipeline import TranslatorPipeline

pipeline = TranslatorPipeline()

# Get supported pairs
pairs = pipeline.supported_pairs()

# Translate
result = pipeline.translate("Hello", "en", "de")
print(result.text)
```

## Project Structure

```
Machine_Translation/
├── config.py                    # Configuration dataclasses
├── api.py                       # Public API functions
├── examples.py                  # Usage examples
├── models/
│   ├── base.py                 # Abstract base classes
│   └── translator.py           # TranslationModel implementation
├── preprocessing/
│   └── language_utils.py       # Language-specific processors
├── pipeline/
│   └── translator_pipeline.py  # Pipeline orchestration
└── requirements.txt            # Dependencies
```

## Architecture

### Core Components

**TranslationModel**
- Loads HuggingFace models
- Handles tokenization and generation
- Manages preprocessing/postprocessing
- Returns `TranslationResult` with metadata

**TranslatorPipeline**
- Routes language pairs to correct models
- Caches loaded models
- Provides batch processing
- Handles model lifecycle

**Language Processors**
- Bidirectional: `Preprocessor` (normalize input) + `Postprocessor` (fix output)
- Language-specific: German capitalization, Russian case handling, etc.
- Extensible: Add new languages by implementing base classes

### Configuration

All model selections and hyperparameters are config-driven:

```python
from config import TranslationConfig

config = TranslationConfig.default()
# Or customize:
config.num_beams = 3
config.batch_size = 16
```

## TranslationResult

Every translation returns a structured result:

```python
@dataclass
class TranslationResult:
    text: str                    # Translated text
    source_lang: str             # Source language code
    target_lang: str             # Target language code
    model_name: str              # Model used
    confidence: float            # Confidence score
    source_tokens: int           # Input token count
    target_tokens: int           # Output token count
    metadata: dict               # Additional info
```

## Performance Considerations

- **Batch Processing**: Translate multiple texts at once for better GPU utilization
- **Model Caching**: Models are cached after first load
- **Device Management**: Automatically uses CUDA if available

```python
from config import TranslationConfig

config = TranslationConfig.default()
config.device = "cuda"  # or "cpu"
config.batch_size = 32

pipeline = TranslatorPipeline(config)
```

## Extending the System

### Add a New Language

1. Create preprocessor/postprocessor in `preprocessing/language_utils.py`
2. Add to config in `config.py`
3. Models automatically available in pipeline

### Add a New Language Pair

```python
from config import TranslationConfig, LanguagePairConfig

config = TranslationConfig.default()
config.language_pairs[("fr", "en")] = LanguagePairConfig(
    source_lang="fr",
    target_lang="en",
    model_name="Helsinki-NLP/Opus-MT-fr-en",
)
```

### Integrate GermanCompoundDecomposer

```python
# In analysis/compound_processor.py
from preprocessing.language_utils import GermanCompoundDecomposer

class CompoundProcessor:
    def __init__(self, decomposer):
        self.decomposer = decomposer
    
    def analyze_translation(self, src_de, pred_de):
        # Decompose and compare
        ...
```

## Example Usage

Run the examples:

```bash
python examples.py
```

## Testing

```bash
# Run basic tests (when test suite is added)
pytest tests/
```

## Future Development

- [ ] Linguistic error analysis (morphology, syntax evaluation)
- [ ] Compound-aware metrics and analysis
- [ ] Integration with GermanCompoundDecomposer
- [ ] Fine-tuning scripts for domain adaptation
- [ ] Evaluation metrics (BLEU, METEOR, chrF)
- [ ] Web API and UI

## Dependencies

- `transformers>=4.30.0` - HuggingFace Transformers
- `torch>=2.0.0` - PyTorch
- `pydantic>=2.0.0` - Data validation
- `numpy>=1.24.0` - Numerical computing
- `tqdm>=4.65.0` - Progress bars

## License

Part of the Computational Linguistics Portfolio

## References

- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [Opus-MT Models](https://huggingface.co/Helsinki-NLP)
- [Machine Translation Evaluation](https://en.wikipedia.org/wiki/BLEU_score)