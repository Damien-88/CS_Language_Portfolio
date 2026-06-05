"""Configuration for multilingual translation models."""

from dataclasses import dataclass
from typing import Dict


@dataclass
class LanguagePairConfig:
    """Configuration for a specific language pair."""
    source_lang: str
    target_lang: str
    model_name: str
    max_length: int = 512
    beam_size: int = 5


@dataclass
class TranslationConfig:
    """Global translation configuration."""
    # Language pair models (source → target)
    language_pairs: Dict[tuple[str, str], LanguagePairConfig]

    # Inference settings
    device: str = "cuda"
    batch_size: int = 8
    num_beams: int = 5
    early_stopping: bool = True

    # Model caching
    cache_dir: str = "./models/checkpoints"

    @classmethod
    def default(cls) -> "TranslationConfig":
        """Create default configuration with Opus-MT models."""
        language_pairs = {
            ("en", "de"): LanguagePairConfig(
                source_lang="en",
                target_lang="de",
                model_name="Helsinki-NLP/Opus-MT-en-de",
            ),
            ("de", "en"): LanguagePairConfig(
                source_lang="de",
                target_lang="en",
                model_name="Helsinki-NLP/Opus-MT-de-en",
            ),
            ("ru", "en"): LanguagePairConfig(
                source_lang="ru",
                target_lang="en",
                model_name="Helsinki-NLP/Opus-MT-ru-en",
            ),
            ("en", "ru"): LanguagePairConfig(
                source_lang="en",
                target_lang="ru",
                model_name="Helsinki-NLP/Opus-MT-en-ru",
            ),
        }
        return cls(language_pairs=language_pairs)

    def get_pair_config(self, src_lang: str, tgt_lang: str) -> LanguagePairConfig:
        """Get configuration for a language pair."""
        key = (src_lang.lower(), tgt_lang.lower())
        if key not in self.language_pairs:
            supported = ", ".join([f"{s}→{t}" for s, t in self.language_pairs.keys()])
            raise ValueError(
                f"Unsupported language pair: {src_lang}→{tgt_lang}. "
                f"Supported pairs: {supported}"
            )
        return self.language_pairs[key]