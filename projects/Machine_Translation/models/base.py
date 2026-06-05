"""Base classes for translation models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class TranslationResult:
    """Result of a translation."""
    text: str
    source_lang: str
    target_lang: str
    model_name: str
    confidence: float = 1.0
    source_tokens: int = 0
    target_tokens: int = 0
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseTranslator(ABC):
    """Abstract base class for translation models."""

    @abstractmethod
    def translate(self, text: str) -> TranslationResult:
        """Translate a single text."""
        pass

    @abstractmethod
    def translate_batch(self, texts: list[str]) -> list[TranslationResult]:
        """Translate a batch of texts."""
        pass

    @abstractmethod
    def supports_language_pair(self, src_lang: str, tgt_lang: str) -> bool:
        """Check if model supports a language pair."""
        pass