"""Public API for machine translation."""

from typing import Optional
from projects.Machine_Translation.pipeline.translator_pipeline import TranslatorPipeline
from projects.Machine_Translation.models.base import TranslationResult
from projects.Machine_Translation.config import TranslationConfig

# Global pipeline instance
_pipeline: Optional[TranslatorPipeline] = None


def initialize(config: Optional[TranslationConfig] = None) -> TranslatorPipeline:
    """
    Initialize the translation pipeline.

    Args:
        config: Translation configuration (uses default if None)

    Returns:
        Initialized TranslatorPipeline
    """
    global _pipeline
    _pipeline = TranslatorPipeline(config)
    return _pipeline


def translate(
    text: str,
    source_lang: str,
    target_lang: str,
) -> TranslationResult:
    """
    Translate text from source to target language.

    Simple API function that handles initialization.

    Args:
        text: Text to translate
        source_lang: Source language code (e.g., 'en', 'de', 'ru')
        target_lang: Target language code

    Returns:
        TranslationResult with translated text and metadata

    Raises:
        ValueError: If language pair is not supported

    Example:
        >>> result = translate("Hello world", "en", "de")
        >>> print(result.text)
        "Hallo Welt"
    """
    global _pipeline
    if _pipeline is None:
        _pipeline = initialize()

    return _pipeline.translate(text, source_lang, target_lang)


def translate_batch(
    texts: list[str],
    source_lang: str,
    target_lang: str,
) -> list[TranslationResult]:
    """
    Translate a batch of texts.

    Args:
        texts: List of texts to translate
        source_lang: Source language code
        target_lang: Target language code

    Returns:
        List of TranslationResult objects

    Example:
        >>> results = translate_batch(
        ...     ["Hello", "Good morning"],
        ...     "en",
        ...     "de"
        ... )
    """
    global _pipeline
    if _pipeline is None:
        _pipeline = initialize()

    return _pipeline.translate_batch(texts, source_lang, target_lang)


def get_pipeline() -> TranslatorPipeline:
    """Get the global pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = initialize()
    return _pipeline