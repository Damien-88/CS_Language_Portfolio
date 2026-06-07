"""Translation pipeline orchestration."""

from typing import Dict, Optional
from models.translator import TranslationModel
from models.base import TranslationResult
from config import TranslationConfig


class TranslatorPipeline:
    """
    Manages translation for multiple language pairs.

    Handles model caching, routing, and batch processing.
    Supports optional morphological preprocessing (e.g., German compounds).
    """

    def __init__(
        self,
        config: Optional[TranslationConfig] = None,
        decomposer: Optional[object] = None,
    ):
        """
        Initialize pipeline.

        Args:
            config: Translation configuration (uses default if None)
            decomposer: Optional external GermanCompoundDecomposer instance
        """
        self.config = config or TranslationConfig.default()
        self.decomposer = decomposer
        self._models: Dict[tuple[str, str], TranslationModel] = {}

    def set_decomposer(self, decomposer: object) -> None:
        """
        Set or replace the decomposer.

        Useful for injecting custom decomposer after initialization.
        """
        self.decomposer = decomposer
        # Clear cache to force reload with new decomposer
        self.clear_cache()

    def _get_model(self, src_lang: str, tgt_lang: str) -> TranslationModel:
        """
        Get or load model for language pair.

        Args:
            src_lang: Source language code (e.g., 'en')
            tgt_lang: Target language code (e.g., 'de')

        Returns:
            TranslationModel instance

        Raises:
            ValueError: If language pair is not supported
        """
        key = (src_lang.lower(), tgt_lang.lower())

        # Return cached model if available
        if key in self._models:
            return self._models[key]

        # Get configuration for this pair
        pair_config = self.config.get_pair_config(src_lang, tgt_lang)

        # Load and cache new model (pass decomposer if available)
        model = TranslationModel(pair_config, self.config, decomposer=self.decomposer)
        self._models[key] = model
        return model

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> TranslationResult:
        """
        Translate a single text.

        Args:
            text: Source text to translate
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            TranslationResult
        """
        model = self._get_model(source_lang, target_lang)
        return model.translate(text)

    def translate_batch(
        self,
        texts: list[str],
        source_lang: str,
        target_lang: str,
    ) -> list[TranslationResult]:
        """
        Translate a batch of texts.

        Args:
            texts: List of source texts
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            List of TranslationResult objects
        """
        model = self._get_model(source_lang, target_lang)
        return model.translate_batch(texts)

    def supported_pairs(self) -> list[tuple[str, str]]:
        """Get list of supported language pairs."""
        return list(self.config.language_pairs.keys())

    def clear_cache(self):
        """Clear all cached models."""
        self._models.clear()

    def _get_model(self, src_lang: str, tgt_lang: str) -> TranslationModel:
        """
        Get or load model for language pair.

        Args:
            src_lang: Source language code (e.g., 'en')
            tgt_lang: Target language code (e.g., 'de')

        Returns:
            TranslationModel instance

        Raises:
            ValueError: If language pair is not supported
        """
        key = (src_lang.lower(), tgt_lang.lower())

        # Return cached model if available
        if key in self._models:
            return self._models[key]

        # Get configuration for this pair
        pair_config = self.config.get_pair_config(src_lang, tgt_lang)

        # Load and cache new model
        model = TranslationModel(pair_config, self.config)
        self._models[key] = model
        return model

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> TranslationResult:
        """
        Translate a single text.

        Args:
            text: Source text to translate
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            TranslationResult
        """
        model = self._get_model(source_lang, target_lang)
        return model.translate(text)

    def translate_batch(
        self,
        texts: list[str],
        source_lang: str,
        target_lang: str,
    ) -> list[TranslationResult]:
        """
        Translate a batch of texts.

        Args:
            texts: List of source texts
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            List of TranslationResult objects
        """
        model = self._get_model(source_lang, target_lang)
        return model.translate_batch(texts)

    def supported_pairs(self) -> list[tuple[str, str]]:
        """Get list of supported language pairs."""
        return list(self.config.language_pairs.keys())

    def clear_cache(self):
        """Clear all cached models."""
        self._models.clear()