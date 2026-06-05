"""Language-specific preprocessing and postprocessing utilities."""

from abc import ABC, abstractmethod
import re


class LanguagePreprocessor(ABC):
    """Base class for language-specific preprocessing."""

    @abstractmethod
    def preprocess(self, text: str) -> str:
        """Preprocess text before translation."""
        pass


class LanguagePostprocessor(ABC):
    """Base class for language-specific postprocessing."""

    @abstractmethod
    def postprocess(self, text: str) -> str:
        """Postprocess translated text."""
        pass


class EnglishPreprocessor(LanguagePreprocessor):
    """English-specific preprocessing."""

    def preprocess(self, text: str) -> str:
        """Normalize English text."""
        # Preserve original casing (English often has meaningful case)
        text = text.strip()
        return text


class EnglishPostprocessor(LanguagePostprocessor):
    """English-specific postprocessing."""

    def postprocess(self, text: str) -> str:
        """Postprocess English text."""
        text = text.strip()
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        return text


class GermanPreprocessor(LanguagePreprocessor):
    """German-specific preprocessing."""

    def preprocess(self, text: str) -> str:
        """Normalize German text, preserving capitalization and umlauts."""
        text = text.strip()
        # Preserve German capitalization (important for nouns)
        return text


class GermanPostprocessor(LanguagePostprocessor):
    """German-specific postprocessing."""

    def postprocess(self, text: str) -> str:
        """Postprocess German text."""
        text = text.strip()
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        # Capitalize after sentence-ending punctuation (German always capitalizes after periods)
        text = re.sub(r'([.!?])\s+([a-z])', lambda m: m.group(1) + ' ' + m.group(2).upper(), text)
        return text


class RussianPreprocessor(LanguagePreprocessor):
    """Russian-specific preprocessing."""

    def preprocess(self, text: str) -> str:
        """Normalize Russian text."""
        text = text.strip()
        # Preserve Cyrillic characters and case
        return text


class RussianPostprocessor(LanguagePostprocessor):
    """Russian-specific postprocessing."""

    def postprocess(self, text: str) -> str:
        """Postprocess Russian text."""
        text = text.strip()
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        # Capitalize after sentence-ending punctuation
        text = re.sub(r'([.!?])\s+([а-яёa-z])', lambda m: m.group(1) + ' ' + m.group(2).upper(), text)
        return text


def get_preprocessor(lang: str) -> LanguagePreprocessor:
    """Get preprocessor for a language."""
    preprocessors = {
        "en": EnglishPreprocessor(),
        "de": GermanPreprocessor(),
        "ru": RussianPreprocessor(),
    }
    lang_lower = lang.lower()
    if lang_lower not in preprocessors:
        raise ValueError(f"No preprocessor for language: {lang}")
    return preprocessors[lang_lower]


def get_postprocessor(lang: str) -> LanguagePostprocessor:
    """Get postprocessor for a language."""
    postprocessors = {
        "en": EnglishPostprocessor(),
        "de": GermanPostprocessor(),
        "ru": RussianPostprocessor(),
    }
    lang_lower = lang.lower()
    if lang_lower not in postprocessors:
        raise ValueError(f"No postprocessor for language: {lang}")
    return postprocessors[lang_lower]