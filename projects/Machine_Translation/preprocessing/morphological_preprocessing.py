"""Morphological preprocessing with optional compound decomposition."""

from typing import Optional, List, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class PreprocessedText:
    """Result of preprocessing."""
    original: str
    preprocessed: str
    tokens: List[str]
    decompositions: dict = None  # token → decomposition list
    metadata: dict = None

    def __post_init__(self):
        if self.decompositions is None:
            self.decompositions = {}
        if self.metadata is None:
            self.metadata = {}


class MorphologicalPreprocessor(ABC):
    """Abstract base for morphological preprocessing."""

    @abstractmethod
    def preprocess(self, text: str) -> PreprocessedText:
        """Preprocess text."""
        pass


class GermanCompoundPreprocessor(MorphologicalPreprocessor):
    """
    German compound decomposition preprocessor.

    Optionally decomposes German compounds to aid translation.
    Example: "Schmetterling" → "Schmetterling" (or "Schmetterling|Haus" for compounds)
    """

    def __init__(self, decomposer: Optional[object] = None, enable: bool = False):
        """
        Initialize German compound preprocessor.

        Args:
            decomposer: Optional external GermanCompoundDecomposer instance
            enable: Whether to enable decomposition
        """
        self.decomposer = decomposer
        self.enable = enable
        self.compound_marker = "|"  # Marker between compound parts

    def set_decomposer(self, decomposer: object) -> None:
        """Set or replace the decomposer."""
        self.decomposer = decomposer

    def _decompose_word(self, word: str) -> List[str]:
        """
        Decompose a single word using external decomposer if available.

        Args:
            word: German word to decompose

        Returns:
            List of morphemes, or [word] if not decomposable
        """
        if not self.decomposer or not self.enable:
            return [word]

        try:
            decomposition = self.decomposer.decompose(word)
            # If it's already a list/tuple, return it
            if isinstance(decomposition, (list, tuple)):
                return list(decomposition)
            # Otherwise treat as single word
            return [word]
        except Exception:
            # If decomposition fails, return original word
            return [word]

    def preprocess(self, text: str) -> PreprocessedText:
        """
        Preprocess German text with optional compound decomposition.

        Args:
            text: German text to preprocess

        Returns:
            PreprocessedText with original, preprocessed, and decomposition info
        """
        tokens = text.split()
        decompositions = {}
        preprocessed_tokens = []

        for token in tokens:
            # Decompose if enabled and we have a decomposer
            if self.enable and self.decomposer:
                decomposed = self._decompose_word(token)
                decompositions[token] = decomposed

                # If compound (multiple parts), join with marker
                if len(decomposed) > 1:
                    preprocessed_token = self.compound_marker.join(decomposed)
                else:
                    preprocessed_token = token
            else:
                # No decomposition
                preprocessed_token = token
                decompositions[token] = [token]

            preprocessed_tokens.append(preprocessed_token)

        preprocessed_text = " ".join(preprocessed_tokens)

        return PreprocessedText(
            original=text,
            preprocessed=preprocessed_text,
            tokens=tokens,
            decompositions=decompositions,
            metadata={
                "language": "de",
                "decomposition_enabled": self.enable,
                "decomposition_count": sum(
                    1 for d in decompositions.values() if len(d) > 1
                ),
                "compound_marker": self.compound_marker,
            },
        )


class PassthroughPreprocessor(MorphologicalPreprocessor):
    """Passthrough preprocessor (no-op)."""

    def preprocess(self, text: str) -> PreprocessedText:
        """Return text unchanged."""
        return PreprocessedText(
            original=text,
            preprocessed=text,
            tokens=text.split(),
            decompositions={token: [token] for token in text.split()},
            metadata={"language": "unknown", "preprocessor": "passthrough"},
        )


class PreprocessorFactory:
    """Factory for creating appropriate preprocessors."""

    @staticmethod
    def create(
        language: str,
        decomposer: Optional[object] = None,
        enable_morphological: bool = False,
    ) -> MorphologicalPreprocessor:
        """
        Create preprocessor for language.

        Args:
            language: Language code ('de', 'en', 'ru', etc.)
            decomposer: Optional external decomposer
            enable_morphological: Whether to enable morphological preprocessing

        Returns:
            Appropriate MorphologicalPreprocessor instance
        """
        if language.lower() == "de" and enable_morphological:
            return GermanCompoundPreprocessor(decomposer=decomposer, enable=True)
        elif language.lower() == "de":
            return GermanCompoundPreprocessor(decomposer=decomposer, enable=False)
        else:
            # Other languages: passthrough
            return PassthroughPreprocessor()


class PreprocessingPipeline:
    """
    Manager for optional preprocessing steps.

    Allows chaining multiple preprocessors if needed.
    """

    def __init__(self):
        self.preprocessors: List[Tuple[str, MorphologicalPreprocessor]] = []

    def add(self, name: str, preprocessor: MorphologicalPreprocessor) -> None:
        """Add a preprocessor to the pipeline."""
        self.preprocessors.append((name, preprocessor))

    def apply(self, text: str, language: str) -> PreprocessedText:
        """
        Apply preprocessing pipeline.

        Args:
            text: Input text
            language: Language code

        Returns:
            Final PreprocessedText after all steps
        """
        result = PreprocessedText(
            original=text,
            preprocessed=text,
            tokens=text.split(),
        )

        for name, preprocessor in self.preprocessors:
            result = preprocessor.preprocess(result.preprocessed)

        return result

    def clear(self) -> None:
        """Clear all preprocessors."""
        self.preprocessors.clear()
