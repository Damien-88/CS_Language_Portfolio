"""Analysis package for linguistic error detection."""

from analysis.error_types import (
    ErrorCategory,
    ErrorSpan,
    SentenceLinguisticAnalysis,
    BatchLinguisticAnalysis,
)
from analysis.error_analyzer import TranslationErrorAnalyzer
from analysis.compound_processor import CompoundProcessor
from analysis.morphological_checker import (
    MorphologicalComparator,
    GermanMorphologyAnalyzer,
    RussianMorphologyAnalyzer,
)

__all__ = [
    "ErrorCategory",
    "ErrorSpan",
    "SentenceLinguisticAnalysis",
    "BatchLinguisticAnalysis",
    "TranslationErrorAnalyzer",
    "CompoundProcessor",
    "MorphologicalComparator",
    "GermanMorphologyAnalyzer",
    "RussianMorphologyAnalyzer",
]