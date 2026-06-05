"""Linguistic error analysis for machine translation."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List
import json


class ErrorCategory(str, Enum):
    """Categories of translation errors."""
    LEXICAL_MISMATCH = "lexical_mismatch"
    MORPHOLOGICAL_LOSS = "morphological_loss"
    AGREEMENT_ERROR = "agreement_error"
    COMPOUND_BREAKDOWN = "compound_breakdown"
    WORD_ORDER_SHIFT = "word_order_shift"
    SEMANTIC_DRIFT = "semantic_drift"
    OOV_UNTRANSLATED = "oov_untranslated"
    VOICE_ASPECT_LOSS = "voice_aspect_loss"
    PRONOUN_ERROR = "pronoun_error"
    NO_ERROR = "no_error"


@dataclass
class ErrorSpan:
    """A detected error in the translation."""
    category: ErrorCategory
    source_span: str
    target_span: str
    reference_span: Optional[str] = None
    severity: str = "medium"  # low, medium, high
    explanation: str = ""
    source_pos: tuple[int, int] = field(default_factory=tuple)  # (start, end)
    target_pos: tuple[int, int] = field(default_factory=tuple)
    confidence: float = 0.7


@dataclass
class SentenceLinguisticAnalysis:
    """Complete linguistic analysis of a translated sentence."""
    source_text: str
    target_text: str
    reference_text: Optional[str] = None
    source_lang: str = "unknown"
    target_lang: str = "unknown"

    # Analysis results
    errors: List[ErrorSpan] = field(default_factory=list)
    error_categories: dict = field(default_factory=dict)  # category → count
    overall_quality: str = "unknown"  # excellent, good, fair, poor
    linguistic_notes: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "source": self.source_text,
            "target": self.target_text,
            "reference": self.reference_text,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "errors": [
                {
                    "category": err.category.value,
                    "source_span": err.source_span,
                    "target_span": err.target_span,
                    "reference_span": err.reference_span,
                    "severity": err.severity,
                    "explanation": err.explanation,
                    "confidence": err.confidence,
                }
                for err in self.errors
            ],
            "error_summary": self.error_categories,
            "overall_quality": self.overall_quality,
            "linguistic_notes": self.linguistic_notes,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to formatted JSON."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


@dataclass
class BatchLinguisticAnalysis:
    """Analysis results for multiple sentence pairs."""
    analyses: List[SentenceLinguisticAnalysis] = field(default_factory=list)
    source_lang: str = "unknown"
    target_lang: str = "unknown"

    def error_distribution(self) -> dict[str, int]:
        """Get distribution of error types across all sentences."""
        distribution = {}
        for analysis in self.analyses:
            for category, count in analysis.error_categories.items():
                distribution[category] = distribution.get(category, 0) + count
        return distribution

    def quality_distribution(self) -> dict[str, int]:
        """Get distribution of quality ratings."""
        distribution = {}
        for analysis in self.analyses:
            quality = analysis.overall_quality
            distribution[quality] = distribution.get(quality, 0) + 1
        return distribution

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "sentence_count": len(self.analyses),
            "error_distribution": self.error_distribution(),
            "quality_distribution": self.quality_distribution(),
            "analyses": [a.to_dict() for a in self.analyses],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to formatted JSON."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)