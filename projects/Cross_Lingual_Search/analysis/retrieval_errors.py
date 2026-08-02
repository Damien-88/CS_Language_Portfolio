"""
Retrieval error categories for mutli-lingual semantic search.
Rather than treating retrieval as simply successful or unsiccessful, this module
defines linguistically motivated error categories that describe why a semantic
retrieval may have failed.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

class RetrievalErrorCategory(Enum):
    """Categories describing common retrieval failures."""
    MORPHOLOGICAL_VARIATION = "morphological_variation"
    COMPOUND_REPRESENTATION = "compound_representation"
    SYNONYM_VARIATION = "synonym_variation"
    WORD_ORDER = "word_order"
    SEMANTIC_DRIFT = "semantic_drift"
    LANGUAGE_MISMATCH = "language_mismatch"
    OUT_OF_VOCABULARY = "out_of_vocabulary"
    UNKNOWN = "unknown"

@dataclass(slots = True)
class RetrievalError:
    """Represents a single retrieval error."""

    category: RetrievalErrorCategory
    explanation: str
    query: str
    retrieved_text: str
    expected_text: str

class RetrievalErrorCollection:
    """Stores retrieval errors generated during evaluation."""

    def __init__(self):
        self.errors = []

    def add_error(self, error):
        """Add a retrieval error."""
        self.errors.append(error)

    def all_errors(self):
        """Return all recorded retrieval errors."""
        return list(self.errors)

    def error_count(self):
        """Return the total number of retrieval errors."""
        return len(self.errors)

    def by_category(self, category):
        """Return all retrieval errors of a specific category."""
        return [err for err in self.errors if err.category == category]

    def error_summary(self):
        """Count errors by linguistic category and return summary map."""
        counts = {}

        for err in self.errors:
            counts[err.category] = counts.get(err.category, 0) + 1

        return counts

    def clear_errors(self):
        """Remove all stored errors."""
        self.errors.clear()