"""
Linguistic analysis for cross-lingual semantic retrieval.
This module analyzes retrieved search results and attempts to explain retrieval
successes and failures using linguistically motivated categories rather than
simple correctness.
"""

from __future__ import annotations

import sys
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from retrieval.semantic_search import SearchResult
from analysis.retrieval_errors import RetrievalError, RetrievalErrorCategory

class LinguisticAnalyzer:
    """Performs lightweight linguistic analysis of semantic retrieval results."""

    def analyze(self, query, retrieved_result, expected_text):
        """
        Analyze a retrived result against the expected text and categorize the 
        retrieval error.
        """
        errors = []
        retrieved_text = retrieved_result.document.text

        # Check for exact match
        if retrieved_text == expected_text:
            return errors  # No errors, exact match

        # German compound heuristic
        if self.possible_compound_difference(retrieved_text, expected_text):
            errors.append(
                RetrievalError(
                    category = RetrievalErrorCategory.COMPOUND_REPRESENTATION,
                    explanation  = (
                        "Possible German compound representation "
                        "difference detected"
                    ),
                    query = query,
                    retrieved_text = retrieved_text,
                    expected_text = expected_text
                )    
            )

        # Word order heuristic
        elif self.possible_word_order_difference(retrieved_text, expected_text):
            errors.append(
                RetrievalError(
                    category = RetrievalErrorCategory.WORD_ORDER,
                    explanation = (
                        "Retrieved and expected texts share the same words "
                        "but differ in order"
                    ),
                    query = query,
                    retrieved_text = retrieved_text,
                    expected_text = expected_text
                )
            )

        # Morphological variation heuristic
        elif self.possible_morphology_difference(retrieved_text, expected_text):
            errors.append(
                RetrievalError(
                    category = RetrievalErrorCategory.MORPHOLOGICAL_VARIATION,
                    explanation = (
                        "Possible inflectional or morphological "
                        "variation detected"
                    ),
                    query = query,
                    retrieved_text = retrieved_text,
                    expected_text = expected_text
                )
            )

        # Script-based language mismatch heuristic
        elif self.possible_language_mismatch(retrieved_result, expected_text):
            errors.append(
                RetrievalError(
                    category = RetrievalErrorCategory.LANGUAGE_MISMATCH,
                    explanation = (
                        "Retrieved document's language does not match the "
                        "script used in the expected text"
                    ),
                    query = query,
                    retrieved_text = retrieved_text,
                    expected_text = expected_text
                )
            )

        # Default case: categorize as unknown error
        else:
            errors.append(
                RetrievalError(
                    category = RetrievalErrorCategory.UNKNOWN,
                    explanation = (
                        "Retrieval differs from expectation but no "
                        "linguistic pattern was identified."
                    ),
                    query = query,
                    retrieved_text = retrieved_text,
                    expected_text = expected_text
                )
            )

        return errors

    def possible_compound_difference(self, retrieved, expected):
        """
        Extremely lightweight heuristic for identifying possible German
        compound differences.
        """
        return (
            len(expected.split()) == 1 and len(retrieved.split()) > 1
            ) or (
                len(retrieved.split()) == 1 and len(expected.split()) > 1
            )

    def possible_morphology_difference(self, retrieved, expected):
        """
        Lightweight heuristic for detecting possible morphological variation:
        same length, but a different surface form (case included, since a
        pure case difference is itself a valid morphological variant here).
        """

        return (
            retrieved != expected and
            len(retrieved) == len(expected)
        )

    def possible_word_order_difference(self, retrieved, expected):
        """
        Lightweight heuristic for detecting possible word order differences:
        both texts contain the same words but in a different sequence.
        """

        retrieved_words = retrieved.lower().split()
        expected_words = expected.lower().split()

        return (
            retrieved_words != expected_words
            and sorted(retrieved_words) == sorted(expected_words)
        )

    def possible_language_mismatch(self, retrieved_result, expected_text):
        """
        Lightweight heuristic for detecting a possible language mismatch.
        Only distinguishes Cyrillic (Russian) from Latin-script languages,
        since script is the most reliable signal available from raw text.
        """

        expected_is_russian = any(
            "а" <= character <= "я" or character in "ёЁ"
            for character in expected_text.lower()
        )
        retrieved_is_russian = retrieved_result.document.language == "Russian"

        return expected_is_russian != retrieved_is_russian