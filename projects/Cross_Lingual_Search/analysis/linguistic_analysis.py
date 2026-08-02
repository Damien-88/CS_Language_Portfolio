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
        Lightweight heuristic for detecting possible morphological variation.
        """

        return (
            retrieved.lower() != expected.lower() and
            len(retrieved) == len(expected)
        )