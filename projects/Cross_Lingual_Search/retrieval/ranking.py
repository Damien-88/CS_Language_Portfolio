"""
Ranking layer for cross-lingual semantic retrieval.
This module reranks retrieved documents after FAISS similarity search.
The goal is not to replace semantic similarity, but to allow additional
linguistic and metadata-aware signals.
"""

from dataclasses import dataclass
from retrieval.semantic_search import SearchResult

@dataclass
class RankedResult:
    """Result after ranking."""

    document: object
    retrieval_score: float
    final_score: float
    ranking_reason: str



class SemanticRanker:
    """Explainable reranking layer."""

    def __init__(self, semantic_weight = 1.0):
        self.semantic_weight = semantic_weight

    def rank(self, results):
        """Rerank retrieved documents."""
        ranked_results = []

        for result in results:
            final_score = (result.score * self.semantic_weight)

            ranked_results.append(
                RankedResult(
                    document = result.document,
                    retrieval_score = result.score,
                    final_score = final_score,
                    ranking_reason = "semantic_similarity"
                )
            )


        ranked_results.sort(
            key = lambda x: x.final_score,
            reverse = True
        )

        return ranked_results