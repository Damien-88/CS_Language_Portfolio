"""
Experiment 1: Cross-Lingual Retrieval.

This experiment addresses Research Question 1:

Can multilingual embeddings align meaning across languages closely enough to
retrieve semantically equivalent documents without translation?

Pipeline:
   English Query -> Multilingual Encoder -> German/Russian Retrieval

Evaluation:
- Recall@K
- Mean Reciprocal Rank (MRR)
- Average semantic similarity of relevant results
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from retrieval.semantic_search import SearchResult


@dataclass
class CrossLingualQueryResult:
    """Stores retrieval results for a single cross-lingual query."""

    query: str
    query_language: str
    retrieved_documents: list["SearchResult"]
    relevant_document_ids: list[int] = field(default_factory=list)

    def retrieved_document_ids(self) -> list[int]:
        """Document IDs of the retrieved results, in ranked order."""

        return [
            result.document.document_id
            for result in self.retrieved_documents
        ]

    def recall_at_k(self, k = None) -> float:
        """Fraction of relevant documents found in the top k results."""

        if not self.relevant_document_ids:
            return 0.0

        retrieved_ids = self.retrieved_document_ids()[:k]
        relevant_ids = set(self.relevant_document_ids)

        found = sum(1 for doc_id in retrieved_ids if doc_id in relevant_ids)

        return found / len(relevant_ids)

    def reciprocal_rank(self) -> float:
        """Inverse rank of the first relevant result, or 0.0 if none found."""

        relevant_ids = set(self.relevant_document_ids)

        for rank, doc_id in enumerate(self.retrieved_document_ids(), start = 1):
            if doc_id in relevant_ids:
                return 1 / rank

        return 0.0

    def relevant_scores(self) -> list[float]:
        """Similarity scores of retrieved documents that are actually relevant."""

        relevant_ids = set(self.relevant_document_ids)

        return [
            float(result.score)
            for result in self.retrieved_documents
            if result.document.document_id in relevant_ids
        ]


class CrossLingualRetrievalExperiment:
    """
    Measures whether multilingual embeddings retrieve semantically
    equivalent documents across English, German, and Russian.
    """

    def __init__(self, search_engine):
        self.search_engine = search_engine

    def retrieve(
        self,
        query,
        query_language,
        top_k = 5,
        relevant_document_ids = None
    ):
        """Retrieve documents for a single cross-lingual query."""

        results = self.search_engine.search(query, top_k = top_k)

        return CrossLingualQueryResult(
            query = query,
            query_language = query_language,
            retrieved_documents = results,
            relevant_document_ids = relevant_document_ids or []
        )

    def run_batch(self, queries, top_k = 5):
        """
        Retrieve documents for an evaluation set.
        Each item in `queries` is a dict with "query", "query_language", and
        an optional "relevant_document_ids" ground truth list.
        """

        return [
            self.retrieve(
                query = item["query"],
                query_language = item["query_language"],
                top_k = top_k,
                relevant_document_ids = item.get("relevant_document_ids", [])
            )
            for item in queries
        ]

    def summarize(self, results, k = None):
        """
        Aggregate Recall@K, MRR, and average relevant-result similarity
        across a batch produced by run_batch().
        """

        recall_scores = [result.recall_at_k(k) for result in results]
        reciprocal_ranks = [result.reciprocal_rank() for result in results]

        relevant_scores = [
            score
            for result in results
            for score in result.relevant_scores()
        ]

        return {
            "recall@k": sum(recall_scores) / len(recall_scores),
            "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
            "average_relevant_similarity": (
                sum(relevant_scores) / len(relevant_scores)
                if relevant_scores else 0.0
            ),
            "queries_evaluated": len(results)
        }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    parent_dir = str(Path(__file__).resolve().parent.parent)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)

    from embeddings.encoder import MultilingualEncoder
    from indexing.document_store import DocumentStore
    from indexing.vector_index import VectorIndex
    from retrieval.semantic_search import SemanticSearchEngine

    documents = [
        {
            "language": "English",
            "text": "The government introduced a new environmental policy.",
            "metadata": {"concept": "environment_policy"}
        },
        {
            "language": "German",
            "text": "Die Regierung führte eine neue Umweltpolitik ein.",
            "metadata": {"concept": "environment_policy"}
        },
        {
            "language": "Russian",
            "text": "Правительство ввело новую экологическую политику.",
            "metadata": {"concept": "environment_policy"}
        },
        {
            "language": "German",
            "text": "Das Unternehmen entwickelt erneuerbare Energietechnologien.",
            "metadata": {"concept": "renewable_energy"}
        },
        {
            "language": "Russian",
            "text": "Компания разрабатывает технологии возобновляемой энергии.",
            "metadata": {"concept": "renewable_energy"}
        }
    ]

    encoder = MultilingualEncoder()
    document_store = DocumentStore()

    for document in documents:
        document_store.add_document(
            text = document["text"],
            language = document["language"],
            metadata = document["metadata"]
        )

    document_embeddings = encoder.encode(
        [document["text"] for document in documents]
    )

    vector_index = VectorIndex(document_embeddings.shape[1])
    vector_index.add(document_embeddings)

    search_engine = SemanticSearchEngine(
        encoder = encoder,
        vector_index = vector_index,
        doc_store = document_store
    )

    experiment = CrossLingualRetrievalExperiment(search_engine)

    queries = [
        {
            "query": "The government created a new environmental policy.",
            "query_language": "English",
            "relevant_document_ids": [1, 2]
        },
        {
            "query": "clean energy technology",
            "query_language": "English",
            "relevant_document_ids": [3, 4]
        }
    ]

    results = experiment.run_batch(queries, top_k = 5)
    summary = experiment.summarize(results, k = 5)

    print("Cross-Lingual Retrieval — Recall@5, MRR, average relevant similarity")
    print(summary)
