"""
Experiment 3: German Compound Decomposition Impact.

This experiment addresses Research Question 3:

Does linguistic preprocessing improve retrieval?

Integrates the German Compound Decomposition project to compare retrieval
using a raw German compound query against the same query with its compound
nouns decomposed into separate words.

Comparison:
   Without preprocessing: "Hausverwaltung"
   With preprocessing:    "Haus Verwaltung"

Measure:
- retrieval improvement (Recall@K, MRR)
- similarity changes (top-result score)

The decomposer is optional and injected by the caller so this module does not
hard-depend on the German_Compound_Decomposition project. It only needs an
object exposing decompose(word) -> {"components": [{"text": ...}, ...]}.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from retrieval.semantic_search import SearchResult


@dataclass
class DecompositionRetrievalResult:
    """Retrieval outcome for one query variant (original or decomposed)."""

    variant: str
    query_text: str
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

    def top_score(self) -> float:
        """Similarity score of the top-ranked retrieved document, or 0.0."""

        return float(self.retrieved_documents[0].score) if self.retrieved_documents else 0.0


class GermanCompoundDecompositionExperiment:
    """
    Compares retrieval quality for German compound queries before and after
    morphological decomposition.
    """

    def __init__(self, search_engine, decomposer = None):
        """
        decomposer is optional so this experiment can still run in
        "without preprocessing" mode when no decomposer has been configured.
        """

        self.search_engine = search_engine
        self.decomposer = decomposer

    def decompose_text(self, text):
        """
        Decompose each word in `text` and join components with spaces.
        Falls back to the original word when it cannot be decomposed or no
        decomposer is configured.
        """

        if self.decomposer is None:
            return text

        decomposed_words = []

        for word in text.split():
            result = self.decomposer.decompose(word)
            components = [component["text"] for component in result["components"]]
            decomposed_words.append(" ".join(components) if components else word)

        return " ".join(decomposed_words)

    def compare(self, query, top_k = 5, relevant_document_ids = None):
        """Retrieve using the original query and its decomposed form."""

        original_result = DecompositionRetrievalResult(
            variant = "original",
            query_text = query,
            retrieved_documents = self.search_engine.search(query, top_k = top_k),
            relevant_document_ids = relevant_document_ids or []
        )

        decomposed_query = self.decompose_text(query)

        decomposed_result = DecompositionRetrievalResult(
            variant = "decomposed",
            query_text = decomposed_query,
            retrieved_documents = self.search_engine.search(decomposed_query, top_k = top_k),
            relevant_document_ids = relevant_document_ids or []
        )

        return {"original": original_result, "decomposed": decomposed_result}

    def run_batch(self, queries, top_k = 5):
        """
        Run both query variants over an evaluation set.
        Each item in `queries` is a dict with "query" and, optionally,
        "relevant_document_ids" used as ground truth for scoring.
        """

        return [
            self.compare(
                query = item["query"],
                top_k = top_k,
                relevant_document_ids = item.get("relevant_document_ids", [])
            )
            for item in queries
        ]

    def summarize(self, batch_results, k = None):
        """
        Aggregate Recall@K, MRR, and top-result similarity per variant, so
        the effect of compound decomposition on retrieval can be measured.
        """

        summary = {}

        for variant in ("original", "decomposed"):
            results = [item[variant] for item in batch_results]

            recall_scores = [result.recall_at_k(k) for result in results]
            reciprocal_ranks = [result.reciprocal_rank() for result in results]
            top_scores = [result.top_score() for result in results]

            summary[variant] = {
                "recall@k": sum(recall_scores) / len(recall_scores),
                "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
                "average_top_similarity": sum(top_scores) / len(top_scores),
                "queries_evaluated": len(results)
            }

        return summary


if __name__ == "__main__":
    import sys
    import tempfile
    from pathlib import Path
    parent_dir = str(Path(__file__).resolve().parent.parent)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)

    from embeddings.encoder import MultilingualEncoder
    from indexing.document_store import DocumentStore
    from indexing.vector_index import VectorIndex
    from retrieval.semantic_search import SemanticSearchEngine

    # The decomposer lives in a sibling portfolio project and is optional.
    decomposition_project = str(
        Path(__file__).resolve().parents[2] / "German_Compound_Decomposition"
    )
    if decomposition_project not in sys.path:
        sys.path.append(decomposition_project)

    try:
        from german_compound_decomposer import GermanCompoundDecomposer

        # Minimal lemma lexicon so the demo compound actually splits.
        with tempfile.NamedTemporaryFile(
            mode = "w", suffix = ".txt", delete = False, encoding = "utf-8"
        ) as lemma_file:
            lemma_file.write("haus\nverwaltung\n")
            lemma_path = lemma_file.name

        decomposer = GermanCompoundDecomposer(lemma_path = lemma_path, use_spacy = False)
    except ImportError:
        print("GermanCompoundDecomposer not available; running without preprocessing.")
        decomposer = None

    documents = [
        {
            "language": "German",
            "text": "Die Hausverwaltung bearbeitet die Anfrage.",
            "metadata": {"concept": "property_management"}
        },
        {
            "language": "German",
            "text": "Die Verwaltung des Hauses bearbeitet die Anfrage.",
            "metadata": {"concept": "property_management"}
        },
        {
            "language": "German",
            "text": "Das Unternehmen entwickelt erneuerbare Energietechnologien.",
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

    experiment = GermanCompoundDecompositionExperiment(
        search_engine = search_engine,
        decomposer = decomposer
    )

    queries = [
        {
            "query": "Hausverwaltung",
            "relevant_document_ids": [0, 1]
        }
    ]

    batch_results = experiment.run_batch(queries, top_k = 3)
    summary = experiment.summarize(batch_results, k = 3)

    print("German Compound Decomposition — Recall@3, MRR, top-result similarity")
    for variant, scores in summary.items():
        print(f"{variant}: {scores}")
