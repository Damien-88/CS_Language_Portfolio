"""
Morphology Effects Experiment.

This experiment addresses Research Question 2:

How does morphology affect semantic retrieval?

Languages encode meaning differently through morphology. German and Russian
both vary surface word forms while the underlying meaning stays stable:

German:
   Haus, Häuser, Hausverwaltung

Russian:
   дом, дома, домовой

This experiment measures whether the multilingual embedding space keeps
morphological variants of the same lexical item close together, and whether
retrieval for a base-form query still finds documents written using an
inflected or compounded variant.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from retrieval.semantic_search import SearchResult


@dataclass
class MorphologyVariantResult:
    """Similarity between a base form and one of its morphological variants."""

    base_form: str
    variant: str
    language: str
    similarity: float


@dataclass
class MorphologyRetrievalResult:
    """Retrieval outcome for a query written in a base morphological form."""

    query: str
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


class MorphologyEffectsExperiment:
    """
    Measures how morphological variation (inflection, compounding) affects
    embedding similarity and retrieval quality.
    """

    def __init__(self, encoder, search_engine = None):
        """
        search_engine is optional; it is only required for retrieval checks,
        not for comparing raw morphological variant similarity.
        """

        self.encoder = encoder
        self.search_engine = search_engine

    def compare_variants(self, base_form, variants, language):
        """
        Encode a base word/phrase and its morphological variants, and return
        the cosine similarity between the base form and each variant.
        Embeddings are L2-normalized, so the dot product is cosine similarity.
        """

        texts = [base_form] + list(variants)
        embeddings = self.encoder.encode(texts)

        base_vector = embeddings[0]

        return [
            MorphologyVariantResult(
                base_form = base_form,
                variant = variant,
                language = language,
                similarity = float(np.dot(base_vector, vector))
            )
            for variant, vector in zip(variants, embeddings[1:])
        ]

    def retrieve_with_base_form(self, query, top_k = 5, relevant_document_ids = None):
        """
        Search using the base morphological form of a query, to test whether
        inflected or compounded documents are still retrieved.
        """

        if self.search_engine is None:
            raise ValueError("A search_engine is required for retrieval checks.")

        results = self.search_engine.search(query, top_k = top_k)

        return MorphologyRetrievalResult(
            query = query,
            retrieved_documents = results,
            relevant_document_ids = relevant_document_ids or []
        )

    def summarize_variants(self, variant_results):
        """Average similarity per language across all compared variants."""

        by_language = {}
        for result in variant_results:
            by_language.setdefault(result.language, []).append(result.similarity)

        return {
            language: sum(values) / len(values)
            for language, values in by_language.items()
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

    encoder = MultilingualEncoder()

    documents = [
        {
            "language": "German",
            "text": "Die Hausverwaltung bearbeitet die Anfrage.",
            "metadata": {"concept": "property_management"}
        },
        {
            "language": "German",
            "text": "Der Hund schläft.",
            "metadata": {"concept": "animal_sleeping"}
        }
    ]

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

    experiment = MorphologyEffectsExperiment(encoder, search_engine)

    german_variants = experiment.compare_variants(
        base_form = "Haus",
        variants = ["Häuser", "Hausverwaltung"],
        language = "German"
    )

    russian_variants = experiment.compare_variants(
        base_form = "дом",
        variants = ["дома", "домовой"],
        language = "Russian"
    )

    variant_summary = experiment.summarize_variants(german_variants + russian_variants)

    retrieval_result = experiment.retrieve_with_base_form(
        query = "Haus",
        top_k = 2,
        relevant_document_ids = [0]
    )

    print("Morphology Effects — variant similarity to base form")
    for result in german_variants + russian_variants:
        print(f"  {result.language}: {result.base_form} vs {result.variant} -> {result.similarity:.4f}")

    print("\nAverage similarity by language:")
    print(variant_summary)

    print("\nRetrieval using base form 'Haus' against a compounded document:")
    print(f"  Recall@2: {retrieval_result.recall_at_k(2)}")
