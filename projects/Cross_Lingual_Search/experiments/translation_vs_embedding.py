"""
Experiment comparing translation-based retrieval with direct cross-lingual 
embedding retrieval.

This experiment addresses Research Question 2:

Does direct semantic alignment outperform translation-based retrieval?

The project spans English, German, and Russian, so `target_language` is not
limited to German: any query/target combination among these three languages
can be evaluated.

Two retrieval approaches are compared:

1. Translation-based retrieval
   Query -> Translation -> Search in the target language

2. Embedding-based retrieval
   Query -> Multilingual embedding -> Search in the target language

Each result records the relevant document IDs for its query so Recall@K and
MRR can be computed per query and aggregated across an evaluation set to
compare the two approaches.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from retrieval.semantic_search import SearchResult


@dataclass
class ExperimentResult:
    """Stores the result of one retrieval approach for a single query."""

    approach: str
    query: str
    translated_query: str
    target_language: str
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


class TranslationVsEmbeddingExperiment:
    """Compares translation-based and direct embedding-based retrieval."""

    def __init__(self, search_engine, translator = None):
        """
        Initialize the experiment.
        The translator is optional so that the experiment can still support 
        direct embedding retrieval when a translation system has not yet been 
        configured.
        """

        self.search_engine = search_engine
        self.translator = translator

    def translation_retrieval(
        self,
        query,
        target_language,
        top_k = 5,
        relevant_document_ids = None
    ):
        """
        Translate the query before performing semantic retrieval.
        target_language is any of the project's supported languages
        (English, German, Russian). The translator must provide a
        translate(text, target_language) method.
        """

        if self.translator is None:
            raise ValueError(
                "A translator is required for translation-based retrieval."
            )

        if not hasattr(self.translator, "translate"):
            raise TypeError(
                "translator must implement a translate(text, target_language) method."
            )

        translated_query = self.translator.translate(
            query,
            target_language = target_language
        )

        results = self.search_engine.search(translated_query, top_k = top_k)

        return ExperimentResult(
            approach = "translation",
            query = query,
            translated_query = translated_query,
            target_language = target_language,
            retrieved_documents = results,
            relevant_document_ids = relevant_document_ids or []
        )

    def embedding_retrieval(
        self,
        query,
        target_language,
        top_k = 5,
        relevant_document_ids = None
    ):
        """
        Perform direct cross-lingual semantic retrieval.
        target_language is recorded for reporting only; the original query is
        passed directly to the multilingual embedding model without
        translation, so it can match documents in English, German, or
        Russian.
        """

        results = self.search_engine.search(query, top_k = top_k)

        return ExperimentResult(
            approach = "embedding",
            query = query,
            translated_query = query,
            target_language = target_language,
            retrieved_documents = results,
            relevant_document_ids = relevant_document_ids or []
        )

    def compare(
        self,
        query,
        target_language,
        top_k = 5,
        relevant_document_ids = None
    ):
        """
        Run both retrieval approaches for the same query against a chosen
        target language (English, German, or Russian).
        """

        translation_result = self.translation_retrieval(
            query = query,
            target_language = target_language,
            top_k = top_k,
            relevant_document_ids = relevant_document_ids
        )

        embedding_result = self.embedding_retrieval(
            query = query,
            target_language = target_language,
            top_k = top_k,
            relevant_document_ids = relevant_document_ids
        )

        return {"translation": translation_result, "embedding": embedding_result}

    def run_batch(self, queries, target_language = None, top_k = 5):
        """
        Run both retrieval approaches over an evaluation set spanning any mix
        of English, German, and Russian targets.
        Each item in `queries` is a dict with "query", an optional
        "target_language" (falls back to the `target_language` argument when
        omitted), and an optional "relevant_document_ids" ground truth list.
        """

        results = []

        for item in queries:
            query_target_language = item.get("target_language", target_language)

            if query_target_language is None:
                raise ValueError(
                    "target_language must be provided either per query or as "
                    "a default for the whole batch."
                )

            results.append(
                self.compare(
                    query = item["query"],
                    target_language = query_target_language,
                    top_k = top_k,
                    relevant_document_ids = item.get("relevant_document_ids", [])
                )
            )

        return results

    def summarize(self, batch_results, k = None):
        """
        Aggregate Recall@K and MRR per approach across a batch produced by
        run_batch(), so the two approaches can be compared as a whole rather
        than query by query.
        """

        summary = {}

        for approach in ("translation", "embedding"):
            results = [item[approach] for item in batch_results]

            recall_scores = [result.recall_at_k(k) for result in results]
            reciprocal_ranks = [result.reciprocal_rank() for result in results]

            summary[approach] = {
                "recall@k": sum(recall_scores) / len(recall_scores),
                "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
                "queries_evaluated": len(results)
            }

        return summary


class DictionaryTranslator:
    """
    Lookup-based stand-in for a real machine translation model.
    Only covers the queries used in the demo run below.
    """

    def __init__(self, translations):
        self.translations = translations

    def translate(self, text, target_language):
        return self.translations[(text, target_language)]


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

    translator = DictionaryTranslator({
        (
            "The government created a new environmental policy.",
            "German"
        ): "Die Regierung führte eine neue Umweltpolitik ein.",
        (
            "clean energy technology",
            "Russian"
        ): "технологии возобновляемой энергии"
    })

    experiment = TranslationVsEmbeddingExperiment(
        search_engine = search_engine,
        translator = translator
    )

    queries = [
        {
            "query": "The government created a new environmental policy.",
            "target_language": "German",
            "relevant_document_ids": [1]
        },
        {
            "query": "clean energy technology",
            "target_language": "Russian",
            "relevant_document_ids": [4]
        }
    ]

    batch_results = experiment.run_batch(queries, top_k = 5)
    summary = experiment.summarize(batch_results, k = 5)

    print("Translation vs. Embedding Retrieval — Recall@5 and MRR")
    for approach, scores in summary.items():
        print(f"{approach}: {scores}")
