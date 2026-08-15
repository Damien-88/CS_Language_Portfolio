import sys
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from indexing.document_store import Document
from retrieval.semantic_search import SearchResult
from experiments.translation_vs_embedding import TranslationVsEmbeddingExperiment


class FakeTranslator:
    """Returns a fixed translation per target language."""

    def translate(self, text, target_language = "German"):
        translations = {
            "German": "Die Regierung führte eine neue Umweltpolitik ein.",
            "Russian": "Компания разрабатывает технологии возобновляемой энергии."
        }
        return translations[target_language]


class FakeSearchEngine:
    """Returns the same ranked documents regardless of query text."""

    def __init__(self, documents):
        self.documents = documents

    def search(self, query, top_k = 5):
        return [
            SearchResult(document = document, score = 1.0 - index * 0.1)
            for index, document in enumerate(self.documents[:top_k])
        ]


documents = [
    Document(
        0,
        "Die Regierung führte eine neue Umweltpolitik ein.",
        "de",
        {"concept": "environment_policy"}
    ),
    Document(
        1,
        "Компания разрабатывает технологии возобновляемой энергии.",
        "ru",
        {"concept": "renewable_energy"}
    )
]

experiment = TranslationVsEmbeddingExperiment(
    search_engine = FakeSearchEngine(documents),
    translator = FakeTranslator()
)

queries = [
    {
        "query": "The government created a new environmental policy.",
        "target_language": "German",
        "relevant_document_ids": [0]
    },
    {
        "query": "clean energy technology",
        "target_language": "Russian",
        "relevant_document_ids": [1]
    }
]

batch_results = experiment.run_batch(queries, top_k = 2)

summary = experiment.summarize(batch_results)

print(summary) # Both approaches should score perfectly against this fake engine.
