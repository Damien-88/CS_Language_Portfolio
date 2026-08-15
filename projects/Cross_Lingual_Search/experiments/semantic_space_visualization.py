"""
Experiment 4: Semantic Space Visualization.

This experiment provides a visual diagnostic for the project's central
question:

Do multilingual embeddings organize sentences by semantic meaning rather
than by language identity?

Multilingual embeddings are reduced to 2D using PCA so that cross-lingual
semantic alignment can be inspected visually and measured with distance
metrics between semantically equivalent items across languages.

Expected behavior:
   Languages should cluster by meaning rather than by alphabet/language.
"""

from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from sklearn.decomposition import PCA


@dataclass
class ReducedDocument:
    """A document projected into 2D semantic space."""

    text: str
    language: str
    concept: str
    x: float
    y: float


class SemanticSpaceVisualization:
    """
    Projects multilingual embeddings into 2D space and measures whether
    semantically equivalent documents across languages remain close
    together.
    """

    def __init__(self, encoder):
        self.encoder = encoder

    def reduce(self, documents):
        """
        Project documents into 2D using PCA.
        `documents` is a list of dicts with "text", "language", "concept".
        """

        embeddings = self.encoder.encode(
            [document["text"] for document in documents]
        )

        coordinates = PCA(n_components = 2).fit_transform(embeddings)

        return [
            ReducedDocument(
                text = document["text"],
                language = document["language"],
                concept = document["concept"],
                x = float(coordinates[index, 0]),
                y = float(coordinates[index, 1])
            )
            for index, document in enumerate(documents)
        ]

    def within_concept_distances(self, reduced_documents):
        """
        Pairwise Euclidean distance between same-concept documents across
        languages. Smaller values indicate stronger cross-lingual semantic
        alignment.
        """

        by_concept = {}
        for document in reduced_documents:
            by_concept.setdefault(document.concept, []).append(document)

        distances = []
        for concept, group in by_concept.items():
            for first, second in combinations(group, 2):
                distance = float(np.hypot(first.x - second.x, first.y - second.y))
                distances.append(
                    {
                        "concept": concept,
                        "language_pair": f"{first.language} → {second.language}",
                        "distance": distance
                    }
                )

        return distances

    def summarize(self, reduced_documents):
        """
        Aggregate the average within-concept distance overall and per
        language pair, to check whether documents cluster by meaning rather
        than by language.
        """

        distances = self.within_concept_distances(reduced_documents)

        if not distances:
            return {"average_distance": 0.0, "by_language_pair": {}}

        average_distance = sum(item["distance"] for item in distances) / len(distances)

        grouped = {}
        for item in distances:
            grouped.setdefault(item["language_pair"], []).append(item["distance"])

        by_language_pair = {
            pair: sum(values) / len(values)
            for pair, values in grouped.items()
        }

        return {
            "average_distance": average_distance,
            "by_language_pair": by_language_pair
        }

    def plot(self, reduced_documents, output_path):
        """Save a scatter plot of the reduced semantic space to output_path."""

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        markers = {"English": "o", "German": "s", "Russian": "^"}

        plt.figure(figsize = (10, 7))

        for language in {document.language for document in reduced_documents}:
            language_documents = [
                document
                for document in reduced_documents
                if document.language == language
            ]

            plt.scatter(
                [document.x for document in language_documents],
                [document.y for document in language_documents],
                marker = markers.get(language, "o"),
                s = 100,
                label = language
            )

            for document in language_documents:
                plt.annotate(
                    document.concept,
                    (document.x, document.y),
                    textcoords = "offset points",
                    xytext = (0, 10)
                )

        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title("Cross-Lingual Semantic Space")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()


if __name__ == "__main__":
    import sys
    from pathlib import Path
    parent_dir = str(Path(__file__).resolve().parent.parent)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)

    from embeddings.encoder import MultilingualEncoder

    documents = [
        {
            "concept": "environmental_policy",
            "language": "English",
            "text": "The government announced a new environmental policy."
        },
        {
            "concept": "environmental_policy",
            "language": "German",
            "text": "Die Regierung kündigte eine neue Umweltpolitik an."
        },
        {
            "concept": "environmental_policy",
            "language": "Russian",
            "text": "Правительство объявило новую экологическую политику."
        },
        {
            "concept": "animal_sleeping",
            "language": "English",
            "text": "The dog is sleeping."
        },
        {
            "concept": "animal_sleeping",
            "language": "German",
            "text": "Der Hund schläft."
        },
        {
            "concept": "animal_sleeping",
            "language": "Russian",
            "text": "Собака спит."
        },
        {
            "concept": "renewable_energy",
            "language": "English",
            "text": "The company develops renewable energy technology."
        },
        {
            "concept": "renewable_energy",
            "language": "German",
            "text": "Das Unternehmen entwickelt erneuerbare Energietechnologien."
        },
        {
            "concept": "renewable_energy",
            "language": "Russian",
            "text": "Компания разрабатывает технологии возобновляемой энергии."
        }
    ]

    encoder = MultilingualEncoder()
    visualization = SemanticSpaceVisualization(encoder)

    reduced_documents = visualization.reduce(documents)
    summary = visualization.summarize(reduced_documents)

    output_path = str(Path(__file__).resolve().parent / "semantic_space.png")
    visualization.plot(reduced_documents, output_path)

    print("Semantic Space Visualization — cross-lingual distance summary")
    print(f"Average distance: {summary['average_distance']:.4f}")
    print("By language pair:")
    for pair, distance in summary["by_language_pair"].items():
        print(f"  {pair}: {distance:.4f}")
    print(f"Plot saved to: {output_path}")
