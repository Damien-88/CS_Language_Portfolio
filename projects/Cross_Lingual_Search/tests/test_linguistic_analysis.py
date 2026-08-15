import sys
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from indexing.document_store import Document
from retrieval.semantic_search import SearchResult
from analysis.linguistic_analysis import LinguisticAnalyzer

analyzer = LinguisticAnalyzer()

# Compound representation: single compound word vs. multi-word expression.
compound_result = analyzer.analyze(
    query = "property management",
    retrieved_result = SearchResult(
        document = Document(0, "Hausverwaltung", "de"),
        score = 0.9
    ),
    expected_text = "Verwaltung des Hauses"
)
print("Compound:", [error.category.value for error in compound_result])

# Word order: same words, different sequence.
word_order_result = analyzer.analyze(
    query = "book read yesterday",
    retrieved_result = SearchResult(
        document = Document(1, "the book that read yesterday i", "en"),
        score = 0.9
    ),
    expected_text = "the book that i read yesterday"
)
print("Word order:", [error.category.value for error in word_order_result])

# Morphological variation: same length, different form (case difference).
morphology_result = analyzer.analyze(
    query = "house",
    retrieved_result = SearchResult(
        document = Document(2, "Haus", "de"),
        score = 0.9
    ),
    expected_text = "haus"
)
print("Morphology:", [error.category.value for error in morphology_result])

# Language mismatch: expected Russian script, retrieved document is German.
language_mismatch_result = analyzer.analyze(
    query = "environmental policy",
    retrieved_result = SearchResult(
        document = Document(3, "Der Hund schläft.", "German"),
        score = 0.9
    ),
    expected_text = "Правительство ввело новую экологическую политику."
)
print("Language mismatch:", [error.category.value for error in language_mismatch_result])
