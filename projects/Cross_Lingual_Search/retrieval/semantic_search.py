"""
Crosslingual semantic retrieval pipeline.
This module combines multilingual embeddings, FAISS vector search and document
metadata lookup. The goal is language-independent retireval.
"""

from __future__ import annotations
from dataclasses import dataclass

import sys
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from embeddings.encoder import MultilingualEncoder
from indexing.document_store import DocumentStore, Document
from indexing.vector_index import VectorIndex

import numpy as np

@dataclass
class SearchResult:
    """Represents a single retrieved document"""
    document: Document
    score: float

class SemanticSearchEngine:
    """High-level interface for cross-lingual semantic search."""

    def __init__(
            self, 
            encoder: MultilingualEncoder, 
            vector_index: VectorIndex, 
            doc_store: DocumentStore
        ):
        """Initializes the search engine with the necessary components."""

        self.encoder = encoder
        self.vector_index = vector_index
        self.doc_store = doc_store


    def search(self, query, top_k = 5):
        """Searches documents using semantic similarity."""

        # Convert query into semantic vector.
        query_embedding = self.encoder.encode([query])

        # Search FAISS index for top_k similar documents.
        scores, indices = self.vector_index.search(query_embedding, top_k)

        results = []

        for score, index in zip(scores[0], indices[0]):
            # FAISS returns -1 for invalid indices.
            if index == -1:
                continue

            # Retrieve the document from the document store.
            document = self.doc_store.get(int(index))

            results.append(SearchResult(document = document, score = score))

        return results