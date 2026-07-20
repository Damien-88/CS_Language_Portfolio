"""
FAISS-backed vector index for semantic retrieval.
This module provides a thin wrapper around FAISS so the rest of the
project does not depend directly on FAISS APIs.
"""

from __future__ import annotations
import faiss
import numpy as np


class VectorIndex:
    """
    Dense vector index using cosine similarity.
    Because embeddings are L2-normalized before insertion,
    cosine similarity is equivalent to the inner product.
    """

    def __init__(self, embedding_dimension):
        self.dimension = embedding_dimension

        # Inner-product index for normalized embeddings.
        self.index = faiss.IndexFlatIP(embedding_dimension)

    def add(self, embeddings: np.ndarray) -> None:
        """ Add embeddings to the index. """

        embeddings = embeddings.astype(np.float32)

        self.index.add(embeddings)

    def search(self, query_embedding, top_k):
        """ Search the vector index. """

        query_embedding = query_embedding.astype(np.float32)

        scores, indices = self.index.search(query_embedding,top_k)

        return scores, indices

    def __len__(self):
        return self.index.ntotal