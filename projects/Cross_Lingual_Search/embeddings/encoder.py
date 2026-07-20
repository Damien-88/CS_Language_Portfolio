"""
Embedding generation layer
Responsible for converting multilingual text into dense sematic vectors.
"""

from typing import Iterable, List
import numpy as np
from .model_loader import load_embedding_model

class MultilingualEncoder:
    """
    Wrapper around SentenceTransformer models.
    Provides: batch encoding, normalization, and consistent numpy output.
    """

    def __init__(
        self, 
        model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    ):
        self.model = load_embedding_model(model_name)

    def encode(self, texts: Iterable[str], normalize = True):
        """ Convert text into semantic vectors."""

        texts = list(texts)

        embeddings = self.model.encode(
            texts, 
            batch_size = 32,
            show_progress_bar = True,
            convert_to_numpy = True            
        )

        if normalize:
            embeddings = self.normalize(embeddings)

        return embeddings
    
    def normalize(self, vectors):
        """
        L2 normalize embeddings. After normalization cosine similarity is 
        equivalent to dot product.
        """

        norms = np.linalg.norm(vectors, axis = 1, keepdims = True)

        return vectors / norms