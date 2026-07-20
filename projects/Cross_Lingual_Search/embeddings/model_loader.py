"""
Model loading utilities for multilingual semantic embeddings.
This module separates model management from encoding logic.
The retrieval pipeline should not care which transformer model is being used.
"""

import os
from functools import lru_cache

# Force Transformers to skip TensorFlow integration in this project.
# This avoids import-time failures on Python 3.13 environments where
# TF symbols expected by sentence-transformers are not exposed.
os.environ.setdefault("USE_TF", "0")

from sentence_transformers import SentenceTransformer

@lru_cache(maxsize=2)
def load_embedding_model(
    model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    ):
    """
    Load and cache a multilingual sentence embedding model.
    The cache prevents repeatedly downloading/loading the same transformer model
    during experiments.
    """

    print(f"Loading embedding model: {model_name}")

    model = SentenceTransformer(model_name)

    return model


# paraphrase-multilingual-mpnet-base-v2 supports 50+ Languages, including 
# English, German, and Russian. 
# It creates vectors where:
# English:
# "The dog is sleeping."
#
# German:
# "Der Hund schläft."

# Russian:
# "Собака спит."

# should be close in vector space.