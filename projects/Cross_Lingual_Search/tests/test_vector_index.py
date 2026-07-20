import sys
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from indexing.vector_index import VectorIndex

import numpy as np

vectors = np.random.rand(10, 768).astype(np.float32)

vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

index = VectorIndex(768)

index.add(vectors)

scores, ids = index.search(vectors[:1], top_k=3)

print(scores)
print(ids)