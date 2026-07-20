import sys
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from embeddings.encoder import MultilingualEncoder
from sklearn.metrics.pairwise import cosine_similarity

encoder = MultilingualEncoder()

sentences = [
    "The dog is sleeping.",
    "Der Hund schläft.",
    "Собака спит.",
    "The economy is growing."
]

vectors = encoder.encode(sentences)

 # Should print (4, 768) for the default model.
 # 4 sentences and 768-dimensional embeddings.
print(vectors.shape)

similarities = cosine_similarity(vectors)

for i, sentence1 in enumerate(sentences):
    for j, sentence2 in enumerate(sentences):
        if i < j:
            print(sentence1, "<->", sentence2, round(similarities[i][j], 3))