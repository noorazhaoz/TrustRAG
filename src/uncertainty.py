
import numpy as np
from itertools import combinations
from sentence_transformers import SentenceTransformer


class ConsistencyEstimator:
    def __init__(self):
        self.encoder = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    def score(self, answers):
        if len(answers) < 2:
            raise ValueError("At least two answers required")

        embeddings = self.encoder.encode(
            answers,
            normalize_embeddings=True
        )

        similarities = [
            float(np.dot(embeddings[i], embeddings[j]))
            for i, j in combinations(range(len(answers)), 2)
        ]

        return {
            "mean_similarity": float(np.mean(similarities)),
            "std_similarity": float(np.std(similarities)),
        }
