
import faiss
from sentence_transformers import SentenceTransformer


class Retriever:
    def __init__(self, documents):
        self.documents = documents
        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        embeddings = self.model.encode(
            documents,
            normalize_embeddings=True
        ).astype("float32")

        self.index = faiss.IndexFlatIP(
            embeddings.shape[1]
        )
        self.index.add(embeddings)

    def search(self, query, k=2):
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        ).astype("float32")

        scores, indices = self.index.search(
            query_embedding,
            min(k, len(self.documents))
        )

        return [
            {
                "id": int(i),
                "text": self.documents[i],
                "score": float(score)
            }
            for i, score in zip(indices[0], scores[0])
        ]

