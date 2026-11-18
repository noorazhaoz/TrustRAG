
"""Online, document-scoped adaptive retrieval for TrustRAG.

Uses the frozen policy; never accesses gold labels or refits models.
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd

from experiments.retrieval.step58_retrieval_repair import BM25, rrf
from src.feature_extractor import extract_runtime_features


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


@dataclass
class RetrievalResult:
    action: str
    predicted_delta: float
    evidence: list
    features: dict
    latency_ms: float
    hybrid_executed: bool


class AdaptiveRetriever:
    def __init__(self, router, embedder=None, cross_encoder=None):
        self.router = router

        if embedder is None:
            from sentence_transformers import SentenceTransformer
            embedder = SentenceTransformer(EMBEDDING_MODEL)

        if cross_encoder is None:
            from sentence_transformers import CrossEncoder
            cross_encoder = CrossEncoder(RERANKER_MODEL)

        self.embedder = embedder
        self.cross_encoder = cross_encoder

        self.documents = {}
        self.vectors = {}

        # Construct BM25 lazily, only when Hybrid is requested.
        self.lexical_indexes = {}

    def index_document(self, document_id, units):
        """Index evidence units from one document."""

        did = str(document_id)
        if did in self.documents:
            raise ValueError(f"Document already indexed: {did}")

        doc = [
            dict(u) for u in units
            if str(u.get("text", "")).strip()
        ]

        if not doc:
            raise ValueError("Document contains no text evidence")

        ids = [str(u["paragraph_id"]) for u in doc]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate evidence unit IDs")

        if any(str(u["document_id"]) != did for u in doc):
            raise ValueError("Evidence unit document_id mismatch")

        vectors = self.embedder.encode(
            [u["text"] for u in doc],
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")

        if vectors.shape[0] != len(doc):
            raise ValueError("Embedding count mismatch")

        self.documents[did] = doc
        self.vectors[did] = vectors

    def remove_document(self, document_id):
        """Remove a document from the in-memory index."""
        did = str(document_id)
        self.documents.pop(did, None)
        self.vectors.pop(did, None)
        self.lexical_indexes.pop(did, None)

    def _rerank(self, question, doc, candidates):
        """Reproduce the Step 58 cross-encoder ordering."""

        order = [int(i) for i in candidates]
        if not order:
            return []

        scores = np.asarray(
            self.cross_encoder.predict(
                [(question, doc[i]["text"]) for i in order],
                show_progress_bar=False,
            )
        ).reshape(-1)

        if len(scores) != len(order):
            raise ValueError("Cross-encoder score count mismatch")

        if not np.isfinite(scores).all():
            raise ValueError("Non-finite cross-encoder score")

        score_map = {
            i: float(s) for i, s in zip(order, scores)
        }

        return sorted(
            order,
            key=lambda i: (-score_map[i], i),
        )[:10]

    @staticmethod
    def _rows(qid, did, method, indices, doc, dense_scores):
        """Step 58-compatible ranking records."""

        return [
            {
                "question_id": qid,
                "document_id": did,
                "method": method,
                "rank": rank,
                "unit_id": str(doc[i]["paragraph_id"]),
                "record_type": doc[i].get("record_type", ""),
                "dense_score": round(float(dense_scores[i]), 8),
            }
            for rank, i in enumerate(indices, 1)
        ]

    def query(self, document_id, question, question_id="online"):
        start = perf_counter()

        did = str(document_id)
        qid = str(question_id)

        if did not in self.documents:
            raise KeyError(f"Unknown document: {did}")
        if not str(question).strip():
            raise ValueError("Question cannot be empty")

        question = str(question)
        doc = self.documents[did]

        qvec = self.embedder.encode(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")[0]

        dense_scores = self.vectors[did] @ qvec

        # Match Step 58 stable dense ranking.
        dense_order = np.argsort(
            -dense_scores, kind="stable"
        )

        dense_pool = dense_order[:30]

        # Baseline reranking ALWAYS happens first.
        baseline_top10 = self._rerank(
            question, doc, dense_pool
        )

        raw_top10 = [
            int(i) for i in dense_order[:10]
        ]

        rows = (
            self._rows(
                qid, did, "dense30_raw",
                raw_top10, doc, dense_scores
            )
            + self._rows(
                qid, did, "dense30_rerank",
                baseline_top10, doc, dense_scores
            )
        )

        feature_frame = extract_runtime_features(
            pd.DataFrame(rows)
        )

        if len(feature_frame) != 1:
            raise RuntimeError("Cannot extract baseline features")

        features = {
            name: feature_frame.iloc[0][name]
            for name in self.router.features
        }

        decision = self.router.predict(features)

        if decision.action not in ("keep", "hybrid"):
            raise ValueError("Unexpected policy action")

        selected = baseline_top10

        if decision.action == "hybrid":
            # Nothing lexical is computed on the Keep path.
            if did not in self.lexical_indexes:
                self.lexical_indexes[did] = BM25(
                    [u["text"] for u in doc]
                )

            bm_scores = self.lexical_indexes[did].score(
                question
            )

            bm_order = np.argsort(
                -bm_scores, kind="stable"
            )

            hybrid_order = rrf(
                dense_order,
                bm_order,
                len(doc),
                constant=60,
            )

            selected = self._rerank(
                question,
                doc,
                hybrid_order[:30],
            )

        evidence = [
            {
                "unit_id": str(doc[i]["paragraph_id"]),
                "record_type": doc[i].get("record_type", ""),
                "text": doc[i]["text"],
                "rank": rank,
            }
            for rank, i in enumerate(selected, 1)
        ]

        return RetrievalResult(
            action=decision.action,
            predicted_delta=decision.predicted_delta,
            evidence=evidence,
            features=features,
            latency_ms=(perf_counter() - start) * 1000,
            hybrid_executed=(decision.action == "hybrid"),
        )
