
"""Unit tests for conditional online retrieval."""

import numpy as np
import pytest

from src.adaptive_retriever import AdaptiveRetriever


class FakeEmbedder:
    def encode(self, texts, **kwargs):
        vectors = []

        for text in texts:
            text = text.lower()

            if "alpha" in text:
                vectors.append([1.0, 0.0])
            elif "beta" in text:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.5, 0.5])

        vectors = np.array(vectors, dtype="float32")
        vectors /= np.linalg.norm(
            vectors, axis=1, keepdims=True
        )
        return vectors


class FakeCrossEncoder:
    def __init__(self):
        self.calls = 0

    def predict(self, pairs, **kwargs):
        self.calls += 1
        return np.array([
            2.0 if "alpha" in passage.lower() else 1.0
            for _, passage in pairs
        ])


class FakeRouter:
    features = (
        "dense_top1_score",
        "dense_top2_margin",
        "dense_top5_mean",
        "dense_rerank_top10_overlap",
        "dense_rank_of_rerank_top1_or_11",
        "baseline_float_fraction",
    )

    def __init__(self, action):
        self.action = action

    def predict(self, features):
        class Decision:
            pass

        result = Decision()
        result.action = self.action
        result.predicted_delta = (
            0.05 if self.action == "hybrid" else 0.01
        )
        return result


def build_retriever(action):
    cross = FakeCrossEncoder()

    retriever = AdaptiveRetriever(
        router=FakeRouter(action),
        embedder=FakeEmbedder(),
        cross_encoder=cross,
    )

    retriever.index_document("paper1", [
        {
            "document_id": "paper1",
            "paragraph_id": "p1",
            "text": "alpha evidence",
            "record_type": "paragraph",
        },
        {
            "document_id": "paper1",
            "paragraph_id": "p2",
            "text": "beta evidence",
            "record_type": "paragraph",
        },
        {
            "document_id": "paper1",
            "paragraph_id": "p3",
            "text": "other evidence",
            "record_type": "float_caption",
        },
    ])

    return retriever, cross


def test_keep_does_not_execute_hybrid():
    retriever, cross = build_retriever("keep")

    result = retriever.query(
        "paper1", "alpha question"
    )

    assert result.action == "keep"
    assert result.hybrid_executed is False
    assert cross.calls == 1
    assert retriever.lexical_indexes == {}


def test_hybrid_executes_second_reranking():
    retriever, cross = build_retriever("hybrid")

    result = retriever.query(
        "paper1", "alpha question"
    )

    assert result.action == "hybrid"
    assert result.hybrid_executed is True
    assert cross.calls == 2
    assert "paper1" in retriever.lexical_indexes


def test_features_extracted_from_baseline():
    retriever, _ = build_retriever("keep")

    result = retriever.query(
        "paper1", "alpha question"
    )

    assert set(result.features) == set(
        retriever.router.features
    )
    assert result.features[
        "dense_top1_score"
    ] == pytest.approx(1.0)


def test_unknown_document_rejected():
    retriever, _ = build_retriever("keep")

    with pytest.raises(KeyError):
        retriever.query("missing", "alpha question")
