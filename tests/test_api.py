
"""HTTP API tests without loading embedding or LLM models."""

from types import SimpleNamespace

from fastapi.testclient import TestClient

from src.api import create_app


class FakeRetriever:
    def __init__(self):
        self.docs = {}

    def index_document(self, document_id, units):
        if document_id in self.docs:
            raise ValueError("Document already indexed")
        self.docs[document_id] = units

    def query(self, document_id, question, question_id="online"):
        if document_id not in self.docs:
            raise KeyError(document_id)

        return SimpleNamespace(
            action="keep",
            predicted_delta=0.01,
            hybrid_executed=False,
            features={"dense_top1_score": 0.9},
            evidence=[
                {
                    "unit_id": "p1",
                    "text": "Test evidence",
                    "rank": 1,
                }
            ],
            latency_ms=12.0,
        )


def test_health():
    with TestClient(create_app(FakeRetriever())) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"


def test_document_and_query():
    with TestClient(create_app(FakeRetriever())) as client:
        indexed = client.post(
            "/documents",
            json={
                "document_id": "paper1",
                "units": [
                    {
                        "paragraph_id": "p1",
                        "text": "Test evidence",
                        "record_type": "paragraph",
                    }
                ],
            },
        )

        assert indexed.status_code == 201

        result = client.post(
            "/query",
            json={
                "document_id": "paper1",
                "question": "What is the evidence?",
            },
        )

        assert result.status_code == 200
        assert result.json()["retrieval_policy"] == "keep"
        assert result.json()["hybrid_executed"] is False


def test_unknown_document():
    with TestClient(create_app(FakeRetriever())) as client:
        response = client.post(
            "/query",
            json={
                "document_id": "missing",
                "question": "Hello?",
            },
        )

        assert response.status_code == 404


def test_empty_question_rejected():
    with TestClient(create_app(FakeRetriever())) as client:
        response = client.post(
            "/query",
            json={
                "document_id": "paper1",
                "question": "   ",
            },
        )

        assert response.status_code == 422
