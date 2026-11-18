
from fastapi.testclient import TestClient

from src.api import create_app


class FakeRetriever:
    def __init__(self):
        self.documents = {}
        self.vectors = {}
        self.lexical_indexes = {}

    def index_document(self, document_id, units):
        self.documents[document_id] = units
        self.vectors[document_id] = [[1.0, 0.0]]

    def remove_document(self, document_id):
        self.documents.pop(document_id, None)
        self.vectors.pop(document_id, None)
        self.lexical_indexes.pop(document_id, None)


class FailingStore:
    def save_document(self, retriever, document_id):
        raise OSError("Simulated disk failure")


def test_document_rollback_on_storage_failure():
    engine = FakeRetriever()

    # See note below: the application must have persistence enabled.
    app = create_app(
        retriever=engine,
        document_store=FailingStore(),
    )

    with TestClient(app) as client:
        response = client.post(
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

        assert response.status_code == 500
        assert "paper1" not in engine.documents
        assert "paper1" not in engine.vectors
