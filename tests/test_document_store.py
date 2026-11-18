
import numpy as np
import pytest

from src.document_store import DocumentStore


class FakeRetriever:
    def __init__(self):
        self.documents = {}
        self.vectors = {}
        self.lexical_indexes = {}


def test_save_and_restore(tmp_path):
    store = DocumentStore(tmp_path)
    original = FakeRetriever()

    original.documents["paper1"] = [
        {
            "document_id": "paper1",
            "paragraph_id": "page1_chunk1",
            "record_type": "paragraph",
            "text": "TrustRAG uses adaptive retrieval.",
        }
    ]

    original.vectors["paper1"] = np.array(
        [[0.6, 0.8]],
        dtype=np.float32,
    )

    store.save_document(original, "paper1")

    restored = FakeRetriever()
    count = store.restore_all(restored)

    assert count == 1
    assert restored.documents == original.documents
    np.testing.assert_array_equal(
        restored.vectors["paper1"],
        original.vectors["paper1"],
    )
    assert restored.lexical_indexes == {}


def test_unknown_document_rejected(tmp_path):
    store = DocumentStore(tmp_path)

    with pytest.raises(ValueError, match="Unknown document"):
        store.save_document(FakeRetriever(), "missing")
