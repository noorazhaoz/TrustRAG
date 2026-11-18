
"""Tests for corrupted persistent document indexes."""

import json
from zipfile import BadZipFile

import numpy as np
import pytest

from src.document_store import DocumentStore


class FakeRetriever:
    def __init__(self):
        self.documents = {}
        self.vectors = {}
        self.lexical_indexes = {}


def test_corrupted_npz_rejected(tmp_path):
    """An invalid index file must not be silently skipped."""
    store = DocumentStore(tmp_path)

    # Create a fake, corrupted index file.
    corrupted_file = tmp_path / "corrupted.npz"
    corrupted_file.write_bytes(b"not a valid npz file")

    retriever = FakeRetriever()

    with pytest.raises(
        (ValueError, BadZipFile, OSError, EOFError)
    ):
        store.restore_all(retriever)

    assert retriever.documents == {}
    assert retriever.vectors == {}


def test_invalid_index_version_rejected(tmp_path):
    """Unsupported index versions must fail explicitly."""
    store = DocumentStore(tmp_path)

    metadata = {
        "version": 999,
        "document_id": "paper1",
        "units": [
            {
                "document_id": "paper1",
                "paragraph_id": "p1",
                "text": "Test evidence",
            }
        ],
    }

    metadata_bytes = json.dumps(metadata).encode("utf-8")

    np.savez_compressed(
        store._path("paper1"),
        metadata=np.frombuffer(
            metadata_bytes,
            dtype=np.uint8,
        ),
        vectors=np.array(
            [[0.6, 0.8]],
            dtype=np.float32,
        ),
    )

    with pytest.raises(
        ValueError,
        match="Unsupported store version",
    ):
        store.restore_all(FakeRetriever())
