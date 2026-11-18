
"""Persistent local document storage for TrustRAG."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from src.adaptive_retriever import (
    EMBEDDING_MODEL,
    RERANKER_MODEL,
)


class DocumentStore:
    def __init__(self, root="data/document_store"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, document_id):
        # Avoid unsafe filenames and path traversal.
        key = hashlib.sha256(
            str(document_id).encode("utf-8")
        ).hexdigest()
        return self.root / f"{key}.npz"

    def save_document(self, retriever, document_id):
        """Save an already-indexed document and its embeddings."""
        did = str(document_id)

        if did not in retriever.documents:
            raise ValueError(f"Unknown document: {did}")

        documents = retriever.documents[did]
        vectors = np.asarray(retriever.vectors[did])

        if (
            vectors.ndim != 2
            or vectors.shape[0] != len(documents)
            or vectors.dtype != np.float32
            or not np.isfinite(vectors).all()
        ):
            raise ValueError("Invalid embedding matrix")

        metadata = {
            "version": 1,
            "document_id": did,
            "embedding_model": EMBEDDING_MODEL,
            "reranker_model": RERANKER_MODEL,
            "units": documents,
        }

        metadata_bytes = json.dumps(
            metadata,
            ensure_ascii=False,
        ).encode("utf-8")

        temp_path = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                suffix=".npz",
                dir=self.root,
                delete=False,
            ) as f:
                temp_path = Path(f.name)

                np.savez_compressed(
                    f,
                    metadata=np.frombuffer(
                        metadata_bytes,
                        dtype=np.uint8,
                    ),
                    vectors=vectors,
                )
                f.flush()
                os.fsync(f.fileno())

            os.replace(temp_path, self._path(did))

        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    def restore_all(self, retriever):
        """Restore saved indexes without running the embedder."""
        restored = 0

        for path in sorted(self.root.glob("*.npz")):
            with np.load(path, allow_pickle=False) as data:
                metadata = json.loads(
                    data["metadata"].tobytes().decode("utf-8")
                )
                vectors = data["vectors"].copy()

            if metadata.get("version") != 1:
                raise ValueError(f"Unsupported store version: {path}")

            if metadata.get("embedding_model") != EMBEDDING_MODEL:
                raise ValueError(f"Embedding model mismatch: {path}")

            if metadata.get("reranker_model") != RERANKER_MODEL:
                raise ValueError(f"Reranker model mismatch: {path}")

            did = metadata["document_id"]
            units = metadata["units"]

            if path != self._path(did):
                raise ValueError(f"Document filename mismatch: {path}")

            if (
                not isinstance(units, list)
                or not units
                or vectors.ndim != 2
                or vectors.shape[0] != len(units)
                or vectors.dtype != np.float32
                or not np.isfinite(vectors).all()
            ):
                raise ValueError(f"Invalid document index: {path}")

            ids = [str(u["paragraph_id"]) for u in units]
            if len(ids) != len(set(ids)):
                raise ValueError(f"Duplicate evidence IDs: {path}")

            if any(str(u["document_id"]) != did for u in units):
                raise ValueError(f"Document ID mismatch: {path}")

            if did in retriever.documents:
                raise ValueError(f"Document already indexed: {did}")

            retriever.documents[did] = units
            retriever.vectors[did] = vectors

            restored += 1

        return restored
