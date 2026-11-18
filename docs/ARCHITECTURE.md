# Architecture and implementation boundaries

## Request flow
`POST /documents/upload` parses a PDF into chunked evidence units (`src/pdf_ingestion.py`), embeds and indexes evidence in a document-scoped in-memory store (`src/adaptive_retriever.py`), and persists the index using `src/document_store.py`.

`POST /query` embeds a question, scores all units with dense similarity, reranks the top 30 using a cross-encoder, calculates **six features from only baseline retrieval**, and passes them to `FrozenPolicyRouter`. `Keep` reuses reranked dense top-10. `Hybrid` creates a BM25 index lazily, fuses rankings with RRF constant 60, and cross-encoder reranks its top 30. The original Step 58 BM25/RRF implementation is retained as a small legacy dependency at `experiments/retrieval/step58_retrieval_repair.py` to avoid changing established ranking semantics.

`POST /answer` selects evidence, applies a context budget, constructs citation IDs, calls the local MLX generation backend and returns an answer, evidence, timings, finish reason, and a structural citation validation result.

## Frozen policy
`src/policy_router.py` loads the included development-trained Ridge pipeline, checks SHA-256 against `frozen_manifest.json`, checks feature order, and uses the **frozen** threshold. Do not train or update this artifact via this portfolio app. The pickle-based joblib artifact must not be loaded from untrusted sources.

## Deployment limitations
- Local, non-authenticated demo API; host on `127.0.0.1` only.
- First run downloads embedding/reranker/LLM weights; local MLX backend requires Apple Silicon.
- Document indexes are local filesystem `.npz` files and are not designed for multi-worker shared-state deployment.
- Context packing may discard evidence chunks because of the character cap.
- Citation validation checks citation ID structure, not correctness or claim-level faithfulness.
- This archive does not include the original benchmark PDF or document `.npz` store.
