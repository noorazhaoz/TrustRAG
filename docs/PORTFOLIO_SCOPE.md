# Portfolio edition: selected files and exclusions

This is an independent **copy**, not an in-place rewrite of the research project.

Included:
- Core `src/`, all current `evaluation/*.py` sources, benchmark dataset, four strategy outputs, correctness reviews, retrieval audit and sufficiency annotations.
- Frozen Ridge pipeline and its original manifest, unmodified.
- Selected tests that do not require missing held-out research CSVs.
- The original Step 58 retrieval implementation used by current runtime imports.

Excluded deliberately:
- Large historical research experiment scripts/outputs unrelated to the shipped API; most of `experiments/`, `scripts/`, and old `data/`.
- The original PDF (`data/optimization.pdf`), local `.npz` document index and Hugging Face/MLX model weights.
- Two held-out QASPER research parity tests from the default pytest suite; source copies remain under `docs/optional_validation/` and need unavailable held-out CSVs.
- Test scripts that eagerly download large language models (`tests/test_mlx.py`, `tests/test_backend.py` in the original) were not included in the default test run.
- Legacy `run.py`, which dispatches historic experiment scripts and cannot work without their archive.

This is an honest limited portfolio release, not a complete research reproduction. Before public posting, review all released benchmark question/answer text, confirm permissions to share research-derived data and the frozen policy artifact, and execute tests in a clean Python 3.11 Apple Silicon environment.

## Verification completed during packaging
- `python scripts/verify_portfolio.py`: passed, including frozen policy checksum and 40 report/answer labels.
- `compileall`: passed.
- 44 lightweight tests passed with 8 scikit-learn version mismatch warnings in the inspection environment, excluding the FAISS-dependent `tests/test_retriever.py` because FAISS was unavailable. No clean pinned Apple Silicon install has been executed here.
