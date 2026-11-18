# TrustRAG — Reliability-Oriented Adaptive RAG

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/) [![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)](https://fastapi.tiangolo.com/) [![Inference](https://img.shields.io/badge/Inference-Apple%20MLX-555)](https://github.com/ml-explore/mlx-lm)

**An end-to-end, locally hosted retrieval-augmented generation system with frozen, cost-aware retrieval routing and auditable reliability evaluation.**

TrustRAG ingests PDF documents, encodes document-scoped evidence, retrieves with dense and hybrid search, reranks with a cross-encoder, and generates grounded answers with a local Qwen model. A **frozen Ridge policy** uses six features available at inference time to select between **Keep** (dense retrieval + reranking) and **Hybrid** (dense + BM25/RRF + reranking). The policy is never retrained at request time and does not access gold labels.

This repository is an **engineering portfolio edition** of a larger research workspace. It prioritizes inspectable code, tests, a versioned frozen policy artifact, and honest reporting of an already-completed pilot. It is **not** a claim of improved answer accuracy or a production-secure public API.

## Architecture

```text
PDF upload / supplied evidence units
          │
          ▼
PDF extraction → chunking → document-scoped embedding index → local document store
          │
Question  ▼
Dense top-30 → cross-encoder rerank top-10 (Keep)
          │
       six baseline-only features
          │
    frozen Ridge + decision threshold
          ├── Keep: reuse dense reranked top-10
          └── Hybrid: BM25 + dense RRF top-30 → cross-encoder top-10
                         │
                 context budget / citation mapping
                         │
          local Qwen3-4B (MLX) → answer + citations + timings
```

The baseline cross-encoder always runs **before** routing; the Hybrid path has additional lexical search and reranking. RRF uses constant 60. Models: `sentence-transformers/all-MiniLM-L6-v2`, `cross-encoder/ms-marco-MiniLM-L6-v2`, and `mlx-community/Qwen3-4B-Instruct-2507-4bit`. The local generation backend is for **Apple Silicon**.

## Quick start (macOS / Apple Silicon)

Requirements: Python 3.11, an Apple Silicon Mac, disk space/network access for downloading model weights, and a trusted clone. Dependency pins were captured from the development machine; a fresh-environment install has **not** been verified.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

On first startup, the embedding and cross-encoder models are downloaded. The Qwen MLX model is initialized on the first `/answer` request. The application persists indexed document data to `data/document_store/` and restores it on restart. **Use only trusted local PDF files**: this development API has no authentication or access control, so do not expose it directly to the internet.

The API starts without the original research PDF or its prebuilt document index. Upload a PDF to build an index:

```bash
curl -X POST 'http://127.0.0.1:8000/documents/upload' \
  -F 'document_id=my-paper' -F 'file=@/path/to/your-paper.pdf'

curl -X POST 'http://127.0.0.1:8000/answer' \
  -H 'Content-Type: application/json' \
  -d '{"document_id":"my-paper","question":"What is the main contribution?","question_id":"demo-001","max_tokens":256}'
```

Other endpoints: `GET /health`, `POST /documents` (structured evidence units), and `POST /query` (retrieval only). Interactive API schema: http://127.0.0.1:8000/docs. PDF uploads have a 10 MB limit.

## Pilot benchmark: saved results, not a population estimate

Ten selected questions about **one research PDF** were evaluated under four strategies (40 generated answers). Correctness was manually reviewed or reused only for **exactly identical answers**, with SHA-256 verification. `uncertain` was not counted as `pass`.

| Strategy | Pass | Fail | Uncertain | Generated tokens (10 queries) |
|---|---:|---:|---:|---:|
| Dense Raw | 7 | 2 | 1 | 1,912 |
| Always Keep | 7 | 2 | 1 | 1,672 |
| Always Hybrid | 6 | 3 | 1 | 1,752 |
| Frozen Ridge Adaptive | 6 | 3 | 1 | 1,755 |

**No correctness gain was demonstrated for Adaptive in this pilot.** These counts are descriptive, not statistically significant. Generated tokens alone are **not total inference cost**; the fixed baselines record generation-only latency while the Adaptive run records pipeline latency, so their latency values are not directly comparable. Citation-ID validity does not imply a claim is faithful. Question Q5's answerability remains uncertain. The original source PDF and its derived local document index are deliberately not bundled, so the saved benchmark can be inspected and validated **but not regenerated end-to-end from this ZIP alone**.

### Reliability case studies

- **Q4 — reranking-associated evidence loss:** Dense Raw retained the formula for combining two fixed pre-trained models; reranked evidence omitted the decisive chunk, and the resulting answer confused `f_fair` with `f_simp`. This case does not establish population-level causality.
- **Q2 — unsupported comparative claims:** Some responses misattributed constraints from other fairness methods to FRAPPÉ; one Dense Raw answer also hit the 512-token output cap.
- **Q10 — mathematical expression error:** All four strategies generated a formula missing parentheses in `(1 - alpha(x)) * (f_simp(x) - f_perf(x))`, despite evidence relating to the correction. A possible PDF extraction contribution has not been ruled out.

Q6 and Q7 show that context truncation (6/10 and 8/10 retrieved units included for Dense Raw, respectively) does **not** always cause an incorrect answer. The context budget can end early on a large chunk; this is a documented engineering limitation.

See [`docs/EVALUATION.md`](docs/EVALUATION.md) and `evaluation/results/benchmark_v1_four_strategy_report.json` for the report, label lineage, and scope.

## Tests and reproducibility

```bash
# Lightweight test suite; runs without loading the 4B generator
python -m pytest -q

# Validate saved policy checksum, answer hashes, and report consistency
python scripts/verify_portfolio.py
```

The developer's **original complete local workspace** previously reported **49 passed, 7 warnings**. This export excludes two data-dependent held-out parity tests because the corresponding research dataset was not included. Those test sources are archived in `docs/optional_validation/`; use the original workspace for full QASPER parity checking. For this export, **44 tests passed (8 version warnings)** in a separate Linux inspection environment after excluding `tests/test_retriever.py`, because FAISS was not installed there. The archived Ridge estimator was fitted with scikit-learn 1.9.1, while that inspection environment used 1.8.0; do not treat this as a compatibility guarantee. Run the full selected suite on your Apple Silicon development machine with the pinned dependencies.

The saved four-strategy report is a historical immutable artifact: rerunning the evaluation requires indexing the **same source PDF with matching chunk IDs**, restoring the original environment/model weights, and using compatible saved retrieval rankings. The included `evaluation/run_fixed_baselines.py` expects local `paper-demo-001` in `data/document_store/` and is not a standalone replay from text-only JSONL.

## Project layout

```text
src/                      # FastAPI, retrieval, frozen routing, generation, storage
evaluation/               # Evaluation scripts, selected datasets/results/reviews
experiments/retrieval/    # Preserved Step 58 BM25/RRF implementation (runtime import)
data/qasper/.../           # Frozen Ridge artifact and checksum manifest
tests/                    # Selected model-free automated tests
scripts/verify_portfolio.py
docs/                      # Architecture, evaluation, packaging notes
```

**Security:** The frozen `.joblib` file is pickle-based; **load only artifacts from trusted sources**. Model downloads and local uploaded files may contain untrusted content. Do not operate this no-authentication demo service on a public interface. No proprietary research corpora, local environments, model weights, or private PDF are included.

## Research provenance

The frozen policy was fit on **599 evaluable QASPER development questions**, with 6 runtime features and Ridge `alpha=20`; the selection threshold is `0.031045372647072564` (see the preserved manifest). Frozen policy development used no held-out labels for fitting, and the policy is not refit in this engineering version. This local single-PDF pilot is separate from held-out QASPER retrieval evidence evaluations; do not conflate the two.
