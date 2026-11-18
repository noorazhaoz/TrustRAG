
"""Compare Keep and Hybrid retrieval for Q4.

Offline diagnostic only.
Does not modify the frozen policy or production API.
"""

import json
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments.retrieval.step58_retrieval_repair import BM25, rrf
from src.adaptive_retriever import AdaptiveRetriever
from src.document_store import DocumentStore


DOCUMENT_ID = "paper-demo-001"
QUESTION = "How does the method combine two pre-trained models?"
TARGET_ID = "page13_chunk3"

OUTPUT_PATH = Path(
    "evaluation/results/q4_retrieval_comparison.json"
)


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(OUTPUT_PATH)

    # Load the existing models and persistent document index.
    # No frozen policy prediction is needed for fixed baselines.
    retriever = AdaptiveRetriever(router=None)
    restored = DocumentStore().restore_all(retriever)

    print("Restored documents:", restored)

    doc = retriever.documents[DOCUMENT_ID]
    vectors = retriever.vectors[DOCUMENT_ID]

    # Encode the question once for both retrieval strategies.
    qvec = retriever.embedder.encode(
        [QUESTION],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")[0]

    dense_scores = vectors @ qvec

    dense_order = np.argsort(
        -dense_scores,
        kind="stable",
    )

    # A: Always Keep — Dense Top-30 + Cross-Encoder.
    start = perf_counter()

    keep_indices = retriever._rerank(
        QUESTION,
        doc,
        dense_order[:30],
    )

    keep_latency_ms = (perf_counter() - start) * 1000

    # B: Always Hybrid — Dense + BM25 + RRF + Cross-Encoder.
    start = perf_counter()

    lexical_index = BM25(
        [unit["text"] for unit in doc]
    )

    bm_scores = lexical_index.score(QUESTION)

    bm_order = np.argsort(
        -bm_scores,
        kind="stable",
    )

    hybrid_order = rrf(
        dense_order,
        bm_order,
        len(doc),
        constant=60,
    )

    hybrid_indices = retriever._rerank(
        QUESTION,
        doc,
        hybrid_order[:30],
    )

    hybrid_latency_ms = (
        perf_counter() - start
    ) * 1000

    def summarize(indices, latency_ms):
        ids = [
            str(doc[i]["paragraph_id"])
            for i in indices
        ]

        return {
            "evidence_ids": ids,
            "target_retrieved": TARGET_ID in ids,
            "target_rank": (
                ids.index(TARGET_ID) + 1
                if TARGET_ID in ids else None
            ),
            "reranking_path_latency_ms": latency_ms,
        }

    results = {
        "question": QUESTION,
        "document_id": DOCUMENT_ID,
        "target_id": TARGET_ID,
        "keep": summarize(
            keep_indices, keep_latency_ms
        ),
        "hybrid": summarize(
            hybrid_indices, hybrid_latency_ms
        ),
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    for method in ("keep", "hybrid"):
        result = results[method]

        print(f"\n{method.upper()}")
        print(
            "Target retrieved:",
            result["target_retrieved"],
        )
        print(
            "Target rank:",
            result["target_rank"],
        )
        print("Top-10 IDs:")

        for rank, unit_id in enumerate(
            result["evidence_ids"], start=1
        ):
            print(f"  {rank:2d}. {unit_id}")

    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
