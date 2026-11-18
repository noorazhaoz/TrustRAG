
"""Multi-question retrieval audit without generation."""

import json
from pathlib import Path

import numpy as np

from src.adaptive_retriever import AdaptiveRetriever, BM25, rrf
from src.document_store import DocumentStore


INPUT = Path(
    "evaluation/datasets/optimization_benchmark_v1.jsonl"
)
OUTPUT = Path(
    "evaluation/results/retrieval_audit_v1.jsonl"
)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)

    retriever = AdaptiveRetriever(router=None)
    DocumentStore().restore_all(retriever)

    questions = [
        json.loads(line)
        for line in INPUT.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    for row in questions:
        did = row["document_id"]
        question = row["question"]

        doc = retriever.documents[did]
        vectors = retriever.vectors[did]

        qvec = retriever.embedder.encode(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")[0]

        scores = vectors @ qvec

        dense_order = np.argsort(
            -scores, kind="stable"
        )

        raw = dense_order[:10]

        keep = retriever._rerank(
            question, doc, dense_order[:30]
        )

        bm25 = BM25([u["text"] for u in doc])
        bm_scores = bm25.score(question)

        bm_order = np.argsort(
            -bm_scores, kind="stable"
        )

        hybrid_order = rrf(
            dense_order,
            bm_order,
            len(doc),
            constant=60,
        )

        hybrid = retriever._rerank(
            question, doc, hybrid_order[:30]
        )

        def unit_ids(indices):
            return [
                str(doc[int(i)]["paragraph_id"])
                for i in indices
            ]

        raw_ids = unit_ids(raw)
        keep_ids = unit_ids(keep)
        hybrid_ids = unit_ids(hybrid)

        result = {
            "question_id": row["question_id"],
            "document_id": did,
            "question": question,
            "dense_raw": raw_ids,
            "keep": keep_ids,
            "hybrid": hybrid_ids,
            "raw_keep_overlap": len(
                set(raw_ids) & set(keep_ids)
            ),
            "keep_hybrid_overlap": len(
                set(keep_ids) & set(hybrid_ids)
            ),
        }

        with OUTPUT.open(
            "a", encoding="utf-8"
        ) as f:
            f.write(
                json.dumps(result, ensure_ascii=False)
                + "\n"
            )

        print(
            row["question_id"],
            "| Raw/Keep overlap:",
            result["raw_keep_overlap"],
            "| Keep/Hybrid overlap:",
            result["keep_hybrid_overlap"],
        )

    print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    main()
