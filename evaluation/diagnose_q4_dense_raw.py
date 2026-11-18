
"""Q4 diagnostic using raw dense top-10 evidence."""

import json
from pathlib import Path

import numpy as np

from src.adaptive_retriever import AdaptiveRetriever
from src.document_store import DocumentStore
from src.answer_service import (
    build_evidence_context,
    SYSTEM_PROMPT,
    validate_citations,
)
from src.llm_backend import MLXBackend


DOCUMENT_ID = "paper-demo-001"
QUESTION = "How does the method combine two pre-trained models?"

OUTPUT_PATH = Path(
    "evaluation/results/q4_dense_raw_answer.json"
)


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(OUTPUT_PATH)

    retriever = AdaptiveRetriever(router=None)
    DocumentStore().restore_all(retriever)

    doc = retriever.documents[DOCUMENT_ID]
    vectors = retriever.vectors[DOCUMENT_ID]

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

    raw_indices = dense_order[:10]

    evidence = [
        {
            "unit_id": str(doc[int(i)]["paragraph_id"]),
            "record_type": doc[int(i)].get("record_type", ""),
            "text": doc[int(i)]["text"],
            "rank": rank,
        }
        for rank, i in enumerate(raw_indices, 1)
    ]

    context, included, truncated = build_evidence_context(
        evidence
    )

    if truncated:
        raise RuntimeError(
            "Evidence context truncated; comparison invalid."
        )

    target_id = "page13_chunk3"

    if not any(
        item["unit_id"] == target_id
        for item in included
    ):
        raise RuntimeError(
            "Correct formula evidence missing from context."
        )

    backend = MLXBackend()

    generated = backend.generate(
        question=QUESTION,
        evidence=context,
        max_tokens=512,
        system_prompt=SYSTEM_PROMPT,
    )

    citation_check = validate_citations(
        generated.answer,
        included,
    )

    output = {
        "experiment": "q4_dense_raw_top10",
        "document_id": DOCUMENT_ID,
        "question": QUESTION,
        "evidence_ids": [
            item["unit_id"] for item in included
        ],
        "answer": generated.answer,
        "generation_tokens": generated.generation_tokens,
        "finish_reason": generated.finish_reason,
        "citation_validation": citation_check,
        "citation_sources": {
            item["citation_id"]: item["unit_id"]
            for item in included
        },
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nDense Raw Top-10:")
    for item in included:
        print(
            item["citation_id"],
            "→",
            item["unit_id"],
        )

    print("\nANSWER:\n")
    print(generated.answer)

    print("\nFINISH:", generated.finish_reason)
    print("TOKENS:", generated.generation_tokens)
    print("CITATIONS:", citation_check)


if __name__ == "__main__":
    main()
