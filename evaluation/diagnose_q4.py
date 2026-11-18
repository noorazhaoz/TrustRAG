
"""Offline diagnostic: add missing evidence to Q4.

Does not modify the frozen policy or production retriever.
"""

import json
from pathlib import Path

import numpy as np

from src.document_store import DocumentStore
from src.answer_service import (
    build_evidence_context,
    SYSTEM_PROMPT,
    validate_citations,
)
from src.llm_backend import MLXBackend


DOCUMENT_ID = "paper-demo-001"
REFERENCE_ID = "page13_chunk3"

QUESTION = "How does the method combine two pre-trained models?"

RESULT_PATH = Path(
    "evaluation/results/faithfulness_run_001.jsonl"
)


def main():
    # Load the original Q4 retrieval results.
    records = [
        json.loads(line)
        for line in RESULT_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    original = next(
        r for r in records
        if r["question_id"] == "faith-q4"
    )

    store = DocumentStore()

    with np.load(
        store._path(DOCUMENT_ID),
        allow_pickle=False,
    ) as data:
        metadata = json.loads(
            data["metadata"].tobytes().decode("utf-8")
        )

    units = {
        item["paragraph_id"]: item
        for item in metadata["units"]
    }

    original_ids = original["retrieved_evidence_ids"]

    assert REFERENCE_ID not in original_ids

    # Keep the original Top-9 and replace rank 10
    # with the missing formula evidence.
    augmented_ids = original_ids[:9] + [REFERENCE_ID]

    evidence = [
        {
            "unit_id": uid,
            "record_type": units[uid]["record_type"],
            "text": units[uid]["text"],
            "rank": i,
        }
        for i, uid in enumerate(augmented_ids, start=1)
    ]

    context, included, truncated = build_evidence_context(
        evidence
    )

    assert not truncated, (
        "Evidence context was truncated; "
        "the intervention must be adjusted."
    )

    assert any(
        item["unit_id"] == REFERENCE_ID
        for item in included
    )

    print("Original policy:", original["retrieval_policy"])
    print("Added evidence:", REFERENCE_ID)

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
        "experiment": "q4_evidence_augmented_keep",
        "question_id": "faith-q4",
        "intervention": REFERENCE_ID,
        "original_policy": original["retrieval_policy"],
        "evidence_ids": augmented_ids,
        "answer": generated.answer,
        "generation_tokens": generated.generation_tokens,
        "finish_reason": generated.finish_reason,
        "citation_validation": citation_check,
        "citation_sources": {
            item["citation_id"]: item["unit_id"]
            for item in included
        },
    }

    output_path = Path(
        "evaluation/results/q4_evidence_augmented.json"
    )

    if output_path.exists():
        raise FileExistsError(output_path)

    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nANSWER:\n")
    print(generated.answer)

    print("\nFINISH:", generated.finish_reason)
    print("TOKENS:", generated.generation_tokens)
    print("CITATIONS:", citation_check)


if __name__ == "__main__":
    main()
