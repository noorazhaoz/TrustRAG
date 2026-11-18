
"""Generate answers using saved fixed retrieval baselines.

Reuses the frozen retrieval audit. Does not run the Ridge router.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from src.answer_service import build_evidence_context, SYSTEM_PROMPT, validate_citations
from src.document_store import DocumentStore
from src.llm_backend import MLXBackend


METHODS = ("dense_raw", "keep", "hybrid")
DOCUMENT_ID = "paper-demo-001"


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_units(document_id):
    with np.load(
        DocumentStore()._path(document_id),
        allow_pickle=False,
    ) as data:
        metadata = json.loads(
            data["metadata"].tobytes().decode("utf-8")
        )

    return {
        unit["paragraph_id"]: unit
        for unit in metadata["units"]
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--method",
        required=True,
        choices=METHODS,
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
    )
    args = parser.parse_args()

    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")

    audit = load_jsonl(
        "evaluation/results/retrieval_audit_v1.jsonl"
    )

    if len(audit) != 10:
        raise ValueError("Expected exactly 10 audit questions")

    output = Path(
        f"evaluation/results/benchmark_v1_{args.method}.jsonl"
    )

    completed = set()

    if output.exists():
        existing = load_jsonl(output)

        audit_ids = [row["question_id"] for row in audit]
        existing_ids = [row["question_id"] for row in existing]

        if len(existing_ids) != len(set(existing_ids)):
            raise ValueError("Duplicate questions in existing results")

        if existing_ids != audit_ids[:len(existing_ids)]:
            raise ValueError(
                "Existing results are not a valid audit prefix"
            )

        for record in existing:
            if record.get("condition") != args.method:
                raise ValueError("Existing result method mismatch")
            if record.get("max_tokens") != args.max_tokens:
                raise ValueError("Existing token setting mismatch")

        completed = set(existing_ids)

        print(
            f"Resuming: {len(completed)} questions already completed"
        )

    units = load_units(DOCUMENT_ID)
    backend = MLXBackend()

    output.parent.mkdir(parents=True, exist_ok=True)

    for row in audit:
        qid = row["question_id"]
        if qid in completed:
            print(f"Skipping completed: {qid}")
            continue
        question = row["question"]
        evidence_ids = row[args.method]

        if row["document_id"] != DOCUMENT_ID:
            raise ValueError(f"Unexpected document: {qid}")

        if len(evidence_ids) != 10:
            raise ValueError(f"Expected 10 evidence units: {qid}")

        evidence = [
            {
                "unit_id": uid,
                "record_type": units[uid].get("record_type", ""),
                "text": units[uid]["text"],
                "rank": rank,
            }
            for rank, uid in enumerate(evidence_ids, 1)
        ]

        context, included, truncated = build_evidence_context(
            evidence
        )

        if not included:
            raise RuntimeError(
                f"No evidence included: {qid}"
            )

        if truncated:
            print(
                f"[Context] {qid}: truncated, "
                f"{len(included)}/{len(evidence_ids)} units included"
            )

        start = perf_counter()

        result = backend.generate(
            question=question,
            evidence=context,
            max_tokens=args.max_tokens,
            system_prompt=SYSTEM_PROMPT,
        )

        generation_latency_ms = (
            perf_counter() - start
        ) * 1000

        citation_check = validate_citations(
            result.answer,
            included,
        )

        record = {
            "question_id": qid,
            "document_id": DOCUMENT_ID,
            "condition": args.method,
            "question": question,
            "evidence_ids": evidence_ids,
            "included_evidence_ids": [
                item["unit_id"] for item in included
            ],
            "context_truncated": truncated,
            "answer": result.answer,
            "max_tokens": args.max_tokens,
            "generation_tokens": result.generation_tokens,
            "finish_reason": result.finish_reason,
            "generation_latency_ms": generation_latency_ms,
            "citation_validation": citation_check,
            "citation_sources": {
                item["citation_id"]: item["unit_id"]
                for item in included
            },
        }

        with output.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

        print(
            qid,
            "|", args.method,
            "| tokens:", result.generation_tokens,
            "| finish:", result.finish_reason,
        )

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
