
"""Repeat Q4 generation using fixed evidence sets.

Offline evaluation only. No policy modifications.
"""

import json
from pathlib import Path

import numpy as np

from src.document_store import DocumentStore
from src.answer_service import build_evidence_context, SYSTEM_PROMPT
from src.llm_backend import MLXBackend


DOCUMENT_ID = "paper-demo-001"
QUESTION = "How does the method combine two pre-trained models?"

REPEATS = 3
MAX_TOKENS = 512

OUTPUT = Path(
    "evaluation/results/q4_repeated_generations.jsonl"
)


def load_units():
    store = DocumentStore()

    with np.load(
        store._path(DOCUMENT_ID),
        allow_pickle=False,
    ) as data:
        metadata = json.loads(
            data["metadata"].tobytes().decode("utf-8")
        )

    return {
        unit["paragraph_id"]: unit
        for unit in metadata["units"]
    }


def get_conditions():
    raw = json.loads(
        Path(
            "evaluation/results/q4_dense_raw_answer.json"
        ).read_text(encoding="utf-8")
    )

    runs = [
        json.loads(line)
        for line in Path(
            "evaluation/results/faithfulness_run_001.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    keep = next(
        run for run in runs
        if run["question_id"] == "faith-q4"
    )

    return {
        "dense_raw": raw["evidence_ids"],
        "reranked_keep": keep["retrieved_evidence_ids"],
    }


def main():
    if OUTPUT.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT}"
        )

    units = load_units()
    conditions = get_conditions()

    contexts = {}

    for name, evidence_ids in conditions.items():
        evidence = [
            {
                "unit_id": unit_id,
                "record_type": units[unit_id].get(
                    "record_type", ""
                ),
                "text": units[unit_id]["text"],
                "rank": rank,
            }
            for rank, unit_id in enumerate(
                evidence_ids, start=1
            )
        ]

        context, included, truncated = (
            build_evidence_context(evidence)
        )

        if truncated:
            raise RuntimeError(
                f"{name}: evidence context truncated"
            )

        if len(included) != len(evidence_ids):
            raise RuntimeError(
                f"{name}: evidence count changed"
            )

        contexts[name] = {
            "context": context,
            "evidence_ids": evidence_ids,
        }

    backend = MLXBackend()

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    for name, details in contexts.items():
        for repeat in range(1, REPEATS + 1):
            print(f"\nRunning {name}: {repeat}/{REPEATS}")

            result = backend.generate(
                question=QUESTION,
                evidence=details["context"],
                max_tokens=MAX_TOKENS,
                system_prompt=SYSTEM_PROMPT,
            )

            record = {
                "question_id": "faith-q4",
                "condition": name,
                "repeat": repeat,
                "evidence_ids": details["evidence_ids"],
                "answer": result.answer,
                "generation_tokens": result.generation_tokens,
                "finish_reason": result.finish_reason,
            }

            with OUTPUT.open(
                "a", encoding="utf-8"
            ) as f:
                f.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    ) + "\n"
                )

            print("Tokens:", result.generation_tokens)
            print("Finish:", result.finish_reason)

    print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    main()
