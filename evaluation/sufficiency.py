
"""Evaluate reference recall and human-reviewed evidence sufficiency."""

import json
from pathlib import Path


METHODS = ("dense_raw", "keep", "hybrid")


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def evaluate_evidence(
    retrieved_ids,
    reference_ids,
    sufficient_groups=None,
):
    retrieved = set(retrieved_ids)
    references = set(reference_ids)

    recall = (
        len(retrieved & references) / len(references)
        if references else None
    )

    if sufficient_groups is None:
        sufficient = None
    else:
        if not sufficient_groups or any(
            not group for group in sufficient_groups
        ):
            raise ValueError("Invalid sufficient evidence groups")

        sufficient = any(
            set(group).issubset(retrieved)
            for group in sufficient_groups
        )

    return {
        "reference_recall": recall,
        "evidence_sufficient": sufficient,
    }


def main():
    questions = {
        row["question_id"]: row
        for row in load_jsonl(
            "evaluation/datasets/optimization_benchmark_v1.jsonl"
        )
    }

    audit = load_jsonl(
        "evaluation/results/retrieval_audit_v1.jsonl"
    )

    sufficiency_labels = {
        row["question_id"]: row["sufficient_evidence_groups"]
        for row in load_jsonl(
            "evaluation/reviews/evidence_sufficiency_labels.jsonl"
        )
    }

    output = []

    for row in audit:
        qid = row["question_id"]

        if qid not in questions:
            raise ValueError(f"Unknown question: {qid}")

        question = questions[qid]

        results = {
            method: evaluate_evidence(
                row[method],
                question["reference_evidence_ids"],
                sufficiency_labels.get(qid),
            )
            for method in METHODS
        }

        output.append({
            "question_id": qid,
            "methods": results,
        })

        print(f"\n{qid}")
        for method, result in results.items():
            print(
                f"  {method}: "
                f"recall={result['reference_recall']} "
                f"sufficient={result['evidence_sufficient']}"
            )

    result_path = Path(
        "evaluation/results/evidence_sufficiency_v4.jsonl"
    )

    if result_path.exists():
        raise FileExistsError(result_path)

    with result_path.open("w", encoding="utf-8") as f:
        for row in output:
            f.write(json.dumps(row) + "\n")

    print("\nSaved:", result_path)


if __name__ == "__main__":
    main()
