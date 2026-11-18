
"""Combine retrieval and human-review results by question."""

import json
from pathlib import Path


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def build_summary(
    audit,
    labels,
    claim_reviews,
    abstention_reviews,
    correctness_reviews,
):
    label_map = {
        row["question_id"]: row
        for row in labels
    }

    abstention_map = {
        row["question_id"]: row
        for row in abstention_reviews
    }

    claims_by_question = {}

    for review in claim_reviews:
        qid = review["question_id"]
        claims_by_question.setdefault(qid, []).append(review)

    correctness_map = {
        row["question_id"]: row
        for row in correctness_reviews
    }

    summary = []

    for row in audit:
        qid = row["question_id"]
        label = label_map.get(qid)

        coverage = None

        if label is not None:
            required = set(label["answer_evidence_ids"])

            if not required:
                raise ValueError(f"{qid}: empty evidence labels")

            coverage = {
                method: required.issubset(set(row[method]))
                for method in ("dense_raw", "keep", "hybrid")
            }

        summary.append({
            "question_id": qid,
            "question": row["question"],
            "document_id": row["document_id"],
            "evidence_coverage": coverage,
            "claim_reviews": claims_by_question.get(qid, []),
            "answerability_review": abstention_map.get(qid),
            "correctness_review": correctness_map.get(qid),
        })

    return summary


def main():
    audit = load_jsonl(
        "evaluation/results/retrieval_audit.jsonl"
    )

    labels = load_jsonl(
        "evaluation/reviews/answer_evidence_labels.jsonl"
    )

    claims = load_jsonl(
        "evaluation/reviews/optimization_claim_reviews.jsonl"
    )

    abstentions = load_jsonl(
        "evaluation/reviews/answerability_reviews.jsonl"
    )
    correctness = load_jsonl(
        "evaluation/reviews/question_correctness_reviews.jsonl"
    )

    summary = build_summary(
        audit, labels, claims, abstentions, correctness
    )

    output = Path(
        "evaluation/results/question_summary_v2.jsonl"
    )

    if output.exists():
        raise FileExistsError(output)

    with output.open("w", encoding="utf-8") as f:
        for row in summary:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    for row in summary:
        print(
            row["question_id"],
            "| coverage:", row["evidence_coverage"],
            "| reviewed claims:", len(row["claim_reviews"]),
            "| answerability:",
            (
                row["answerability_review"]["answerability"]
                if row["answerability_review"] else None
            ),
            "| correctness:",
            (
                row["correctness_review"]["correctness"]
                if row["correctness_review"] else None
            ),
        )


if __name__ == "__main__":
    main()
