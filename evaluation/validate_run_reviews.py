
"""Validate that human reviews correspond to saved generations."""

import json
from pathlib import Path
import hashlib

from evaluation.schema import QuestionCorrectnessReview


RUNS_PATH = Path(
    "evaluation/results/q4_repeated_generations.jsonl"
)

REVIEWS_PATH = Path(
    "evaluation/reviews/q4_run_correctness_reviews.jsonl"
)


def load_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]

def validate_answer_hash(review, run):
    if review.answer_sha256 is None:
        return

    actual_hash = hashlib.sha256(
        run["answer"].encode("utf-8")
    ).hexdigest()

    if actual_hash != review.answer_sha256:
        raise ValueError(
            f"Answer hash mismatch: {review.run_id}"
        )

def main():
    runs = load_jsonl(RUNS_PATH)

    run_map = {}
    for run in runs:
        run_id = (
            f"{run['question_id']}:"
            f"{run['condition']}:"
            f"{run['repeat']}"
        )

        if run_id in run_map:
            raise ValueError(f"Duplicate run: {run_id}")

        run_map[run_id] = run

    reviews = [
        QuestionCorrectnessReview.model_validate(row)
        for row in load_jsonl(REVIEWS_PATH)
    ]

    reviewed_ids = set()

    for review in reviews:
        if review.run_id in reviewed_ids:
            raise ValueError(
                f"Duplicate review: {review.run_id}"
            )

        reviewed_ids.add(review.run_id)

        if review.run_id not in run_map:
            raise ValueError(
                f"Unknown run: {review.run_id}"
            )

        run = run_map[review.run_id]

        if review.question_id != run["question_id"]:
            raise ValueError("Question ID mismatch")

        if review.condition != run["condition"]:
            raise ValueError("Condition mismatch")

        validate_answer_hash(review, run)

        print(
            review.run_id,
            "→",
            review.correctness,
            "✓"
        )

    print(f"\nValidated {len(reviews)} run-level reviews.")


if __name__ == "__main__":
    main()
