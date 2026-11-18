
"""Build a reproducible report from saved benchmark artifacts."""

import hashlib
import json
from collections import Counter
from pathlib import Path

from evaluation.schema import QuestionCorrectnessReview
from evaluation.validate_run_reviews import validate_answer_hash


BASE = Path("evaluation")
RESULTS = BASE / "results"
REVIEWS = BASE / "reviews"

OUTPUT = RESULTS / "benchmark_v1_report.json"


def load_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def unique_map(rows, key, label):
    mapping = {}
    for row in rows:
        value = row[key]
        if value in mapping:
            raise ValueError(f"Duplicate {label}: {value}")
        mapping[value] = row
    return mapping


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)

    runs = unique_map(
        load_jsonl(RESULTS / "benchmark_v1_adaptive.jsonl"),
        "question_id",
        "generation question",
    )

    evidence = unique_map(
        load_jsonl(RESULTS / "evidence_sufficiency_v4.jsonl"),
        "question_id",
        "evidence question",
    )

    review_files = [
        REVIEWS / "benchmark_v1_correctness_reviews_q1_q5.jsonl",
        REVIEWS / "benchmark_v1_correctness_reviews.jsonl",
    ]

    reviews = [
        QuestionCorrectnessReview.model_validate(row)
        for path in review_files
        for row in load_jsonl(path)
    ]

    review_map = unique_map(
        [review.model_dump() for review in reviews],
        "question_id",
        "review question",
    )

    ids = set(runs)

    if ids != set(evidence) or ids != set(review_map):
        raise ValueError("Question IDs do not match across files")

    questions = []

    for qid in sorted(ids):
        run = runs[qid]
        review = QuestionCorrectnessReview.model_validate(
            review_map[qid]
        )

        if review.run_id != f"benchmark_v1_adaptive:{qid}":
            raise ValueError(f"Run ID mismatch: {qid}")

        if review.condition != "adaptive":
            raise ValueError(f"Condition mismatch: {qid}")

        if review.answer_sha256 is None:
            raise ValueError(f"Missing answer hash: {qid}")

        validate_answer_hash(review, run)

        questions.append({
            "question_id": qid,
            "retrieval_policy": run["retrieval_policy"],
            "correctness": review.correctness,
            "answer_sha256": review.answer_sha256,
            "citation_ids_valid": run["citation_ids_valid"],
            "generation_tokens": run.get("generation_tokens"),
            "finish_reason": run.get("finish_reason"),
            "pipeline_latency_ms": run["pipeline_latency_ms"],
            "evidence_sufficiency": evidence[qid]["methods"],
        })

    counts = Counter(
        question["correctness"]
        for question in questions
    )

    latency_values = [
        q["pipeline_latency_ms"]
        for q in questions
    ]

    report = {
        "benchmark": "optimization_benchmark_v1",
        "method": "frozen_ridge_adaptive",
        "total_questions": len(questions),
        "correctness": {
            "pass": counts["pass"],
            "fail": counts["fail"],
            "uncertain": counts["uncertain"],
        },
        "citation_id_validity": {
            "valid": sum(
                q["citation_ids_valid"]
                for q in questions
            ),
            "total": len(questions),
        },
        "mean_pipeline_latency_ms": (
            sum(latency_values) / len(latency_values)
        ),
        "total_generation_tokens": sum(
            q["generation_tokens"] or 0
            for q in questions
        ),
        "questions": questions,
        "limitations": [
            "Single PDF and ten selected questions.",
            "Correctness is human-reviewed, not an automatic metric.",
            "Citation ID validity does not imply factual faithfulness.",
            "Evidence sufficiency uses selected human-reviewed evidence groups.",
            "The likely-unanswerable question remains provisionally labeled.",
            "Comparative generation results for fixed retrieval baselines are not yet available.",
        ],
    }

    OUTPUT.write_text(
        json.dumps(
            report, indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )

    print("Saved:", OUTPUT)
    print("Questions:", report["total_questions"])
    print("Correctness:", report["correctness"])
    print("Citation validity:", report["citation_id_validity"])
    print(
        "Mean latency:",
        round(report["mean_pipeline_latency_ms"], 1),
        "ms",
    )


if __name__ == "__main__":
    main()
