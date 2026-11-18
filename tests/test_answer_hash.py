
import hashlib

import pytest

from evaluation.schema import QuestionCorrectnessReview
from evaluation.validate_run_reviews import validate_answer_hash


def test_answer_hash_valid():
    answer = "Correct answer."

    review = QuestionCorrectnessReview(
        question_id="q1",
        run_id="run-1",
        condition="dense_raw",
        correctness="pass",
        answer_sha256=hashlib.sha256(
            answer.encode("utf-8")
        ).hexdigest(),
    )

    validate_answer_hash(
        review,
        {"answer": answer},
    )


def test_answer_hash_mismatch():
    original = "Correct answer."

    review = QuestionCorrectnessReview(
        question_id="q1",
        run_id="run-1",
        condition="dense_raw",
        correctness="pass",
        answer_sha256=hashlib.sha256(
            original.encode("utf-8")
        ).hexdigest(),
    )

    with pytest.raises(
        ValueError,
        match="Answer hash mismatch",
    ):
        validate_answer_hash(
            review,
            {"answer": "Modified answer."},
        )
