
import pytest
from pydantic import ValidationError

from evaluation.schema import (
    EvaluationQuestion,
    EvaluationRun,
    HumanReview,
)


def test_evaluation_question():
    question = EvaluationQuestion(
        question_id="q1",
        document_id="paper1",
        question="What is the main objective?",
    )

    assert question.reference_answer is None
    assert question.reference_evidence_ids == []


def test_evaluation_run():
    run = EvaluationRun(
        question_id="q1",
        document_id="paper1",
        answer="The model balances fidelity and fairness [E1].",
        retrieval_policy="keep",
        predicted_delta=0.015,
        retrieved_evidence_ids=["page7_chunk1"],
        cited_evidence_ids=["E1"],
        citation_ids_valid=True,
        retrieval_latency_ms=600,
        generation_latency_ms=11000,
        pipeline_latency_ms=11600,
    )

    assert run.retrieval_policy == "keep"
    assert run.citation_ids_valid is True


def test_human_review():
    review = HumanReview(
        question_id="q1",
        correctness="pass",
        faithfulness="uncertain",
    )

    assert review.correctness == "pass"
    assert review.faithfulness == "uncertain"


def test_negative_latency_rejected():
    with pytest.raises(ValidationError):
        EvaluationRun(
            question_id="q1",
            document_id="paper1",
            answer="Test",
            retrieval_policy="keep",
            predicted_delta=0,
            retrieved_evidence_ids=[],
            cited_evidence_ids=[],
            citation_ids_valid=True,
            retrieval_latency_ms=-1,
            generation_latency_ms=0,
            pipeline_latency_ms=0,
        )

def test_generation_diagnostics():
    from evaluation.schema import EvaluationRun

    run = EvaluationRun(
        question_id="q1",
        document_id="paper1",
        answer="Answer [E1].",
        retrieval_policy="keep",
        predicted_delta=0.01,
        retrieved_evidence_ids=["page1_chunk1"],
        cited_evidence_ids=["E1"],
        citation_ids_valid=True,
        retrieval_latency_ms=500,
        generation_latency_ms=1200,
        pipeline_latency_ms=1700,
        max_tokens=512,
        generation_tokens=380,
        finish_reason="stop",
    )

    assert run.generation_tokens == 380
    assert run.finish_reason == "stop"

def test_claim_review():
    from evaluation.schema import ClaimReview

    review = ClaimReview(
        question_id="eval-q2",
        claim_id="q2-c1",
        claim="FRAPPE is limited to binary sensitive attributes.",
        cited_evidence_ids=["E9"],
        support_label="not_supported",
        notes=(
            "E9 discusses limitations of other "
            "post-processing methods, not FRAPPE."
        ),
    )

    assert review.support_label == "not_supported"
    assert review.cited_evidence_ids == ["E9"]

def test_citation_source_mapping():
    from evaluation.schema import EvaluationRun

    run = EvaluationRun(
        question_id="q2",
        document_id="paper1",
        answer="Example answer [E9].",
        retrieval_policy="hybrid",
        predicted_delta=0.04,
        retrieved_evidence_ids=["page22_chunk2"],
        cited_evidence_ids=["E9"],
        citation_ids_valid=True,
        retrieval_latency_ms=500,
        generation_latency_ms=1200,
        pipeline_latency_ms=1700,
        citation_sources={
            "E9": "page22_chunk2"
        },
    )

    assert run.citation_sources["E9"] == "page22_chunk2"

def test_answerability_review():
    from evaluation.schema import AnswerabilityReview

    review = AnswerabilityReview(
        question_id="eval-q5",
        document_id="paper-demo-001",
        answerability="likely_unanswerable",
        response_type="abstained",
        abstention_appropriate="likely",
        review_method="document_keyword_audit_and_human_review",
    )

    assert review.response_type == "abstained"
    assert review.abstention_appropriate == "likely"

def test_question_correctness_review():
    from evaluation.schema import QuestionCorrectnessReview

    review = QuestionCorrectnessReview(
        question_id="eval-q4",
        correctness="fail",
        notes="Wrong model formulation.",
    )

    assert review.correctness == "fail"

def test_run_level_correctness_review():
    from evaluation.schema import QuestionCorrectnessReview

    review = QuestionCorrectnessReview(
        question_id="eval-q4",
        run_id="q4-dense-raw-repeat-1",
        condition="dense_raw",
        correctness="pass",
        notes="Correct f_perf and f_fair formulation.",
    )

    assert review.run_id == "q4-dense-raw-repeat-1"
    assert review.condition == "dense_raw"
    assert review.correctness == "pass"