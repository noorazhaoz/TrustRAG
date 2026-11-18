
from types import SimpleNamespace

from src.answer_service import (
    AnswerService,
    validate_citations,
)


EVIDENCE = [
    {
        "unit_id": "page1_chunk1",
        "citation_id": "E1",
        "text": "The study included 250 participants.",
        "rank": 1,
    }
]


def test_valid_citations():
    result = validate_citations(
        "The study included 250 participants [E1].",
        EVIDENCE,
    )

    assert result["valid"] is True
    assert result["used_citations"] == ["[E1]"]
    assert result["invalid_citations"] == []


def test_invalid_citation_detected():
    result = validate_citations(
        "The result was significant [E99].",
        EVIDENCE,
    )

    assert result["valid"] is False
    assert result["invalid_citations"] == ["[E99]"]


def test_no_citations_reported():
    result = validate_citations(
        "Insufficient evidence.",
        EVIDENCE,
    )

    assert result["valid"] is True
    assert result["has_citations"] is False


class FakeRetriever:
    def query(self, document_id, question, question_id="online"):
        return SimpleNamespace(
            action="keep",
            predicted_delta=0.01,
            latency_ms=10.0,
            evidence=[
                {
                    "unit_id": "page1_chunk1",
                    "text": "250 participants.",
                    "rank": 1,
                }
            ],
        )


class FakeGenerator:
    def generate(self, question, evidence, **kwargs):
        return SimpleNamespace(
            answer="The study included 250 participants [E1].",
            model_id="fake-qwen",
        )


def test_generation_metrics():
    service = AnswerService(
        FakeRetriever(), FakeGenerator()
    )

    result = service.answer(
        "paper1", "How many participants?"
    )

    assert result.generation_latency_ms >= 0
    assert result.pipeline_latency_ms >= 0
    assert result.citation_validation["valid"] is True
    assert result.retrieval_policy == "keep"

def test_grouped_citations():
    evidence = [
        {"citation_id": "E1"},
        {"citation_id": "E2"},
        {"citation_id": "E3"},
    ]

    result = validate_citations(
        "Supported by [E1, E2] and [E3, E99].",
        evidence,
    )

    assert result["valid"] is False
    assert result["used_citations"] == [
        "[E1]", "[E2]", "[E3]", "[E99]"
    ]
    assert result["invalid_citations"] == ["[E99]"]