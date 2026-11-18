
from types import SimpleNamespace

from src.answer_service import (
    AnswerService,
    build_evidence_context,
)


class FakeRetriever:
    def query(self, document_id, question, question_id="online"):
        return SimpleNamespace(
            action="hybrid",
            predicted_delta=0.05,
            latency_ms=100.0,
            evidence=[
                {
                    "unit_id": "page1_chunk1",
                    "text": "The study included 250 participants.",
                    "rank": 1,
                }
            ],
        )


class FakeGenerator:
    def generate(self, question, evidence, **kwargs):
        assert "[E1]" in evidence
        assert "250 participants" in evidence
        assert kwargs["max_tokens"] == 256

        return SimpleNamespace(
            answer="The study included 250 participants [E1].",
            model_id="fake-qwen",
        )


def test_grounded_answer():
    service = AnswerService(
        FakeRetriever(),
        FakeGenerator(),
    )

    result = service.answer(
        "paper1",
        "How many participants?",
    )

    assert "250 participants" in result.answer
    assert result.retrieval_policy == "hybrid"
    assert result.evidence[0]["citation_id"] == "E1"
    assert result.evidence_truncated is False


def test_context_limit():
    evidence = [
        {"unit_id": "p1", "text": "A" * 100},
        {"unit_id": "p2", "text": "B" * 100},
    ]

    context, included, truncated = build_evidence_context(
        evidence,
        limit=130,
    )

    assert len(included) == 1
    assert "[E1]" in context
    assert truncated is True
