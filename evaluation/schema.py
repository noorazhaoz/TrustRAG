
"""TrustRAG evaluation data schema."""

from typing import Literal

from pydantic import BaseModel, Field


ReviewLabel = Literal["pass", "fail", "uncertain"]


class EvaluationQuestion(BaseModel):
    question_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    question: str = Field(min_length=1)

    # Optional reference information, never used for routing.
    reference_answer: str | None = None
    reference_evidence_ids: list[str] = Field(default_factory=list)


class EvaluationRun(BaseModel):
    question_id: str
    document_id: str

    answer: str
    retrieval_policy: Literal["keep", "hybrid"]
    predicted_delta: float

    retrieved_evidence_ids: list[str]
    cited_evidence_ids: list[str]

    citation_ids_valid: bool

    retrieval_latency_ms: float = Field(ge=0)
    generation_latency_ms: float = Field(ge=0)
    pipeline_latency_ms: float = Field(ge=0)

    max_tokens: int | None = Field(default=None, ge=1)
    generation_tokens: int | None = Field(default=None, ge=0)
    finish_reason: str | None = None

    citation_sources: dict[str, str] = Field(default_factory=dict)


class HumanReview(BaseModel):
    question_id: str

    correctness: ReviewLabel
    faithfulness: ReviewLabel

    notes: str = ""

ClaimSupportLabel = Literal[
    "supported",
    "contradicted",
    "not_supported",
    "uncertain",
]


class ClaimReview(BaseModel):
    question_id: str = Field(min_length=1)
    claim_id: str = Field(min_length=1)
    claim: str = Field(min_length=1)

    cited_evidence_ids: list[str] = Field(default_factory=list)

    support_label: ClaimSupportLabel
    notes: str = ""

class AnswerabilityReview(BaseModel):
    question_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)

    answerability: Literal[
        "answerable",
        "likely_unanswerable",
        "unanswerable",
        "uncertain",
    ]

    response_type: Literal[
        "answered",
        "abstained",
        "partial",
    ]

    abstention_appropriate: Literal[
        "yes",
        "no",
        "likely",
        "uncertain",
    ]

    review_method: str = Field(min_length=1)
    notes: str = ""


class QuestionCorrectnessReview(BaseModel):
    question_id: str = Field(min_length=1)
    correctness: ReviewLabel

    run_id: str | None = None
    condition: str | None = None

    notes: str = ""
    answer_sha256: str | None = None