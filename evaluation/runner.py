
"""Run TrustRAG evaluations through the existing API."""

import json
from pathlib import Path
from urllib.request import Request, urlopen

from evaluation.schema import EvaluationQuestion, EvaluationRun


def run_one(
    question: EvaluationQuestion,
    api_url: str,
    max_tokens: int = 256,
):
    """Run one question through POST /answer."""

    payload = {
        "question_id": question.question_id,
        "document_id": question.document_id,
        "question": question.question,
        "max_tokens": max_tokens,
    }

    request = Request(
        f"{api_url.rstrip('/')}/answer",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(request, timeout=180) as response:
        result = json.load(response)

    return EvaluationRun(
        question_id=result["question_id"],
        document_id=result["document_id"],
        answer=result["answer"],
        retrieval_policy=result["retrieval_policy"],
        predicted_delta=result["predicted_delta"],
        retrieved_evidence_ids=[
            item["unit_id"]
            for item in result["evidence"]
        ],
        cited_evidence_ids=[
            citation.strip("[]")
            for citation in result["citation_validation"][
                "used_citations"
            ]
        ],
        citation_ids_valid=result["citation_validation"]["valid"],
        retrieval_latency_ms=result["retrieval_latency_ms"],
        generation_latency_ms=result["generation_latency_ms"],
        pipeline_latency_ms=result["pipeline_latency_ms"],
        max_tokens=result.get("max_tokens"),
        generation_tokens=result.get("generation_tokens"),
        finish_reason=result.get("finish_reason"),

        citation_sources={
            item["citation_id"]: item["unit_id"]
            for item in result["evidence"]
            if "citation_id" in item
        },
    )


def run_evaluation(
    input_path,
    output_path,
    api_url="http://127.0.0.1:8000",
    max_tokens=256,
):
    """Evaluate questions sequentially and save each result."""

    input_path = Path(input_path)
    output_path = Path(output_path)

    if output_path.exists():
        raise FileExistsError(
            f"Output already exists: {output_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    completed = 0

    with input_path.open(encoding="utf-8") as source:
        with output_path.open("x", encoding="utf-8") as target:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue

                question = EvaluationQuestion.model_validate_json(line)

                print(
                    f"Running {question.question_id} "
                    f"(line {line_number})...",
                    flush=True,
                )

                result = run_one(question, api_url, max_tokens=max_tokens)

                target.write(
                    result.model_dump_json() + "\n"
                )
                target.flush()

                completed += 1

    return completed
