
import json

from evaluation import runner


def test_run_evaluation(tmp_path, monkeypatch):
    input_file = tmp_path / "questions.jsonl"
    output_file = tmp_path / "results.jsonl"

    input_file.write_text(
        json.dumps({
            "question_id": "q1",
            "document_id": "paper1",
            "question": "What is the objective?",
        }) + "\n",
        encoding="utf-8",
    )

    def fake_run_one(question, api_url, max_tokens=256):
        assert max_tokens == 256
        from evaluation.schema import EvaluationRun

        return EvaluationRun(
            question_id=question.question_id,
            document_id=question.document_id,
            answer="The objective balances fidelity and fairness [E1].",
            retrieval_policy="keep",
            predicted_delta=0.015,
            retrieved_evidence_ids=["page7_chunk1"],
            cited_evidence_ids=["E1"],
            citation_ids_valid=True,
            retrieval_latency_ms=500,
            generation_latency_ms=10000,
            pipeline_latency_ms=10500,
        )

    monkeypatch.setattr(runner, "run_one", fake_run_one)

    completed = runner.run_evaluation(
        input_file,
        output_file,
    )

    assert completed == 1

    saved = [
        json.loads(line)
        for line in output_file.read_text(
            encoding="utf-8"
        ).splitlines()
    ]

    assert saved[0]["question_id"] == "q1"
    assert saved[0]["retrieval_policy"] == "keep"
    assert saved[0]["cited_evidence_ids"] == ["E1"]
