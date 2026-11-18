
from evaluation.sufficiency import evaluate_evidence


def test_alternative_sufficient_evidence():
    result = evaluate_evidence(
        retrieved_ids=["page16_chunk1"],
        reference_ids=["page13_chunk2", "page13_chunk3"],
        sufficient_groups=[
            ["page16_chunk1"],
            ["page13_chunk2"],
            ["page13_chunk3"],
        ],
    )

    assert result["reference_recall"] == 0.0
    assert result["evidence_sufficient"] is True


def test_missing_sufficiency_labels():
    result = evaluate_evidence(
        retrieved_ids=["page7_chunk1"],
        reference_ids=["page7_chunk1"],
    )

    assert result["reference_recall"] == 1.0
    assert result["evidence_sufficient"] is None
