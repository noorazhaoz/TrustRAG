import json

from evaluation.review_summary import summarize_claim_reviews


def test_review_summary(tmp_path):
    path = tmp_path / "reviews.jsonl"

    rows = [
        {
            "question_id": "q1",
            "claim_id": "c1",
            "claim": "Claim A",
            "support_label": "supported",
        },
        {
            "question_id": "q1",
            "claim_id": "c2",
            "claim": "Claim B",
            "support_label": "not_supported",
        },
    ]

    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    result = summarize_claim_reviews(path)

    assert result["total_claims"] == 2
    assert result["supported"] == 1
    assert result["not_supported"] == 1
    assert result["contradicted"] == 0