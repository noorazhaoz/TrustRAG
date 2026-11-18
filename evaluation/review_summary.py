
"""Summarize claim-level human reviews."""

from collections import Counter
from pathlib import Path

from evaluation.schema import ClaimReview


def summarize_claim_reviews(path):
    reviews = []

    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                reviews.append(
                    ClaimReview.model_validate_json(line)
                )

    counts = Counter(
        review.support_label for review in reviews
    )

    return {
        "total_claims": len(reviews),
        "supported": counts["supported"],
        "contradicted": counts["contradicted"],
        "not_supported": counts["not_supported"],
        "uncertain": counts["uncertain"],
    }
