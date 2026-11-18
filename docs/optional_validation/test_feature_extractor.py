
"""Compare production feature extraction to frozen research outputs."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.feature_extractor import (
    FEATURES,
    extract_runtime_features,
)


ROOT = Path(__file__).resolve().parents[1]

PREDICTION_DIR = (
    ROOT / "data/qasper/step66_new_300/heldout_validation"
    / "retrieval_step71b_prediction_only"
)

KEYS = ["question_id", "document_id"]


def test_all_heldout_features_match():
    predictions = pd.read_csv(
        PREDICTION_DIR / "predictions.csv",
        dtype={"question_id": str, "document_id": str},
    )

    expected = pd.read_csv(
        PREDICTION_DIR / "runtime_features_ONLY.csv",
        dtype={"question_id": str, "document_id": str},
    )

    actual = extract_runtime_features(predictions)

    assert len(actual) == len(expected) == 188
    assert not actual.duplicated(KEYS).any()
    assert not expected.duplicated(KEYS).any()

    merged = actual.merge(
        expected,
        on=KEYS,
        suffixes=("_new", "_original"),
        validate="one_to_one",
    )

    assert len(merged) == 188

    for feature in FEATURES:
        np.testing.assert_allclose(
            merged[f"{feature}_new"].to_numpy(dtype=float),
            merged[f"{feature}_original"].to_numpy(dtype=float),
            rtol=1e-10,
            atol=1e-10,
            equal_nan=True,
            err_msg=f"Feature mismatch: {feature}",
        )

    print(
        "\nVerified all six features "
        "for 188 held-out questions."
    )


def test_dense_rank_sentinel():
    rows = []

    # Dense ranks 1 through 15.
    for rank in range(1, 16):
        rows.append({
            "question_id": "q1",
            "document_id": "doc1",
            "method": "dense30_raw",
            "rank": rank,
            "unit_id": f"p{rank}",
            "dense_score": 1.0 / rank,
            "record_type": "paragraph",
        })

    # Reranker Top-1 was Dense rank 15.
    rows.append({
        "question_id": "q1",
        "document_id": "doc1",
        "method": "dense30_rerank",
        "rank": 1,
        "unit_id": "p15",
        "dense_score": 1.0 / 15,
        "record_type": "paragraph",
    })

    result = extract_runtime_features(pd.DataFrame(rows))

    assert result.iloc[0][
        "dense_rank_of_rerank_top1_or_11"
    ] == 11
