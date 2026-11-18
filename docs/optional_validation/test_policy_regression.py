
"""Regression tests against frozen Step 71B predictions.

Only NO_GOLD files are used. No model fitting or retrieval.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.policy_router import FrozenPolicyRouter


ROOT = Path(__file__).resolve().parents[1]

ARTIFACT_DIR = (
    ROOT / "data/qasper/step66_new_300/development"
    / "retrieval_step71a_frozen"
)

PREDICTION_DIR = (
    ROOT / "data/qasper/step66_new_300/heldout_validation"
    / "retrieval_step71b_prediction_only"
)

KEYS = ["document_id", "question_id"]


def load_frozen_data():
    features = pd.read_csv(
        PREDICTION_DIR / "runtime_features_ONLY.csv"
    )

    decisions = pd.read_csv(
        PREDICTION_DIR / "frozen_routing_decisions_NO_GOLD.csv"
    )

    # Ensure one row per question and no accidental many-to-many join.
    assert not features.duplicated(KEYS).any()
    assert not decisions.duplicated(KEYS).any()

    merged = features.merge(
        decisions,
        on=KEYS,
        how="inner",
        validate="one_to_one",
    )

    assert len(features) == len(decisions) == len(merged)
    return merged


def test_heldout_predictions_match():
    router = FrozenPolicyRouter(ARTIFACT_DIR)
    data = load_frozen_data()

    assert len(data) == 188

    # Recreate production decisions from saved baseline features.
    actual = [
        router.predict(row)
        for row in data[list(router.features)].to_dict("records")
    ]

    actual_scores = np.array(
        [result.predicted_delta for result in actual]
    )
    actual_triggers = np.array(
        [result.action == "hybrid" for result in actual]
    )

    expected_scores = data["predicted_delta_hit"].to_numpy(
        dtype=float
    )

    # CSV serialization may introduce tiny floating-point differences.
    np.testing.assert_allclose(
        actual_scores,
        expected_scores,
        rtol=1e-10,
        atol=1e-10,
    )

    # Normalize original Boolean flags.
    expected_triggers = (
        data["trigger_hybrid"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "1.0": True,
            "0.0": False,
        })
    )

    assert expected_triggers.notna().all()

    np.testing.assert_array_equal(
        actual_triggers,
        expected_triggers.to_numpy(dtype=bool),
    )

    assert int(actual_triggers.sum()) == 46

    print(
        f"\nVerified {len(data)} frozen predictions, "
        f"{actual_triggers.sum()} hybrid calls, "
        "100% decision agreement."
    )


def test_no_gold_features_used():
    router = FrozenPolicyRouter(ARTIFACT_DIR)

    assert len(router.features) == 6
    assert not any(
        "gold" in name.lower() or "label" in name.lower()
        for name in router.features
    )
