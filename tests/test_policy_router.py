
import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

from src.policy_router import FrozenPolicyRouter


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = (
    ROOT / "data/qasper/step66_new_300/development"
    / "retrieval_step71a_frozen"
)


@pytest.fixture(scope="module")
def router():
    return FrozenPolicyRouter(ARTIFACT_DIR)


def sample_features(router):
    return dict.fromkeys(router.features, 0.0)


def test_feature_order(router):
    manifest = json.loads(
        (ARTIFACT_DIR / "frozen_manifest.json").read_text()
    )
    assert list(router.features) == manifest["features_in_order"]


def test_prediction_matches_original_pipeline(router):
    features = sample_features(router)

    original_model = joblib.load(
        ARTIFACT_DIR / "net_benefit_ridge.joblib"
    )

    frame = pd.DataFrame(
        [features], columns=list(router.features)
    )

    original_prediction = float(
        original_model.predict(frame)[0]
    )

    result = router.predict(features)

    assert result.predicted_delta == pytest.approx(
        original_prediction, abs=1e-12
    )

    expected_action = (
        "hybrid"
        if original_prediction >= router.threshold
        else "keep"
    )

    assert result.action == expected_action


def test_missing_feature_rejected(router):
    features = sample_features(router)
    features.pop(router.features[0])

    with pytest.raises(ValueError):
        router.predict(features)


def test_extra_feature_rejected(router):
    features = sample_features(router)
    features["gold_hit"] = 1

    with pytest.raises(ValueError):
        router.predict(features)


def test_deterministic_prediction(router):
    features = sample_features(router)

    assert router.predict(features) == router.predict(features)
