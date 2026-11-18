
"""Production inference wrapper for TrustRAG's frozen Ridge policy."""

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping
import hashlib
import json
import math

import joblib
import pandas as pd


DEFAULT_ARTIFACT_DIR = (
    Path(__file__).resolve().parents[1]
    / "data/qasper/step66_new_300/development"
    / "retrieval_step71a_frozen"
)


@dataclass(frozen=True)
class PolicyDecision:
    action: str
    predicted_delta: float
    threshold: float
    model_sha256: str


class FrozenPolicyRouter:
    def __init__(self, artifact_dir=DEFAULT_ARTIFACT_DIR):
        root = Path(artifact_dir)

        manifest = json.loads(
            (root / "frozen_manifest.json").read_text()
        )

        self.features = tuple(manifest["features_in_order"])
        self.threshold = float(manifest["decision_threshold"])
        self.model_sha256 = manifest["model_sha256"]

        if len(self.features) != 6 or len(set(self.features)) != 6:
            raise ValueError("Unexpected policy feature specification")

        if not math.isfinite(self.threshold):
            raise ValueError("Invalid decision threshold")

        model_path = root / manifest["model_filename"]

        actual_hash = hashlib.sha256(
            model_path.read_bytes()
        ).hexdigest()

        if actual_hash != self.model_sha256:
            raise ValueError("Frozen model SHA256 mismatch")

        # Only load trusted, locally generated joblib artifacts.
        self.model = joblib.load(model_path)

        if not hasattr(self.model, "predict"):
            raise TypeError("Frozen artifact has no predict() method")

        fitted_names = getattr(self.model, "feature_names_in_", None)

        if fitted_names is not None:
            if list(fitted_names) != list(self.features):
                raise ValueError("Model feature order mismatch")

    def predict(self, features: Mapping) -> PolicyDecision:
        missing = set(self.features) - set(features)
        extra = set(features) - set(self.features)

        if missing or extra:
            raise ValueError(
                f"Feature mismatch: missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )

        values = {}

        for name in self.features:
            raw = features[name]

            if raw is None:
                value = float("nan")
            else:
                if isinstance(raw, bool):
                    raise ValueError(f"Invalid boolean feature: {name}")
                value = float(raw)

            if math.isinf(value):
                raise ValueError(f"Infinite feature: {name}")

            values[name] = value

        # Preserve the exact training feature names and order.
        frame = pd.DataFrame([values], columns=list(self.features))

        predicted_delta = float(self.model.predict(frame)[0])

        if not math.isfinite(predicted_delta):
            raise ValueError("Model returned non-finite prediction")

        action = (
            "hybrid"
            if predicted_delta >= self.threshold
            else "keep"
        )

        return PolicyDecision(
            action=action,
            predicted_delta=predicted_delta,
            threshold=self.threshold,
            model_sha256=self.model_sha256,
        )
