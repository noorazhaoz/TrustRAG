
"""Baseline-only feature extraction for the frozen TrustRAG policy.

Preserves the Step 69 runtime feature definitions.
No gold labels or hybrid results are required.
"""

import numpy as np
import pandas as pd


RAW = "dense30_raw"
BASE = "dense30_rerank"

FEATURES = [
    "dense_top1_score",
    "dense_top2_margin",
    "dense_top5_mean",
    "dense_rerank_top10_overlap",
    "dense_rank_of_rerank_top1_or_11",
    "baseline_float_fraction",
]

REQUIRED_COLUMNS = {
    "question_id",
    "document_id",
    "method",
    "rank",
    "unit_id",
    "dense_score",
    "record_type",
}


def extract_runtime_features(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return Step 69-compatible features from retrieval rankings."""

    missing = REQUIRED_COLUMNS - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing retrieval columns: {sorted(missing)}")

    pred = predictions.copy()
    pred["question_id"] = pred["question_id"].astype(str)
    pred["document_id"] = pred["document_id"].astype(str)

    if pred.duplicated(
        ["question_id", "document_id", "method", "rank"]
    ).any():
        raise ValueError("Duplicate retrieval ranks detected")

    grouped = {}

    for (qid, doc, method), group in pred.groupby(
        ["question_id", "document_id", "method"],
        sort=False,
    ):
        if method in (RAW, BASE):
            grouped.setdefault((str(qid), str(doc)), {})[
                method
            ] = group.sort_values("rank")

    rows = []

    for (qid, doc), methods in grouped.items():
        if RAW not in methods or BASE not in methods:
            continue

        a = methods[RAW]
        b = methods[BASE]

        scores = pd.to_numeric(
            a["dense_score"], errors="coerce"
        ).to_numpy(dtype=float)

        dense_ids = a["unit_id"].astype(str).tolist()
        rerank_ids = b["unit_id"].astype(str).tolist()

        rows.append({
            "question_id": qid,
            "document_id": doc,
            "dense_top1_score": (
                scores[0] if len(scores) else np.nan
            ),
            "dense_top2_margin": (
                scores[0] - scores[1]
                if len(scores) > 1 else np.nan
            ),
            "dense_top5_mean": (
                float(np.nanmean(scores[:5]))
                if len(scores) else np.nan
            ),
            "dense_rerank_top10_overlap": (
                len(
                    set(dense_ids[:10]) & set(rerank_ids[:10])
                ) / 10
            ),
            "dense_rank_of_rerank_top1_or_11": (
                dense_ids[:10].index(rerank_ids[0]) + 1
                if rerank_ids and rerank_ids[0] in dense_ids[:10]
                else 11
            ),
            "baseline_float_fraction": float(
                b.head(10)["record_type"]
                .eq("float_caption")
                .mean()
            ),
        })

    return pd.DataFrame(
        rows,
        columns=["question_id", "document_id", *FEATURES],
    )
