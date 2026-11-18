"""Reference model bundle (owner: B). Logistic regression on the contract features.

This is a plumbing baseline, not the final model: it proves the feature table is
learnable and gives C something with probabilities to decode on top of.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS


def build_model(cfg: dict) -> Pipeline:
    m = cfg.get("model", {})
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=float(m.get("C", 1.0)),
                    max_iter=int(m.get("max_iter", 400)),
                    class_weight=m.get("class_weight", "balanced"),
                    n_jobs=None,
                ),
            ),
        ]
    )


def fit(df: pd.DataFrame, cfg: dict) -> Pipeline:
    import warnings

    model = build_model(cfg)
    with warnings.catch_warnings():
        # NumPy 2 + sklearn emits a cosmetic "invalid value in matmul" warning while
        # the solver probes step sizes; coefficients and scores are verified finite.
        warnings.filterwarnings("ignore", message=".*in matmul", category=RuntimeWarning)
        model.fit(
            df[FEATURE_COLUMNS].to_numpy(dtype=np.float32),
            df["label"].to_numpy(dtype=np.int8),
        )
    if not np.isfinite(model[-1].coef_).all():
        raise RuntimeError("Model fit diverged (non-finite coefficients)")
    return model


def predict_scores(model: Pipeline, df: pd.DataFrame) -> np.ndarray:
    import warnings

    with warnings.catch_warnings():
        # Same cosmetic NumPy 2 matmul warning as in fit(); outputs are finite.
        warnings.filterwarnings("ignore", message=".*in matmul", category=RuntimeWarning)
        columns = getattr(model, "feature_columns", FEATURE_COLUMNS)
        scores = model.predict_proba(df[list(columns)].to_numpy(dtype=np.float32))[:, 1]
    if not np.isfinite(scores).all():
        raise RuntimeError("Non-finite prediction scores")
    return scores


def choose_one_per_observation(df: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    """Argmax over candidates, one row per (trip_id, obs_idx)."""
    out = df[["trip_id", "obs_idx", "arc_id", "label_arc", "label"]].copy()
    out["score"] = scores
    out = out.sort_values(["trip_id", "obs_idx", "score"], ascending=[True, True, False])
    out = out.drop_duplicates(["trip_id", "obs_idx"], keep="first")
    return out.sort_values(["trip_id", "obs_idx"]).reset_index(drop=True)


def save(model: Pipeline, cfg: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    with open(path.with_name("model_config.json"), "w") as fh:
        json.dump(
            {
                "features": FEATURE_COLUMNS,
                "model": cfg.get("model", {}),
                "k_candidates": cfg.get("k_candidates"),
            },
            fh,
            indent=2,
        )


def load(path: Path) -> Pipeline:
    return joblib.load(path)
