"""Evaluation protocol for workstream B (see docs/plan_B_modelling.md §4).

Design rules encoded here so they cannot drift between scripts:

* every split is grouped by ``trip_id`` — no trip is ever in two folds;
* negative subsampling happens on training rows only, never on validation;
* aggregation is per observation, not per candidate row;
* Top-1 is reported twice: over all labelled observations (``top1_accuracy``)
  and over observations whose reference arc is actually in the candidate set
  (``top1_covered``). The gap is A's candidate miss rate, not B's error.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import baseline, metrics

MODEL_RUN_COLUMNS = [
    "utc",
    "family",
    "params",
    "seed",
    "neg_frac",
    "n_train_rows",
    "n_dev_trips",
    "n_holdout_trips",
    "cv_top1_mean",
    "cv_top1_std",
    "cv_macro_f1_mean",
    "cv_log_loss_mean",
    "holdout_top1",
    "holdout_top1_covered",
    "holdout_macro_f1",
    "holdout_log_loss",
    "test_top1",
    "test_top1_covered",
    "test_log_loss",
    "calibrated",
    "holdout_top1_cal",
    "holdout_log_loss_cal",
    "fit_seconds",
    "inference_ms_per_1k",
    "artifact",
    "notes",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_split(
    trip_ids,
    holdout_frac: float = 0.2,
    seed: int = 42,
    min_holdout: int = 2,
    min_dev: int = 4,
) -> dict:
    """Deterministic trip-level dev/holdout split."""
    ids = sorted({str(t) for t in trip_ids})
    n = len(ids)
    if n < min_dev + 1:
        raise ValueError(
            f"Need at least {min_dev + 1} trips to split, got {n}. "
            "Build more features or lower min_dev."
        )
    rng = np.random.default_rng(seed)
    order = [ids[i] for i in rng.permutation(n)]
    n_hold = max(min_holdout, int(round(holdout_frac * n)))
    n_hold = min(n_hold, n - min_dev)
    holdout = sorted(order[:n_hold])
    dev = sorted(order[n_hold:])
    return {"dev": dev, "holdout": holdout, "seed": seed, "holdout_frac": holdout_frac}


def load_or_make_split(path: Path, trip_ids, **kwargs) -> dict:
    """Reuse the split on disk so every family is scored on the same trips."""
    ids = sorted({str(t) for t in trip_ids})
    if path.exists():
        with open(path) as fh:
            saved = json.load(fh)
        if sorted(saved.get("dev", []) + saved.get("holdout", [])) == ids:
            return saved
    split = make_split(ids, **kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(split, fh, indent=2)
    return split


def subsample_negatives(df: pd.DataFrame, frac: float, seed: int = 0) -> pd.DataFrame:
    """Keep all positives, thin negatives. Training rows only."""
    if frac >= 1.0:
        return df
    pos = df[df["label"]]
    neg = df[~df["label"]].sample(frac=frac, random_state=seed)
    return pd.concat([pos, neg]).sample(frac=1.0, random_state=seed)


def cap_rows(df: pd.DataFrame, max_rows: int, seed: int = 0) -> pd.DataFrame:
    """Hard cap on training rows, keeping every positive."""
    if not max_rows or len(df) <= max_rows:
        return df
    pos = df[df["label"]]
    neg = df[~df["label"]]
    keep_neg = max(0, max_rows - len(pos))
    if len(neg) > keep_neg:
        neg = neg.sample(n=keep_neg, random_state=seed)
    return pd.concat([pos, neg]).sample(frac=1.0, random_state=seed)


def lattice_view(df: pd.DataFrame, k: int, require_positive: bool = True) -> pd.DataFrame:
    """Rows that a real decoder would ever see.

    ``features.py`` appends one synthetic row per observation whose reference arc
    was missed by candidate generation, tagged ``cand_rank == k``. Those rows exist
    to give training a positive example, but scoring them would let the model pick
    the true arc out of a candidate it never generated — an evaluation leak.
    Training additionally drops observations with no positive candidate, because
    such observations only teach the model to reject every arc it was shown.
    """
    out = df[df["cand_rank"] < k]
    if require_positive and len(out):
        keep = out.groupby(["trip_id", "obs_idx"], sort=False)["label"].transform("any")
        out = out[keep.to_numpy(dtype=bool)]
    return out.reset_index(drop=True)


def score_frame(df: pd.DataFrame, scores: np.ndarray, k: int) -> tuple[dict, pd.DataFrame]:
    """Observation-level metrics for a scored feature frame.

    Scoring uses the candidate lattice only (``cand_rank < k``): synthetic
    reference rows appended by the feature build are excluded, so Top-1 cannot
    benefit from an arc the candidate generator never produced.

    Returns ``(metrics_dict, chosen)`` where ``chosen`` is one row per
    ``(trip_id, obs_idx)`` — the same frame shape the API uses.
    """
    in_lattice = (df["cand_rank"].to_numpy() < k)
    lattice = df.loc[in_lattice]
    chosen = baseline.choose_one_per_observation(lattice, scores[in_lattice])
    met = metrics.classification_metrics(
        chosen["label_arc"].to_numpy(), chosen["arc_id"].to_numpy()
    )
    met["candidate_log_loss"] = metrics.candidate_log_loss(
        lattice["label"].to_numpy(), scores[in_lattice]
    )

    # Covered = the reference arc was in the candidate lattice (cand_rank < k).
    covered = (
        lattice.loc[lattice["label"].to_numpy(), ["trip_id", "obs_idx"]]
        .drop_duplicates()
        .assign(_covered=True)
    )
    merged = chosen.merge(covered, on=["trip_id", "obs_idx"], how="left")
    mask = merged["_covered"].notna().to_numpy()
    if mask.any():
        met["top1_covered"] = float(
            np.mean(
                merged.loc[mask, "label_arc"].to_numpy()
                == merged.loc[mask, "arc_id"].to_numpy()
            )
        )
    else:
        met["top1_covered"] = float("nan")
    met["n_observations"] = int(len(chosen))
    met["n_covered"] = int(mask.sum())
    met["coverage"] = float(mask.mean()) if len(chosen) else float("nan")
    if np.isfinite(met["top1_covered"]) and met["top1_accuracy"] > met["top1_covered"] + 1e-9:
        raise AssertionError(
            "Top-1 over all observations exceeds Top-1 over covered observations; "
            "a synthetic reference row is leaking into the candidate lattice."
        )
    return met, chosen


def per_trip_table(chosen: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tid, g in chosen.groupby("trip_id", sort=True):
        met = metrics.classification_metrics(
            g["label_arc"].to_numpy(), g["arc_id"].to_numpy()
        )
        rows.append({"trip_id": tid, **met})
    return pd.DataFrame(rows)


def fold_summary(fold_rows: list[dict]) -> dict:
    def agg(key):
        vals = np.array([r[key] for r in fold_rows if np.isfinite(r.get(key, np.nan))])
        return (float(vals.mean()), float(vals.std(ddof=0))) if vals.size else (float("nan"), float("nan"))

    top1_mean, top1_std = agg("top1_accuracy")
    f1_mean, _ = agg("macro_f1")
    ll_mean, _ = agg("candidate_log_loss")
    return {
        "cv_top1_mean": top1_mean,
        "cv_top1_std": top1_std,
        "cv_macro_f1_mean": f1_mean,
        "cv_log_loss_mean": ll_mean,
    }


def paired_win_rate(chosen_a: pd.DataFrame, chosen_b: pd.DataFrame) -> dict:
    """How often model A beats model B, per held-out trip (paired comparison)."""
    a = chosen_a[["trip_id", "obs_idx", "arc_id", "label_arc"]].rename(columns={"arc_id": "a"})
    b = chosen_b[["trip_id", "obs_idx", "arc_id"]].rename(columns={"arc_id": "b"})
    m = a.merge(b, on=["trip_id", "obs_idx"])
    m["hit_a"] = m["a"] == m["label_arc"]
    m["hit_b"] = m["b"] == m["label_arc"]
    per_trip = m.groupby("trip_id")[["hit_a", "hit_b"]].mean()
    wins = int((per_trip["hit_a"] > per_trip["hit_b"]).sum())
    ties = int((per_trip["hit_a"] == per_trip["hit_b"]).sum())
    return {"trips": int(len(per_trip)), "wins": wins, "ties": ties}


def append_model_run(path: Path, row: dict) -> pd.DataFrame:
    """Append one run to results/model_runs.csv without losing older columns."""
    new = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        combined = pd.concat([old, new], ignore_index=True, sort=False)
    else:
        combined = new
    ordered = MODEL_RUN_COLUMNS + [c for c in combined.columns if c not in MODEL_RUN_COLUMNS]
    combined = combined.reindex(columns=ordered)
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    return combined
