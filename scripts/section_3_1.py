#!/usr/bin/env python3
"""Fill the report's §3.1 consolidated model-comparison table from real results.

Every value is recomputed from the saved artifacts rather than copied by hand:

* CV macro F1, configuration and inference cost come from ``results/model_runs.csv``
  (latest non-smoke run per family).
* Top-1 accuracy and log loss are recomputed on the frozen holdout split, scoring
  the promoted/saved bundle on ``features_train.parquet`` and applying the same
  candidate-lattice rule as evaluation.
* Student-test macro F1 is recomputed on ``features_student_test.parquet``.
* Route continuity is measured by decoding each student-test trip through the
  exact API path (``api.service.predict_trip``), averaging over trips.

Writes ``results/section_3_1_table.csv`` and ``results/section_3_1_table.md``.

    PYTHONPATH=src python scripts/section_3_1.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from api.service import predict_trip
from pathsnap import baseline, config, evaluation, io

MODEL_LABELS = {
    "hist_gbdt": "Hist GBDT",
    "extra_trees": "Extra Trees",
    "mlp": "MLP",
    "nystroem_sgd": "Nystroem SGD",
    "logistic": "Logistic",
    "sgd_huber": "SGD Huber",
}
FAMILY_KIND = {
    "hist_gbdt": "Gradient-boosted trees",
    "extra_trees": "Extremely randomised trees",
    "mlp": "Feed-forward neural net",
    "nystroem_sgd": "Nystroem RBF + SGD",
    "logistic": "L2 logistic regression",
    "sgd_huber": "Linear SGD (Huber)",
}
PARAM_ABBREV = {
    "learning_rate": "lr",
    "max_iter": "iters",
    "max_leaf_nodes": "leaves",
    "min_samples_leaf": "min_leaf",
    "l2_regularization": "l2",
    "class_weight": "weight",
    "n_estimators": "trees",
    "max_depth": "depth",
    "max_features": "feat",
    "hidden_layer_sizes": "hidden",
    "learning_rate_init": "lr0",
    "batch_size": "batch",
    "n_components": "components",
    "average": "avg",
}


def latest_runs(path: Path) -> pd.DataFrame:
    """Latest non-smoke run per family — same rule as the report figures."""
    runs = pd.read_csv(path)
    runs["notes"] = runs.get("notes", pd.Series(index=runs.index, dtype=object)).fillna("")
    runs = runs[(runs["notes"] != "smoke") & runs["holdout_top1"].notna()]
    if runs.empty:
        raise SystemExit(f"No completed runs in {path}")
    return runs.sort_values("utc").groupby("family", as_index=False).tail(1).reset_index(drop=True)


def _short_params(raw: str) -> str:
    try:
        params = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return str(raw)
    parts = []
    for key, value in params.items():
        name = PARAM_ABBREV.get(key, key)
        if isinstance(value, list):
            value = "-".join(str(v) for v in value)
        parts.append(f"{name}={value}")
    return ", ".join(parts)


def feature_frame(split_parquet: Path, trip_ids: set[str] | None) -> pd.DataFrame:
    from pathsnap.features import FEATURE_COLUMNS

    cols = list(dict.fromkeys(["trip_id", "obs_idx", "arc_id", "label", "label_arc", *FEATURE_COLUMNS]))
    frame = pd.read_parquet(split_parquet, columns=cols)
    if trip_ids is not None:
        frame = frame[frame["trip_id"].isin(trip_ids)].reset_index(drop=True)
    return frame


def partition_metrics(runs: pd.DataFrame, frame: pd.DataFrame, k: int) -> dict[str, dict]:
    rows = {}
    for row in runs.itertuples(index=False):
        model = joblib.load(ROOT / str(row.artifact))
        scores = baseline.predict_scores(model, frame)
        met, _ = evaluation.score_frame(frame, scores, k)
        rows[row.family] = met
    return rows


def continuity_means(runs: pd.DataFrame, cfg: dict) -> dict[str, float]:
    """Mean route continuity per family over the student-test trips."""
    trip_ids = io.trip_ids(cfg["splits"]["student_test"])
    models = {row.family: joblib.load(ROOT / str(row.artifact)) for row in runs.itertuples(index=False)}
    per_family: dict[str, list[float]] = {family: [] for family in models}
    for position, trip_id in enumerate(trip_ids, start=1):
        trip = io.load_trip(cfg["splits"]["student_test"], trip_id)
        for family, model in models.items():
            out = predict_trip(trip, model, cfg)
            value = out["metrics"].get("route_continuity")
            if value is not None and np.isfinite(value):
                per_family[family].append(float(value))
        print(f"  decoded trip {position}/{len(trip_ids)}")
    return {family: float(np.mean(values)) if values else float("nan") for family, values in per_family.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(config.results_dir() / "model_runs.csv"))
    ap.add_argument("--out", default=str(config.results_dir()))
    ap.add_argument("--skip-continuity", action="store_true", help="reuse results/section_3_1_table.csv values")
    args = ap.parse_args()

    cfg = config.load_config()
    k = int(cfg["k_candidates"])
    runs = latest_runs(Path(args.runs))

    with open(config.artifacts_dir() / "split.json") as fh:
        holdout_ids = set(json.load(fh)["holdout"])
    holdout = feature_frame(config.processed_dir() / "features_train.parquet", holdout_ids)
    test = feature_frame(config.processed_dir() / "features_student_test.parquet", None)
    print(f"  holdout {len(holdout):,} rows / test {len(test):,} rows")

    print("  scoring holdout...")
    holdout_met = partition_metrics(runs, holdout, k)
    print("  scoring student test...")
    test_met = partition_metrics(runs, test, k)

    if args.skip_continuity:
        table = pd.read_csv(Path(args.out) / "section_3_1_table.csv")
        continuity = dict(zip(table["model_key"], table["route_continuity"]))
    else:
        print("  decoding student-test trips per family...")
        continuity = continuity_means(runs, cfg)

    rows = []
    for row in runs.itertuples(index=False):
        family = row.family
        label = MODEL_LABELS.get(family, family)
        if family == "hist_gbdt":
            label += " (champion)"
        rows.append(
            {
                "model_key": family,
                "model": label,
                "algorithm_family": FAMILY_KIND.get(family, family),
                "key_configuration": _short_params(row.params),
                "cv_macro_f1": float(row.cv_macro_f1_mean),
                "holdout_top1": float(holdout_met[family]["top1_accuracy"]),
                "holdout_top1_covered": float(holdout_met[family]["top1_covered"]),
                "route_continuity": float(continuity.get(family, float("nan"))),
                "student_test_macro_f1": float(test_met[family]["macro_f1"]),
                "holdout_log_loss": float(holdout_met[family]["candidate_log_loss"]),
                "inference_ms_per_1k": float(row.inference_ms_per_1k),
            }
        )
    table = pd.DataFrame(rows).sort_values("holdout_top1", ascending=False).reset_index(drop=True)
    table.insert(0, "rank", range(1, len(table) + 1))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "section_3_1_table.csv", index=False)

    def fmt(value: float, digits: int = 3) -> str:
        return "n/a" if not np.isfinite(value) else f"{value:.{digits}f}"

    header = (
        "| # | Model | Algorithm family | Key configuration | CV macro F1 | Top-1 acc. (holdout) "
        "| Route continuity | Student-test macro F1 | Log loss (holdout) | Inference |\n"
        "|---|-------|------------------|-------------------|-------------|----------------------"
        "|------------------|----------------------|--------------------|-----------|\n"
    )
    lines = [
        f"| {r.rank} | {r.model} | {r.algorithm_family} | `{r.key_configuration}` "
        f"| {fmt(r.cv_macro_f1)} | {fmt(r.holdout_top1)} | {fmt(r.route_continuity)} "
        f"| {fmt(r.student_test_macro_f1)} | {fmt(r.holdout_log_loss)} "
        f"| {fmt(r.inference_ms_per_1k, 2)} ms / 1k rows |"
        for r in table.itertuples(index=False)
    ]
    (out_dir / "section_3_1_table.md").write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
    print(table.to_string(index=False))
    print(f"\nwrote {out_dir / 'section_3_1_table.csv'} and .md")


if __name__ == "__main__":
    main()
