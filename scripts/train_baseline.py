#!/usr/bin/env python3
"""Train + cross-validate the reference model (owner: B).

    python scripts/train_baseline.py                 # full data
    python scripts/train_baseline.py --neg-frac 0.3  # fast phase-1 run

Writes artifacts/model_bundle.joblib, results/baseline_metrics.json and appends
results/experiments.csv.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pathsnap import baseline, metrics
from pathsnap import config as psconfig
from pathsnap.config import artifacts_dir, load_config, processed_dir, results_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--neg-frac", type=float, default=1.0, help="fraction of negative rows kept in training")
    ap.add_argument("--folds", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = args.seed or int(cfg["seed"])
    train_path = processed_dir() / "features_train.parquet"
    test_path = processed_dir() / "features_student_test.parquet"
    if not train_path.exists():
        raise SystemExit("Missing features. Run scripts/build_features.py first.")

    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    train = train[train["label_arc"] >= 0].reset_index(drop=True)

    def subsample(train_part: pd.DataFrame) -> pd.DataFrame:
        """Keep all positives; thin negatives for speed (training only, never validation)."""
        if args.neg_frac >= 1.0:
            return train_part
        pos = train_part[train_part["label"]]
        neg = train_part[~train_part["label"]].sample(
            frac=args.neg_frac, random_state=seed
        )
        return pd.concat([pos, neg]).sample(frac=1.0, random_state=seed)

    folds = args.folds or min(int(cfg["cv_folds"]), train["trip_id"].nunique())
    print(f"train rows {len(train):,} | trips {train['trip_id'].nunique()} | {folds} group folds")

    oof, fold_rows = [], []
    t_cv = time.time()
    for i, (tr, va) in enumerate(
        GroupKFold(n_splits=folds).split(train, groups=train["trip_id"]), 1
    ):
        t0 = time.time()
        model = baseline.fit(subsample(train.iloc[tr]), cfg)
        scores = baseline.predict_scores(model, train.iloc[va])
        chosen = baseline.choose_one_per_observation(train.iloc[va], scores)
        oof.append(chosen)
        met = metrics.classification_metrics(
            chosen["label_arc"].to_numpy(), chosen["arc_id"].to_numpy()
        )
        fold_rows.append({"fold": i, **met, "seconds": round(time.time() - t0, 1)})
        print(f"  fold {i}: top1={met['top1_accuracy']:.3f} macroF1={met['macro_f1']:.3f}")

    oof_df = pd.concat(oof, ignore_index=True)
    cv_metrics = metrics.classification_metrics(
        oof_df["label_arc"].to_numpy(), oof_df["arc_id"].to_numpy()
    )
    cv_metrics["seconds"] = round(time.time() - t_cv, 1)
    print(f"CV: top1={cv_metrics['top1_accuracy']:.3f} macroF1={cv_metrics['macro_f1']:.3f}")

    t0 = time.time()
    final = baseline.fit(subsample(train), cfg)
    train_seconds = round(time.time() - t0, 1)
    model_path = artifacts_dir() / "model_bundle.joblib"
    baseline.save(final, cfg, model_path)

    t0 = time.perf_counter()
    scores = baseline.predict_scores(final, test)
    inference_ms = round((time.perf_counter() - t0) * 1000, 1)
    chosen = baseline.choose_one_per_observation(test, scores)
    test_metrics = metrics.classification_metrics(
        chosen["label_arc"].to_numpy(), chosen["arc_id"].to_numpy()
    )
    test_metrics["candidate_log_loss"] = metrics.candidate_log_loss(
        test["label"].to_numpy(), scores
    )
    print(
        f"student-test: top1={test_metrics['top1_accuracy']:.3f} "
        f"macroF1={test_metrics['macro_f1']:.3f} logloss={test_metrics['candidate_log_loss']:.3f}"
    )

    per_trip = []
    for tid, g in chosen.groupby("trip_id"):
        per_trip.append(
            {
                "trip_id": tid,
                **metrics.classification_metrics(
                    g["label_arc"].to_numpy(), g["arc_id"].to_numpy()
                ),
            }
        )

    payload = {
        "model": "logistic-regression-baseline",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": cfg,
        "neg_frac": args.neg_frac,
        "cv": cv_metrics,
        "cv_folds": fold_rows,
        "student_test": test_metrics,
        "student_test_per_trip": per_trip,
        "timing": {"train_seconds": train_seconds, "inference_ms": inference_ms},
    }
    with open(results_dir() / "baseline_metrics.json", "w") as fh:
        json.dump(payload, fh, indent=2)

    row = {
        "utc": payload["created_utc"],
        "model": payload["model"],
        "neg_frac": args.neg_frac,
        "cv_top1": cv_metrics["top1_accuracy"],
        "cv_macro_f1": cv_metrics["macro_f1"],
        "test_top1": test_metrics["top1_accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "candidate_log_loss": test_metrics["candidate_log_loss"],
        "train_seconds": train_seconds,
        "inference_ms": inference_ms,
    }
    exp_path = results_dir() / "experiments.csv"
    pd.DataFrame([row]).to_csv(exp_path, mode="a", header=not exp_path.exists(), index=False)

    print(f"saved {model_path}\nwrote {exp_path}")


if __name__ == "__main__":
    main()
