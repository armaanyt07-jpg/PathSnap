#!/usr/bin/env python3
"""Train and compare one model family under the workstream-B protocol (owner: B).

    # phase 1: reference model
    python scripts/train_models.py --family logistic --neg-frac 0.5 --set-champion

    # any other family, identical folds
    python scripts/train_models.py --family hist_gbdt --neg-frac 0.5

    # fast smoke run on the first three trips
    python scripts/train_models.py --family extra_trees --neg-frac 0.5 --smoke --trips 3

Writes artifacts/candidates/<family>.joblib, appends results/model_runs.csv, and
with --set-champion writes artifacts/model_bundle.joblib in the exact shape
src/api/main.py already loads.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pathsnap import baseline, evaluation, model_zoo
from pathsnap.config import artifacts_dir, load_config, processed_dir, results_dir
from pathsnap.features import FEATURE_COLUMNS


def _jsonable(obj):
    return json.dumps(obj, default=str)


def load_features(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(
            f"Missing {path}. Build features first:\n"
            f"  python scripts/build_features.py --splits train student_test"
        )
    df = pd.read_parquet(path)
    print(f"{label}: {len(df):,} rows, {df['trip_id'].nunique()} trips")
    return df


def train_one(args, cfg, family: str) -> dict:
    seed = args.seed or int(cfg["seed"])
    k = int(cfg["k_candidates"])
    params = dict(model_zoo.DEFAULT_PARAMS[family])
    if args.params_file:
        with open(args.params_file) as fh:
            params.update(json.load(fh))
    if args.params:
        params.update(json.loads(args.params))

    train = load_features(processed_dir() / "features_train.parquet", "train")
    if args.trips:
        keep = sorted(train["trip_id"].unique())[: args.trips]
        train = train[train["trip_id"].isin(keep)].reset_index(drop=True)
        print(f"limited to {len(keep)} trips -> {len(train):,} rows")

    train = train[train["label_arc"] >= 0].reset_index(drop=True)
    if train["label"].nunique() < 2:
        raise SystemExit("Training frame has a single class; check the label column.")

    split_name = "split_smoke.json" if args.smoke else "split.json"
    split_path = artifacts_dir() / split_name
    if args.smoke and split_path.exists():
        split_path.unlink()
    split = evaluation.load_or_make_split(
        split_path,
        train["trip_id"].unique(),
        holdout_frac=0.34 if args.smoke else 0.2,
        seed=seed,
    )
    dev_mask = train["trip_id"].isin(split["dev"])
    hold_mask = train["trip_id"].isin(split["holdout"])
    dev, holdout = train[dev_mask].reset_index(drop=True), train[hold_mask].reset_index(drop=True)
    print(
        f"split: dev {len(split['dev'])} trips / {len(dev):,} rows | "
        f"holdout {len(split['holdout'])} trips / {len(holdout):,} rows"
    )

    def training_rows(part: pd.DataFrame) -> pd.DataFrame:
        # Train only where the reference arc is in the candidate lattice: synthetic
        # rows (cand_rank == k) never exist at inference, and observations without
        # any positive candidate carry no "which arc" signal. Validation keeps
        # every observation, so Top-1 still counts misses as errors.
        part = evaluation.lattice_view(part, k, require_positive=True)
        part = evaluation.subsample_negatives(part, args.neg_frac, seed)
        return evaluation.cap_rows(part, args.max_train_rows, seed)

    folds = args.folds or min(int(cfg["cv_folds"]), dev["trip_id"].nunique())
    folds = max(2, min(folds, dev["trip_id"].nunique()))

    oof_scores = np.full(len(dev), np.nan)
    fold_rows = []
    t_cv = time.time()
    for i, (tr, va) in enumerate(GroupKFold(n_splits=folds).split(dev, groups=dev["trip_id"]), 1):
        t0 = time.time()
        train_part = training_rows(dev.iloc[tr])
        if i == 1:
            print(f"  fold 1 training rows: {len(train_part):,} (lattice + neg_frac)")
        model = model_zoo.fit(family, train_part, params, seed)
        scores = baseline.predict_scores(model, dev.iloc[va])
        oof_scores[va] = scores
        met, _ = evaluation.score_frame(dev.iloc[va], scores, k)
        fold_rows.append({"fold": i, **met, "seconds": round(time.time() - t0, 1)})
        print(
            f"  fold {i}/{folds}: top1={met['top1_accuracy']:.4f} "
            f"covered={met['top1_covered']:.4f} logloss={met['candidate_log_loss']:.4f} "
            f"({fold_rows[-1]['seconds']}s)"
        )
    cv = evaluation.fold_summary(fold_rows)
    cv["seconds"] = round(time.time() - t_cv, 1)
    print(
        f"CV: top1={cv['cv_top1_mean']:.4f}±{cv['cv_top1_std']:.4f} "
        f"macroF1={cv['cv_macro_f1_mean']:.4f} logloss={cv['cv_log_loss_mean']:.4f}"
    )

    final_train = training_rows(dev)
    t0 = time.time()
    final = model_zoo.fit(family, final_train, params, seed)
    fit_seconds = round(time.time() - t0, 1)

    t0 = time.perf_counter()
    hold_scores = baseline.predict_scores(final, holdout)
    inference_ms = round((time.perf_counter() - t0) * 1000, 2)
    hold_met, hold_chosen = evaluation.score_frame(holdout, hold_scores, k)
    print(
        f"holdout: top1={hold_met['top1_accuracy']:.4f} "
        f"covered={hold_met['top1_covered']:.4f} logloss={hold_met['candidate_log_loss']:.4f}"
    )

    # Calibration is fitted on out-of-fold dev scores only.
    cal_met = {}
    champion_model = final
    if args.calibrate:
        ok = np.isfinite(oof_scores)
        iso = model_zoo.fit_calibrator(oof_scores[ok], dev.loc[ok, "label"].to_numpy())
        calibrated = model_zoo.CalibratedScorer(final, iso)
        cal_scores = baseline.predict_scores(calibrated, holdout)
        cal_met, _ = evaluation.score_frame(holdout, cal_scores, k)
        champion_model = calibrated
        print(
            f"holdout (calibrated): top1={cal_met['top1_accuracy']:.4f} "
            f"logloss={cal_met['candidate_log_loss']:.4f}"
        )
        if abs(cal_met["top1_accuracy"] - hold_met["top1_accuracy"]) > 1e-9:
            print("  warning: calibration changed argmax; isotonic should be monotone")

    test_path = processed_dir() / "features_student_test.parquet"
    test_met = {}
    if test_path.exists():
        test = pd.read_parquet(test_path)
        scores = baseline.predict_scores(champion_model, test)
        test_met, _ = evaluation.score_frame(test, scores, k)
        print(
            f"student-test: top1={test_met['top1_accuracy']:.4f} "
            f"covered={test_met['top1_covered']:.4f} logloss={test_met['candidate_log_loss']:.4f}"
        )

    cand_dir = artifacts_dir() / "candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)
    artifact = cand_dir / f"{family}.joblib"
    joblib.dump(champion_model, artifact)
    meta = {
        "family": family,
        "params": params,
        "seed": seed,
        "neg_frac": args.neg_frac,
        "calibrated": bool(args.calibrate),
        "features": FEATURE_COLUMNS,
        "k_candidates": k,
        "cv": cv,
        "holdout": hold_met,
        "holdout_calibrated": cal_met,
        "student_test": test_met,
        "created_utc": evaluation.utc_now(),
    }
    with open(cand_dir / f"{family}_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2, default=str)

    if args.set_champion:
        joblib.dump(champion_model, artifacts_dir() / "model_bundle.joblib")
        with open(artifacts_dir() / "model_config.json", "w") as fh:
            json.dump(
                {
                    "features": FEATURE_COLUMNS,
                    "family": family,
                    "calibrated": bool(args.calibrate),
                    "params": params,
                    "k_candidates": k,
                    "model": params,  # key kept for baseline.save compatibility
                },
                fh,
                indent=2,
                default=str,
            )
        print(f"champion -> {artifacts_dir() / 'model_bundle.joblib'}")

    row = {
        "utc": evaluation.utc_now(),
        "family": family,
        "params": _jsonable(params),
        "seed": seed,
        "neg_frac": args.neg_frac,
        "n_train_rows": int(len(final_train)),
        "n_dev_trips": len(split["dev"]),
        "n_holdout_trips": len(split["holdout"]),
        **cv,
        "holdout_top1": hold_met["top1_accuracy"],
        "holdout_top1_covered": hold_met["top1_covered"],
        "holdout_macro_f1": hold_met["macro_f1"],
        "holdout_log_loss": hold_met["candidate_log_loss"],
        "test_top1": test_met.get("top1_accuracy", float("nan")),
        "test_top1_covered": test_met.get("top1_covered", float("nan")),
        "test_log_loss": test_met.get("candidate_log_loss", float("nan")),
        "calibrated": bool(args.calibrate),
        "holdout_top1_cal": cal_met.get("top1_accuracy", float("nan")),
        "holdout_log_loss_cal": cal_met.get("candidate_log_loss", float("nan")),
        "fit_seconds": fit_seconds,
        "inference_ms_per_1k": round(inference_ms / max(len(holdout), 1) * 1000, 2),
        "artifact": str(artifact.relative_to(artifacts_dir().parent)),
        "notes": "smoke" if args.smoke else "",
    }
    runs_path = results_dir() / "model_runs.csv"
    evaluation.append_model_run(runs_path, row)
    print(f"appended {runs_path}")
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="logistic", help=f"one of {', '.join(model_zoo.FAMILIES)}, or 'all'")
    ap.add_argument("--config", default=None)
    ap.add_argument("--neg-frac", type=float, default=0.5)
    ap.add_argument("--folds", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-train-rows", type=int, default=0)
    ap.add_argument("--trips", type=int, default=0, help="limit to first N train trips")
    ap.add_argument("--smoke", action="store_true", help="fresh split, few trips")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--set-champion", action="store_true")
    ap.add_argument("--params", default=None, help="JSON overrides")
    ap.add_argument("--params-file", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    families = list(model_zoo.FAMILIES) if args.family == "all" else [args.family]
    rows = [train_one(args, cfg, fam) for fam in families]

    if len(rows) > 1:
        table = pd.DataFrame(rows)[["family", "cv_top1_mean", "holdout_top1", "test_top1"]]
        print("\n" + table.to_string(index=False))


if __name__ == "__main__":
    main()
