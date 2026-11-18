#!/usr/bin/env python3
"""Optuna tuning for workstream B (owner: B). Post-demo work.

    python scripts/tune_optuna.py --family hist_gbdt --trials 60 --timeout 720

Objective: mean candidate log loss over grouped inner folds on the dev trips.
Log loss, not Top-1: Top-1 is a step function of the scores and gives TPE almost
no signal, while log loss is exactly what C's Viterbi consumes.

Studies persist to artifacts/optuna_<family>.db, so a crash costs a restart and
not the run. Best parameters land in artifacts/best_params_<family>.json, ready
for:  python scripts/train_models.py --family <family> --params-file <that file>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pathsnap import baseline, evaluation, metrics, model_zoo
from pathsnap.config import artifacts_dir, load_config, processed_dir


def suggest_params(family: str, trial) -> dict:
    if family == "logistic":
        return {
            "C": trial.suggest_float("C", 1e-3, 30.0, log=True),
            "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
        }
    if family == "extra_trees":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
            "max_depth": trial.suggest_categorical("max_depth", [None, 10, 16, 24, 32]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50, log=True),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", 0.3, 0.6]),
            "class_weight": trial.suggest_categorical(
                "class_weight", ["balanced_subsample", "balanced"]
            ),
        }
    if family == "hist_gbdt":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "max_iter": trial.suggest_int("max_iter", 100, 400, step=50),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127, log=True),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 200, log=True),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
            "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
        }
    if family == "nystroem_sgd":
        return {
            "n_components": trial.suggest_int("n_components", 32, 128, step=32),
            "gamma": trial.suggest_float("gamma", 1e-3, 1.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-6, 1e-2, log=True),
            "average": trial.suggest_categorical("average", [True, False]),
        }
    if family == "mlp":
        hidden = trial.suggest_categorical("hidden", ["32", "64x32", "128x64"])
        return {
            "hidden_layer_sizes": [int(v) for v in hidden.split("x")],
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-2, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
        }
    if family == "sgd_huber":
        return {
            "alpha": trial.suggest_float("alpha", 1e-6, 1e-2, log=True),
            "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
        }
    raise KeyError(f"No search space for '{family}'")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--timeout", type=int, default=0, help="seconds; 0 = no limit")
    ap.add_argument("--neg-frac", type=float, default=0.5)
    ap.add_argument("--max-train-rows", type=int, default=400_000, help="per fold; 0 = all")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    try:
        import optuna
    except ImportError:
        raise SystemExit(
            "optuna is not installed. Install it first:\n"
            "  .venv/bin/pip install optuna\n"
            "  (then add 'optuna' to requirements.txt)"
        )
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    cfg = load_config(args.config)
    seed = args.seed or int(cfg["seed"])
    k = int(cfg["k_candidates"])

    train_path = processed_dir() / "features_train.parquet"
    if not train_path.exists():
        raise SystemExit("Missing features. Run scripts/build_features.py first.")
    train = pd.read_parquet(train_path)
    train = train[train["label_arc"] >= 0].reset_index(drop=True)
    train = evaluation.lattice_view(train, k, require_positive=True)

    split = evaluation.load_or_make_split(
        artifacts_dir() / "split.json", train["trip_id"].unique(), seed=seed
    )
    dev = train[train["trip_id"].isin(split["dev"])].reset_index(drop=True)
    folds = max(2, min(args.folds, dev["trip_id"].nunique()))
    print(f"tuning {args.family} on {folds} grouped folds, {len(dev):,} dev rows")

    def training_rows(part: pd.DataFrame) -> pd.DataFrame:
        part = evaluation.subsample_negatives(part, args.neg_frac, seed)
        return evaluation.cap_rows(part, args.max_train_rows, seed)

    def objective(trial):
        params = suggest_params(args.family, trial)
        losses = []
        for step, (tr, va) in enumerate(
            GroupKFold(n_splits=folds).split(dev, groups=dev["trip_id"])
        ):
            model = model_zoo.fit(args.family, training_rows(dev.iloc[tr]), params, seed)
            scores = baseline.predict_scores(model, dev.iloc[va])
            losses.append(
                metrics.candidate_log_loss(dev.iloc[va]["label"].to_numpy(), scores)
            )
            trial.report(float(np.mean(losses)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(losses))

    db_path = artifacts_dir() / f"optuna_{args.family}.db"
    study = optuna.create_study(
        study_name=args.family,
        direction="minimize",
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
    )
    study.optimize(
        objective,
        n_trials=args.trials,
        timeout=args.timeout or None,
        show_progress_bar=False,
    )

    best = dict(study.best_params)
    if args.family == "mlp":
        best["hidden_layer_sizes"] = [int(v) for v in best.pop("hidden").split("x")]
    payload = {
        "family": args.family,
        "best_params": best,
        "best_value_log_loss": float(study.best_value),
        "n_trials": len(study.trials),
        "n_pruned": sum(t.state.name == "PRUNED" for t in study.trials),
    }
    out = artifacts_dir() / f"best_params_{args.family}.json"
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(json.dumps(payload, indent=2, default=str))
    print(f"\nwrote {out}\nnext: python scripts/train_models.py --family {args.family} --params-file {out}")


if __name__ == "__main__":
    main()
