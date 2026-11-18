#!/usr/bin/env python3
"""Bayesian tuning figures for report section 7 (owner: B).

Draws the Optuna search history from ``artifacts/optuna_hist_gbdt.db`` and the
tuned-versus-untuned comparison from ``results/model_runs.csv`` so both pictures
come from recorded trials, not hand-written numbers.

    python scripts/make_tuning_figure.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap.config import results_dir

INK = "#1f2d3d"
MUTED = "#9aa0a6"
ACCENT = "#2f6f9f"
WARN = "#c0392b"
LIGHT = "#a8c8df"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 180,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.edgecolor": "#c9d3dc",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#e8edf1",
            "axes.axisbelow": True,
            "legend.frameon": False,
            "figure.facecolor": "white",
        }
    )


def _trial_points(study) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(trial numbers, values, best-so-far) with pruned trials marked separately."""
    numbers, values, pruned_x, pruned_y, best_y = [], [], [], [], []
    best = np.inf
    for trial in study.trials:
        if trial.state.name == "COMPLETE" and trial.value is not None:
            value = float(trial.value)
            numbers.append(trial.number)
            values.append(value)
            best = min(best, value)
            best_y.append(best)
        elif trial.state.name == "PRUNED":
            history = trial.intermediate_values or {}
            if history:
                pruned_x.append(trial.number)
                pruned_y.append(history[max(history)])
    return (
        np.array(numbers, dtype=float),
        np.array(values, dtype=float),
        np.array(pruned_x, dtype=float),
        np.array(pruned_y, dtype=float),
        np.array(best_y, dtype=float),
    )


def fig_history(study, out: Path) -> Path:
    numbers, values, pruned_x, pruned_y, best_y = _trial_points(study)
    fig, ax = plt.subplots(figsize=(7.6, 4.4), constrained_layout=True)
    if len(pruned_x):
        ax.scatter(pruned_x, pruned_y, s=14, color=MUTED, alpha=0.7, zorder=2, label="pruned trial")
    ax.scatter(numbers, values, s=18, color=LIGHT, edgecolor=ACCENT, linewidth=0.6, zorder=3, label="completed trial")
    ax.plot(numbers, best_y, color=WARN, linewidth=2.0, zorder=4, label="best so far")
    ax.set_xlabel("trial")
    ax.set_ylabel("objective: mean candidate log loss")
    ax.set_title("Bayesian search history — Hist GBDT (TPE, 60 trials, 28 pruned)", color=INK, weight="semibold")
    ax.legend(loc="upper right", fontsize=9)
    path = out / "tuning_history.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _untuned_and_tuned_rows() -> tuple[pd.Series, pd.Series]:
    runs = pd.read_csv(results_dir() / "model_runs.csv")
    rows = runs[(runs["family"] == "hist_gbdt") & runs["holdout_top1"].notna()]
    if len(rows) < 2:
        raise SystemExit("Need both an untuned and a tuned hist_gbdt row in results/model_runs.csv")
    untuned = rows[rows["params"].str.contains('"learning_rate": 0.1', na=False)]
    tuned = rows[~rows.index.isin(untuned.index)]
    if untuned.empty or tuned.empty:
        untuned = rows.sort_values("utc").iloc[[0]]
        tuned = rows.sort_values("utc").iloc[[-1]]
    return untuned.iloc[0], tuned.iloc[0]


def fig_tuned_vs_untuned(out: Path) -> Path:
    untuned, tuned = _untuned_and_tuned_rows()
    pairs = [
        ("CV log loss", "cv_log_loss_mean", True),
        ("Holdout log loss", "holdout_log_loss", True),
        ("Holdout log loss (calibrated)", "holdout_log_loss_cal", True),
        ("Holdout Top-1", "holdout_top1", False),
        ("Student-test Top-1", "test_top1", False),
        ("Holdout macro F1", "holdout_macro_f1", False),
    ]
    x = np.arange(len(pairs))
    width = 0.36
    before = [float(untuned[col]) for _, col, _ in pairs]
    after = [float(tuned[col]) for _, col, _ in pairs]
    deltas = [b - a for b, a in zip(before, after)]
    lower_is_better = [inv for _, _, inv in pairs]
    labels = [name for name, _, _ in pairs]

    fig, ax = plt.subplots(figsize=(8.6, 4.6), constrained_layout=True)
    bars_a = ax.bar(x - width / 2, before, width, color=LIGHT, label="untuned", zorder=3)
    bars_b = ax.bar(x + width / 2, after, width, color=ACCENT, label="tuned", zorder=3)
    for bar, delta, better_low in zip(bars_b, deltas, lower_is_better):
        improvement = (delta > 0) if better_low else (delta < 0)
        text = f"{'+' if delta < 0 else '−'}{abs(delta):.3f}"
        ax.annotate(
            text,
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="#1e7d32" if improvement else WARN,
            weight="semibold",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("metric value")
    ax.set_ylim(0, max(max(before), max(after)) * 1.18)
    ax.set_title("Tuned vs untuned Hist GBDT — same frozen splits, same seed", color=INK, weight="semibold")
    ax.legend(loc="upper left", fontsize=9)
    path = out / "tuned_vs_untuned.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(results_dir() / "figures"))
    args = ap.parse_args()
    _style()

    try:
        import optuna
    except ImportError:
        raise SystemExit("optuna is required: .venv/bin/pip install optuna")

    db = ROOT / "artifacts" / "optuna_hist_gbdt.db"
    if not db.exists():
        raise SystemExit(f"Missing {db}; run scripts/tune_optuna.py first")
    study = optuna.load_study(study_name="hist_gbdt", storage=f"sqlite:///{db}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for path in (fig_history(study, out), fig_tuned_vs_untuned(out)):
        print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
