#!/usr/bin/env python3
"""Report figures from recorded model runs (owner: B).

Every plotted number comes from ``results/model_runs.csv`` or from scoring a
saved artifact on the frozen holdout split — nothing is illustrative.

    python scripts/make_figures.py                 # all figures -> results/figures/
    python scripts/make_figures.py --only families,calibration
    python scripts/make_figures.py --dpi 300

Figures that need candidate scores load ``artifacts/candidates/*.joblib``; if
those are absent (they are gitignored) the script skips them with a warning and
still draws the comparison charts from the run log.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pathsnap import baseline, evaluation, metrics
from pathsnap.config import ROOT, artifacts_dir, load_config, processed_dir, results_dir
from pathsnap.features import FEATURE_COLUMNS

MODEL_LABELS = {
    "champion": "champion",
    "hist_gbdt": "Hist GBDT",
    "extra_trees": "Extra Trees",
    "mlp": "MLP",
    "nystroem_sgd": "Nystroem SGD",
    "logistic": "Logistic",
    "sgd_huber": "SGD Huber",
}

INK = "#1f2d3d"
MUTED = "#8a97a5"
ACCENT = "#2f6f9f"
LIGHT = "#a8c8df"
WARN = "#c0392b"


def _label(family: str) -> str:
    return MODEL_LABELS.get(family, family.replace("_", " ").title())


class _FitAdapter:
    """Give a fitted scorer the ``fit`` hook permutation_importance checks for."""

    def __init__(self, model):
        self.model = model

    def fit(self, X=None, y=None):
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)


def _style(dpi: int) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": dpi,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlecolor": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": "#c9d3dc",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#e8edf1",
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "xtick.color": INK,
            "ytick.color": INK,
            "legend.frameon": False,
            "figure.facecolor": "white",
        }
    )


def load_runs(path: Path) -> pd.DataFrame:
    """Latest non-smoke run per family, best holdout first."""
    runs = pd.read_csv(path)
    runs["notes"] = runs.get("notes", pd.Series(index=runs.index, dtype=object)).fillna("")
    runs = runs[runs["notes"] != "smoke"]
    runs = runs[runs["holdout_top1"].notna()]
    if runs.empty:
        raise SystemExit(f"No completed runs in {path}")
    runs = runs.sort_values("utc").groupby("family", as_index=False).tail(1)
    return runs.sort_values("holdout_top1", ascending=False).reset_index(drop=True)


def _artifact_path(row: pd.Series) -> Path:
    return ROOT / str(row["artifact"])


def load_holdout(k: int) -> pd.DataFrame | None:
    path = processed_dir() / "features_train.parquet"
    split_path = artifacts_dir() / "split.json"
    if not path.exists() or not split_path.exists():
        print(f"  skipped: {path} or {split_path} missing")
        return None
    with open(split_path) as fh:
        holdout_ids = set(json.load(fh)["holdout"])
    frame = _read_feature_columns(path)
    frame = frame[frame["trip_id"].isin(holdout_ids)].reset_index(drop=True)
    print(f"  holdout frame: {len(frame):,} rows, {frame['trip_id'].nunique()} trips")
    return frame


def load_test() -> pd.DataFrame | None:
    path = processed_dir() / "features_student_test.parquet"
    if not path.exists():
        print(f"  skipped: {path} missing")
        return None
    frame = _read_feature_columns(path)
    print(f"  student-test frame: {len(frame):,} rows, {frame['trip_id'].nunique()} trips")
    return frame


def _read_feature_columns(path: Path) -> pd.DataFrame:
    cols = list(dict.fromkeys(["trip_id", "obs_idx", "arc_id", "label", "label_arc", *FEATURE_COLUMNS]))
    return pd.read_parquet(path, columns=cols)


def score_families(runs: pd.DataFrame, frame: pd.DataFrame, partition: str = "holdout") -> dict[str, np.ndarray]:
    scores = {}
    for row in runs.itertuples(index=False):
        path = ROOT / str(row.artifact)
        if not path.exists():
            print(f"  skipped {row.family}: {path} missing")
            continue
        import joblib

        model = joblib.load(path)
        scores[row.family] = baseline.predict_scores(model, frame)
        print(f"  scored {row.family} on {partition}")
    return scores


def score_test_partition(runs: pd.DataFrame, test: pd.DataFrame, scores: dict[str, np.ndarray], k: int) -> pd.DataFrame:
    rows = []
    for family in runs["family"]:
        if family not in scores:
            continue
        met, _ = evaluation.score_frame(test, scores[family], k)
        rows.append(
            {
                "family": family,
                "top1_accuracy": met["top1_accuracy"],
                "top1_covered": met["top1_covered"],
                "macro_f1": met["macro_f1"],
                "log_loss": met["candidate_log_loss"],
                "n_observations": met["n_observations"],
            }
        )
    table = pd.DataFrame(rows)
    if len(table):
        out = results_dir() / "student_test_metrics.csv"
        table.to_csv(out, index=False)
        print(f"  wrote {out}")
    return table


# ---------------------------------------------------------------- figures


def fig_families(runs: pd.DataFrame, out: Path) -> Path:
    df = runs.sort_values("holdout_top1")
    y = np.arange(len(df))
    top1 = df["holdout_top1_cal"].fillna(df["holdout_top1"]).to_numpy()
    covered = df["holdout_top1_covered"].to_numpy()

    fig, ax = plt.subplots(figsize=(9.2, 0.52 * len(df) + 2.0))
    ax.barh(y + 0.19, top1, height=0.36, color=ACCENT, label="Top-1, all observations")
    ax.barh(y - 0.19, covered, height=0.36, color=LIGHT, label="Top-1, reference arc was a candidate")
    ax.plot(df["cv_top1_mean"], y, "D", color=INK, ms=5, label="5-fold CV mean (dev trips)")
    ax.hlines(y, df["cv_top1_mean"] - df["cv_top1_std"], df["cv_top1_mean"] + df["cv_top1_std"],
              color=INK, lw=1, alpha=0.6)

    for yi, value in zip(y + 0.19, top1):
        ax.text(value + 0.004, yi, f"{value:.3f}", va="center", fontsize=8.5, color=INK)
    for yi, value in zip(y - 0.19, covered):
        ax.text(value + 0.004, yi, f"{value:.3f}", va="center", fontsize=8.5, color="#4a6b80")

    ax.set_yticks(y, [_label(f) for f in df["family"]])
    floor = max(0.0, np.floor(min(top1.min(), df["cv_top1_mean"].min()) * 10) / 10 - 0.1)
    ax.set_xlim(floor, 1.0)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("accuracy  (axis starts at {:.1f})".format(floor))
    ax.set_title("Candidate classification on held-out trips (16 trips)")
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    path = out / "family_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_cost(runs: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for row in runs.itertuples(index=False):
        x = row.inference_ms_per_1k
        y = row.holdout_top1_cal if np.isfinite(row.holdout_top1_cal) else row.holdout_top1
        champion = row.family == runs.iloc[0]["family"]
        ax.scatter(x, y, s=90 if champion else 60, color=ACCENT if champion else MUTED,
                   edgecolor="white", linewidth=1.2, zorder=3)
        ax.annotate(_label(row.family), (x, y), textcoords="offset points", xytext=(7, 5),
                    fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("inference time, ms per 1,000 candidate rows (log scale)")
    ax.set_ylabel("holdout Top-1 accuracy")
    ax.set_title("Accuracy versus scoring cost")
    fig.tight_layout()
    path = out / "accuracy_vs_cost.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_cv_macro_f1(runs: pd.DataFrame, out: Path) -> Path:
    df = runs.sort_values("cv_macro_f1_mean")
    y = np.arange(len(df))
    values = df["cv_macro_f1_mean"].to_numpy()

    fig, ax = plt.subplots(figsize=(8.4, 0.5 * len(df) + 1.9))
    ax.barh(y, values, height=0.62, color=ACCENT)
    for yi, value in zip(y, values):
        ax.text(value + 0.003, yi, f"{value:.3f}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y, [_label(f) for f in df["family"]])
    floor = max(0.0, np.floor(values.min() * 10) / 10 - 0.1)
    ax.set_xlim(floor, 1.0)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("macro F1  (axis starts at {:.1f})".format(floor))
    ax.set_title("Cross-validation macro F1 by model (5 grouped folds)")
    fig.tight_layout()
    path = out / "cv_macro_f1.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_student_test(table: pd.DataFrame, out: Path) -> Path:
    df = table.sort_values("top1_accuracy")
    y = np.arange(len(df))
    top1 = df["top1_accuracy"].to_numpy()
    f1 = df["macro_f1"].to_numpy()

    fig, ax = plt.subplots(figsize=(8.8, 0.55 * len(df) + 2.0))
    ax.barh(y + 0.18, top1, height=0.34, color=ACCENT, label="Top-1 accuracy")
    ax.barh(y - 0.18, f1, height=0.34, color=LIGHT, label="macro F1")
    for yi, value in zip(y + 0.18, top1):
        ax.text(value + 0.004, yi, f"{value:.3f}", va="center", fontsize=8.5, color=INK)
    for yi, value in zip(y - 0.18, f1):
        ax.text(value + 0.004, yi, f"{value:.3f}", va="center", fontsize=8.5, color="#4a6b80")
    ax.set_yticks(y, [_label(f) for f in df["family"]])
    floor = max(0.0, np.floor(min(top1.min(), f1.min()) * 10) / 10 - 0.1)
    ax.set_xlim(floor, 1.0)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("score on the 10 reserved student-test trips  (axis starts at {:.1f})".format(floor))
    ax.set_title("Student-test Top-1 and macro F1 by model")
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    path = out / "student_test_metrics.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_log_loss(runs: pd.DataFrame, table: pd.DataFrame, out: Path) -> Path:
    df = runs[["family", "holdout_log_loss_cal", "holdout_log_loss"]].copy()
    df["holdout"] = df["holdout_log_loss_cal"].fillna(df["holdout_log_loss"])
    df = df.merge(
        table[["family", "log_loss"]].rename(columns={"log_loss": "student_test"}),
        on="family",
        how="left",
    )
    df = df.sort_values("holdout")
    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(8.6, 0.52 * len(df) + 2.0))
    ax.barh(y + 0.19, df["holdout"].to_numpy(), height=0.36, color=ACCENT,
            label="holdout, calibrated")
    ax.barh(y - 0.19, df["student_test"].to_numpy(), height=0.36, color=LIGHT,
            label="student-test, calibrated")
    for yi, value in zip(y + 0.19, df["holdout"]):
        ax.text(value + 0.0015, yi, f"{value:.3f}", va="center", fontsize=8.5, color=INK)
    for yi, value in zip(y - 0.19, df["student_test"]):
        ax.text(value + 0.0015, yi, f"{value:.3f}", va="center", fontsize=8.5, color="#4a6b80")
    ax.set_yticks(y, [_label(f) for f in df["family"]])
    ax.set_xlim(0, float(df[["holdout", "student_test"]].max().max()) * 1.25)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("candidate log loss (lower is better)")
    ax.set_title("Log loss by model, holdout versus student-test")
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    path = out / "log_loss_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_mlp_loss(out: Path) -> Path | None:
    path = ROOT / "artifacts" / "candidates" / "mlp.joblib"
    if not path.exists():
        print(f"  skipped MLP loss curve: {path} missing")
        return None
    import joblib

    model = joblib.load(path)
    base = getattr(model, "base_", model)
    parts = [base] + list(getattr(base, "named_steps", {}).values())
    curve = next((getattr(p, "loss_curve_", None) for p in parts if getattr(p, "loss_curve_", None)), None)
    if not curve:
        print("  skipped MLP loss curve: no epoch history in the saved artifact")
        return None
    epochs = np.arange(1, len(curve) + 1)

    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    ax.plot(epochs, curve, "o-", color=ACCENT, lw=2, ms=4)
    ax.set_xlabel("training epoch")
    ax.set_ylabel("training log loss")
    ax.set_title("MLP training loss per epoch (saved final model)")
    ax.annotate(
        f"final {curve[-1]:.4f} after {len(curve)} epochs\n"
        "no validation history: early stopping was disabled",
        xy=(epochs[-1], curve[-1]),
        xytext=(-12, 18),
        textcoords="offset points",
        ha="right",
        fontsize=8.5,
        color=INK,
    )
    ax.set_xlim(0, len(curve) + 1)
    fig.tight_layout()
    path = out / "mlp_training_loss.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_calibration(runs: pd.DataFrame, holdout: pd.DataFrame, out: Path, k: int) -> Path | None:
    row = runs.iloc[0]
    path = _artifact_path(row)
    if not path.exists():
        print(f"  skipped calibration: {path} missing")
        return None
    import joblib

    model = joblib.load(path)
    lat = evaluation.lattice_view(holdout, k, require_positive=False)
    p = baseline.predict_scores(model, lat)
    y = lat["label"].to_numpy(dtype=int)

    order = np.argsort(p, kind="mergesort")
    bins = np.array_split(order, 10)
    mean_p = np.array([p[b].mean() for b in bins])
    frac_pos = np.array([y[b].mean() for b in bins])
    share = np.array([len(b) / len(p) for b in bins])
    ece = float(np.sum(share * np.abs(mean_p - frac_pos)))

    fig, (ax, side) = plt.subplots(1, 2, figsize=(9.6, 4.2), gridspec_kw={"width_ratios": [2.1, 1]})
    ax.plot([0, 1], [0, 1], "--", color=MUTED, lw=1, label="perfect calibration")
    ax.plot(mean_p, frac_pos, "o-", color=ACCENT, lw=2, ms=5, label="champion (10 quantile bins)")
    ax.fill_between(mean_p, mean_p, frac_pos, color=ACCENT, alpha=0.12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed fraction of true candidates")
    ax.set_title(f"Calibration, {_label(row.family)} on holdout")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.text(0.03, 0.72, f"ECE = {ece:.3f}", transform=ax.transAxes, fontsize=9, color=INK)

    raw = row.holdout_log_loss
    cal = row.holdout_log_loss_cal if np.isfinite(row.holdout_log_loss_cal) else raw
    bars = side.bar(["raw", "isotonic"], [raw, cal], color=[MUTED, ACCENT], width=0.6)
    for bar, value in zip(bars, [raw, cal]):
        side.text(bar.get_x() + bar.get_width() / 2, value + 0.004, f"{value:.3f}",
                  ha="center", fontsize=9, color=INK)
    side.set_ylabel("candidate log loss (lower is better)")
    side.set_title("Calibration effect")
    side.set_ylim(0, max(raw, cal) * 1.25)
    side.grid(axis="x", visible=False)

    fig.tight_layout()
    path = out / "calibration_champion.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_per_trip(runs: pd.DataFrame, holdout: pd.DataFrame, scores: dict[str, np.ndarray], out: Path, k: int) -> Path | None:
    rows = []
    for family, family_scores in scores.items():
        met, chosen = evaluation.score_frame(holdout, family_scores, k)
        table = evaluation.per_trip_table(chosen)
        table["family"] = family
        rows.append(table)
    if not rows:
        return None
    table = pd.concat(rows, ignore_index=True)
    pivot = table.pivot(index="family", columns="trip_id", values="top1_accuracy")
    order = runs.set_index("family").index
    pivot = pivot.loc[[f for f in order if f in pivot.index]]
    pivot = pivot[pivot.mean().sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(12.4, 0.55 * len(pivot) + 2.4))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(pivot.shape[1]), pivot.columns, rotation=60, ha="right", fontsize=8.5)
    ax.set_yticks(range(pivot.shape[0]), [_label(f) for f in pivot.index], fontsize=9)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7.2,
                    color="white" if value < 0.55 else "black")
    ax.set_title("Per-trip Top-1 accuracy on held-out trips")
    ax.grid(False)
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("Top-1 accuracy", fontsize=9)
    fig.tight_layout()
    path = out / "per_trip_heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_miss(holdout: pd.DataFrame, out: Path, k: int) -> Path | None:
    labelled = holdout[holdout["label_arc"] >= 0]
    if labelled.empty:
        return None
    all_obs = labelled[["trip_id", "obs_idx"]].drop_duplicates().set_index(["trip_id", "obs_idx"])
    in_lattice = labelled[labelled["cand_rank"] < k]
    hit = in_lattice.groupby(["trip_id", "obs_idx"])["label"].any()
    hit = hit.reindex(all_obs.index, fill_value=False)
    miss = (1.0 - hit.groupby(level=0).mean()).sort_values(ascending=False)
    counts = all_obs.groupby(level=0).size().reindex(miss.index)

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    colors = [WARN if v > 0.10 else MUTED for v in miss]
    y = np.arange(len(miss))
    ax.barh(y, miss.to_numpy(), color=colors, height=0.7)
    for yi, (value, n) in enumerate(zip(miss, counts)):
        ax.text(value + 0.005, yi, f"{value:.1%}  (n={n:,})", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y, miss.index, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, max(0.12, float(miss.max()) * 1.35))
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("share of observations whose reference arc was not in the k=10 candidates")
    ax.set_title("Candidate-generation miss rate per held-out trip")
    fig.tight_layout()
    path = out / "candidate_miss_by_trip.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_separation(runs: pd.DataFrame, holdout: pd.DataFrame, scores: dict[str, np.ndarray], out: Path, k: int) -> Path | None:
    family = runs.iloc[0]["family"]
    if family not in scores:
        return None
    lat = evaluation.lattice_view(holdout, k, require_positive=False)
    p = scores[family][holdout["cand_rank"].to_numpy() < k]
    y = lat["label"].to_numpy(dtype=int)

    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    bins = np.linspace(0, 1, 41)
    ax.hist(p[y == 0], bins=bins, density=True, histtype="stepfilled", alpha=0.35, color=MUTED,
            label=f"wrong candidate (n={int((y == 0).sum()):,})")
    ax.hist(p[y == 1], bins=bins, density=True, histtype="step", lw=2.2, color=ACCENT,
            label=f"reference candidate (n={int((y == 1).sum()):,})")
    ax.set_yscale("log")
    ax.set_xlabel("predicted probability that the candidate is the reference arc")
    ax.set_ylabel("density (log scale)")
    ax.set_title(f"Score separation on holdout, {_label(family)}")
    ax.legend(loc="upper center", fontsize=8.5)
    fig.tight_layout()
    path = out / "score_separation.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_importance(runs: pd.DataFrame, holdout: pd.DataFrame, out: Path, k: int, sample: int = 20000) -> Path | None:
    from sklearn.inspection import permutation_importance

    row = runs.iloc[0]
    path = _artifact_path(row)
    if not path.exists():
        return None
    import joblib

    model = joblib.load(path)
    lat = evaluation.lattice_view(holdout, k, require_positive=False)
    rng = np.random.default_rng(0)
    take = rng.choice(len(lat), size=min(sample, len(lat)), replace=False)
    X = lat.iloc[take][FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = lat.iloc[take]["label"].to_numpy(dtype=int)

    def negative_log_loss(estimator, X_eval, y_eval):
        prob = np.asarray(estimator.predict_proba(X_eval))[:, 1]
        return -metrics.candidate_log_loss(np.asarray(y_eval, dtype=int), prob)

    result = permutation_importance(_FitAdapter(model), X, y, scoring=negative_log_loss,
                                    n_repeats=5, random_state=0)
    order = np.argsort(result.importances_mean)
    names = [FEATURE_COLUMNS[i].replace("_", " ") for i in order]
    mean = result.importances_mean[order]
    std = result.importances_std[order]

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.barh(names, mean, xerr=std, color=[ACCENT if v > 0 else MUTED for v in mean],
            height=0.68, error_kw={"ecolor": "#555", "capsize": 3, "lw": 1})
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlabel("increase in candidate log loss when the feature is shuffled")
    ax.set_title(f"Permutation importance, {_label(row.family)} (holdout sample, 5 repeats)")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    path = out / "feature_importance.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(results_dir() / "model_runs.csv"))
    ap.add_argument("--out", default=str(results_dir() / "figures"))
    ap.add_argument("--dpi", type=int, default=180)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument(
        "--only",
        default="",
        help="comma-separated subset: families,cost,cv_macro_f1,student_test,log_loss,mlp_loss,"
             "calibration,per_trip,miss,separation,importance",
    )
    args = ap.parse_args()

    _style(args.dpi)
    cfg = load_config()
    k = args.k or int(cfg["k_candidates"])
    runs = load_runs(Path(args.runs))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wanted = {name.strip() for name in args.only.split(",") if name.strip()}
    print(f"{len(runs)} families: {', '.join(runs['family'])}")

    written: list[Path] = []
    if not wanted or "families" in wanted:
        written.append(fig_families(runs, out))
    if not wanted or "cost" in wanted:
        written.append(fig_cost(runs, out))
    if not wanted or "cv_macro_f1" in wanted:
        written.append(fig_cv_macro_f1(runs, out))
    if not wanted or "mlp_loss" in wanted:
        path = fig_mlp_loss(out)
        if path:
            written.append(path)

    holdout_needed = not wanted or bool(wanted & {"calibration", "per_trip", "separation", "importance", "miss"})
    test_needed = not wanted or bool(wanted & {"student_test", "log_loss"})

    holdout, scores = None, {}
    if holdout_needed:
        holdout = load_holdout(k)
        if holdout is not None:
            scores = score_families(runs, holdout)
    if holdout is not None:
        if not wanted or "calibration" in wanted:
            path = fig_calibration(runs, holdout, out, k)
            if path:
                written.append(path)
        if not wanted or "per_trip" in wanted:
            path = fig_per_trip(runs, holdout, scores, out, k)
            if path:
                written.append(path)
        if not wanted or "miss" in wanted:
            path = fig_miss(holdout, out, k)
            if path:
                written.append(path)
        if not wanted or "separation" in wanted:
            path = fig_separation(runs, holdout, scores, out, k)
            if path:
                written.append(path)
        if not wanted or "importance" in wanted:
            path = fig_importance(runs, holdout, out, k)
            if path:
                written.append(path)

    if test_needed:
        test = load_test()
        if test is not None:
            test_scores = score_families(runs, test, "student-test")
            test_table = score_test_partition(runs, test, test_scores, k)
            if len(test_table):
                if not wanted or "student_test" in wanted:
                    written.append(fig_student_test(test_table, out))
                if not wanted or "log_loss" in wanted:
                    written.append(fig_log_loss(runs, test_table, out))

    for path in written:
        print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")
    if not written:
        raise SystemExit("No figures written; check --only names and artifact paths.")


if __name__ == "__main__":
    main()
