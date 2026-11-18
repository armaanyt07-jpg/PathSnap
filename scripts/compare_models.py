#!/usr/bin/env python3
"""Build the model comparison table for the slide and the report (owner: B).

    python scripts/compare_models.py

Reads results/model_runs.csv (written by scripts/train_models.py), keeps the best
run per family, and writes results/model_comparison.csv plus a markdown table.
Smoke runs (notes == "smoke") are excluded.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pathsnap.config import results_dir

COLUMNS = [
    "family",
    "calibrated",
    "cv_top1_mean",
    "cv_top1_std",
    "holdout_top1",
    "holdout_top1_covered",
    "holdout_macro_f1",
    "holdout_log_loss",
    "test_top1",
    "test_log_loss",
    "fit_seconds",
    "inference_ms_per_1k",
]


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--include-smoke", action="store_true", help="dev only: include smoke runs")
    args = ap.parse_args()

    path = results_dir() / "model_runs.csv"
    if not path.exists():
        raise SystemExit("No results/model_runs.csv. Run scripts/train_models.py first.")
    runs = pd.read_csv(path)
    if "notes" in runs.columns and not args.include_smoke:
        runs = runs[runs["notes"].fillna("") != "smoke"]

    if runs.empty:
        raise SystemExit("Only smoke runs found; run a full training pass first.")

    # One row per family: the most recent run. Re-running a family supersedes the
    # earlier row, so a stale run under a different protocol cannot win on score.
    latest = runs.sort_values("utc").drop_duplicates("family", keep="last")
    best = latest.sort_values("holdout_top1", ascending=False).reset_index(drop=True)

    table = best[[c for c in COLUMNS if c in best.columns]].copy()
    out_csv = results_dir() / "model_comparison.csv"
    table.to_csv(out_csv, index=False)

    def fmt(v):
        if pd.isna(v):
            return "-"
        if isinstance(v, bool):
            return "yes" if v else "no"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    header = "| " + " | ".join(table.columns) + " |"
    rule = "|" + "|".join(["---"] * len(table.columns)) + "|"
    lines = [header, rule]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in table.columns) + " |")
    markdown = "\n".join(lines)

    out_md = results_dir() / "model_comparison.md"
    out_md.write_text(
        "# Model comparison (workstream B)\n\n"
        "Ranked by held-out Top-1 over labelled observations. `holdout_top1_covered`\n"
        "removes observations whose reference arc is missing from the candidate set,\n"
        "so the gap between the two columns is the candidate miss rate, not model error.\n\n"
        f"{markdown}\n\n"
        f"Runs compared: {len(best)} families. Source: results/model_runs.csv.\n"
    )

    print(markdown)
    print(f"\nwrote {out_csv}\nwrote {out_md}")


if __name__ == "__main__":
    main()
