#!/usr/bin/env python3
"""Build the candidate/feature table (owner: A).

Usage:
    python scripts/build_features.py --splits train student_test
    python scripts/build_features.py --splits train --limit 5      # fast smoke run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pathsnap import io
from pathsnap.config import load_config, processed_dir, results_dir
from pathsnap.features import build_trip_features


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--splits", nargs="+", default=["train", "student_test"])
    ap.add_argument("--limit", type=int, default=0, help="max trips per split (0 = all)")
    ap.add_argument("--trips", nargs="*", default=None, help="explicit trip ids")
    args = ap.parse_args()

    cfg = load_config(args.config)
    report = {}

    for split in args.splits:
        ids = io.trip_ids(split)
        if args.trips:
            ids = [t for t in ids if t in set(args.trips)]
        if args.limit:
            ids = ids[: args.limit]
        if not ids:
            continue

        frames, diags = [], []
        t0 = time.time()
        for i, tid in enumerate(ids, 1):
            trip = io.load_trip(split, tid)
            frame, diag = build_trip_features(
                trip,
                k=int(cfg["k_candidates"]),
                buffer_m=float(cfg["buffer_m"]),
                max_segments=int(cfg["network_prefilter"]["max_segments"]),
            )
            frames.append(frame)
            diags.append(diag)
            print(
                f"[{split}] {i:3d}/{len(ids)} {tid}  obs={diag['n_obs']:5d}  "
                f"rows={diag['n_rows']:6d}  miss={diag['candidate_miss_rate']:.3f}  "
                f"net={diag['n_network_segments_used']}",
                flush=True,
            )

        df = pd.concat(frames, ignore_index=True)
        out = processed_dir() / f"features_{split}.parquet"
        df.to_parquet(out, index=False)

        labeled = sum(d["n_labeled"] for d in diags)
        missed = sum(d["n_reference_arcs_missing_from_candidates"] for d in diags)
        report[split] = {
            "n_trips": len(ids),
            "rows": int(len(df)),
            "observations": int(sum(d["n_obs"] for d in diags)),
            "labeled_observations": int(labeled),
            "candidate_miss_rate": (missed / labeled) if labeled else None,
            "median_route_pos_nondecreasing": float(
                pd.Series([d["fraction_route_pos_nondecreasing"] for d in diags]).median()
            ),
            "seconds": round(time.time() - t0, 1),
            "output": str(out),
            "per_trip": diags,
        }
        print(f"[{split}] wrote {out} ({len(df):,} rows) in {report[split]['seconds']}s")

    with open(results_dir() / "feature_build_report.json", "w") as fh:
        json.dump(report, fh, indent=2)


if __name__ == "__main__":
    main()
