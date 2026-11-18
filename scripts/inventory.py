#!/usr/bin/env python3
"""Dataset inventory (owner: A). Writes results/inventory.csv + a summary."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pathsnap import io
from pathsnap.config import load_config, results_dir


def n_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "rb") as fh:
        return sum(1 for _ in fh)


def main() -> None:
    cfg = load_config()
    rows = []
    for split in cfg["splits"].values():
        for tid in io.trip_ids(split):
            base = io.trip_base(split, tid)
            files = {ext: base.with_suffix(ext) for ext in (".track", ".nodes", ".arcs", ".route")}
            rows.append(
                {
                    "split": split,
                    "trip_id": tid,
                    "n_obs": n_lines(files[".track"]),
                    "n_nodes": n_lines(files[".nodes"]),
                    "n_arcs": n_lines(files[".arcs"]),
                    "n_route": n_lines(files[".route"]),
                    "mb": round(sum(p.stat().st_size for p in files.values() if p.exists()) / 1e6, 2),
                }
            )

    df = pd.DataFrame(rows)
    out = results_dir() / "inventory.csv"
    df.to_csv(out, index=False)

    summary = {}
    for split, g in df.groupby("split"):
        summary[split] = {
            "trips": int(len(g)),
            "observations": int(g.n_obs.sum()),
            "observations_per_trip_median": float(g.n_obs.median()),
            "observations_per_trip_min": int(g.n_obs.min()),
            "observations_per_trip_max": int(g.n_obs.max()),
            "nodes_median": int(g.n_nodes.median()),
            "arcs_median": int(g.n_arcs.median()),
            "trips_without_route": int((g.n_route == 0).sum()),
            "trips_with_missing_files": int(
                (g[["n_obs", "n_nodes", "n_arcs"]] == 0).any(axis=1).sum()
            ),
        }
        print(
            f"{split:14s} {len(g):3d} trips | {g.n_obs.sum():>8,} obs "
            f"(median {int(g.n_obs.median()):,}) | network {int(g.n_nodes.median()):,} nodes / "
            f"{int(g.n_arcs.median()):,} arcs | no route: {(g.n_route == 0).sum()}"
        )

    with open(results_dir() / "inventory_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
