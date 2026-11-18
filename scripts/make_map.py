#!/usr/bin/env python3
"""Render a Folium map for one trip (owner: A).

    python scripts/make_map.py --split student_test --trip 00000056
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pathsnap import baseline, metrics, pipeline, viz
from pathsnap.config import artifacts_dir, load_config, results_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="student_test")
    ap.add_argument("--trip", default="00000056")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    model_path = artifacts_dir() / "model_bundle.joblib"
    model = baseline.load(model_path) if model_path.exists() else None
    df, diag = pipeline.get_features(args.split, args.trip, cfg, include_labels=False, append_missing_labels=False)
    trip = pipeline.get_trip(args.split, args.trip)
    chosen = None
    if model is not None:
        scores = baseline.predict_scores(model, df)
        chosen = baseline.choose_one_per_observation(df, scores)

    nodes, arcs = trip["nodes"], trip["arcs"]
    pred_line = pipeline.line_from_arcs(chosen["arc_id"].to_numpy(), nodes, arcs) if chosen is not None else None
    ref_line = (
        pipeline.line_from_arcs(trip["route"], nodes, arcs)
        if trip["route"] is not None
        else None
    )
    network = pipeline.network_segments(trip, cfg)
    obs = trip["track"][["lon", "lat"]].to_numpy(dtype=float)

    fmap = viz.make_map(
        obs_lonlat=obs,
        network_segments_lonlat=network,
        pred_line=pred_line,
        ref_line=ref_line,
    )
    out = Path(args.out) if args.out else results_dir() / "maps" / f"{args.trip}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out))

    print(f"{args.trip}: {'baseline prediction included' if chosen is not None else 'map only; train baseline for prediction layer'}; wrote {out}")


if __name__ == "__main__":
    main()
