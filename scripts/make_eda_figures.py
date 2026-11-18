#!/usr/bin/env python3
"""Dataset figures required by the report's EDA section (owner: A/B boundary).

Draws only what the brief asks for as plots: GPS trajectories and road-network
geometry, both from the raw trip files on the stated split. Missing-value counts
are already reported in ``results/eda_summary.json`` and are left as a table.

    python scripts/make_eda_figures.py
    python scripts/make_eda_figures.py --split train --only gps
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
from matplotlib.collections import LineCollection
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pathsnap import io
from pathsnap.config import results_dir
from pathsnap.geo import LocalProjection

INK = "#1f2d3d"
MUTED = "#9aa0a6"
ACCENT = "#2f6f9f"
ROUTE = "#2ca02c"
OBS = "#1f77b4"
SPEED_MAX = 20.0  # m/s; clipped so a few outliers do not flatten the colour scale


def _style(dpi: int) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": dpi,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.titlecolor": INK,
            "axes.labelcolor": INK,
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


def _speed(track: pd.DataFrame) -> tuple[np.ndarray, float]:
    lon = track["lon"].to_numpy(float)
    lat = track["lat"].to_numpy(float)
    ts = track["t"].to_numpy(float)
    proj = LocalProjection(float(np.median(lon)), float(np.median(lat)))
    xy = proj.xy(lon, lat)
    dt = np.diff(ts)
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    speed = np.zeros(len(track))
    good = dt > 0
    speed[1:][good] = step[good] / dt[good]
    return np.clip(speed, 0, SPEED_MAX), float(np.median(lat))


def _relax_overlapping_ticks(fig, axes, pad: float = 3.0) -> None:
    """Rotate or thin tick labels on panels whose axes are too narrow for them.

    Trips that run mostly north-south keep an equal aspect ratio, which leaves the
    axes only a few dozen pixels wide; the default longitude labels then print on
    top of each other.
    """
    fig.canvas.draw()
    for ax in axes.ravel():
        if not ax.get_visible():
            continue
        labels = [t for t in ax.get_xticklabels() if t.get_text()]
        boxes = sorted((t.get_window_extent() for t in labels), key=lambda b: b.x0)
        if any(a.x1 + pad > b.x0 for a, b in zip(boxes, boxes[1:])):
            ax.tick_params(axis="x", labelrotation=90, labelsize=7.5, pad=2)
        labels = [t for t in ax.get_yticklabels() if t.get_text()]
        boxes = sorted((t.get_window_extent() for t in labels), key=lambda b: b.y0)
        if any(a.y1 + pad > b.y0 for a, b in zip(boxes, boxes[1:])):
            ax.yaxis.set_major_locator(MaxNLocator(3))


def _sample(rows: pd.DataFrame, column: str, count: int) -> pd.DataFrame:
    ordered = rows.sort_values(column).reset_index(drop=True)
    if len(ordered) <= count:
        return ordered
    positions = np.linspace(0.1, 0.95, count) * (len(ordered) - 1)
    return ordered.iloc[sorted({int(round(p)) for p in positions})].reset_index(drop=True)


def fig_gps(rows: pd.DataFrame, split: str, out: Path, count: int) -> Path:
    picks = _sample(rows, "n_obs", count)
    cols = 3
    nrows = int(np.ceil(len(picks) / cols))
    fig = plt.figure(figsize=(4.9 * cols + 0.8, 3.9 * nrows + 1.0), constrained_layout=True)
    gs = fig.add_gridspec(nrows, cols + 1, width_ratios=[1] * cols + [0.05])
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(cols)] for r in range(nrows)])
    scatter = None
    for ax, row in zip(axes.ravel(), picks.itertuples(index=False)):
        trip = io.load_trip(split, row.trip_id)
        track = trip["track"]
        speed, lat0 = _speed(track)
        scatter = ax.scatter(
            track["lon"], track["lat"], c=speed, cmap="viridis",
            vmin=0, vmax=SPEED_MAX, s=1.6, linewidths=0,
        )
        ax.set_aspect(1.0 / np.cos(np.radians(lat0)))
        ax.set_title(f"{split}/{row.trip_id} · {row.n_obs:,} obs", fontsize=9)
        ax.set_xlabel("longitude", fontsize=8.5)
        ax.set_ylabel("latitude", fontsize=8.5)
        ax.tick_params(labelsize=8)
        ax.ticklabel_format(useOffset=False, style="plain")
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
    for ax in axes.ravel()[len(picks):]:
        ax.set_visible(False)
    if scatter is not None:
        cbar = fig.colorbar(scatter, cax=fig.add_subplot(gs[:, cols]))
        cbar.set_label("speed, m/s (clipped at {:.0f})".format(SPEED_MAX), fontsize=9)
        cbar.ax.tick_params(labelsize=8)
    fig.suptitle(f"GPS trajectories sampled across the {split} split", color=INK, fontsize=13, weight="semibold")
    _relax_overlapping_ticks(fig, axes)
    path = out / "eda_gps_trajectories.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_network(rows: pd.DataFrame, split: str, out: Path) -> Path:
    picks = _sample(rows, "n_arcs", 2)
    fig, axes = plt.subplots(1, len(picks), figsize=(6.6 * len(picks), 5.4), squeeze=False)
    for ax, row in zip(axes.ravel(), picks.itertuples(index=False)):
        trip = io.load_trip(split, row.trip_id)
        nodes, arcs = trip["nodes"], trip["arcs"]
        lon = trip["track"]["lon"].to_numpy(float)
        lat = trip["track"]["lat"].to_numpy(float)
        lat0 = float(np.median(lat))

        segments = np.stack([nodes[arcs[:, 0]], nodes[arcs[:, 1]]], axis=1)
        if len(segments) > 80_000:
            keep = np.linspace(0, len(segments) - 1, 80_000).astype(int)
            segments = segments[keep]
        ax.add_collection(LineCollection(segments, colors=MUTED, linewidths=0.35, alpha=0.6))
        if trip["route"] is not None and len(trip["route"]):
            route = np.stack([nodes[arcs[trip["route"], 0]], nodes[arcs[trip["route"], 1]]], axis=1)
            ax.add_collection(LineCollection(route, colors=ROUTE, linewidths=1.5, alpha=0.9))
        ax.scatter(lon, lat, s=2, color=OBS, alpha=0.8, linewidths=0)

        ax.set_aspect(1.0 / np.cos(np.radians(lat0)))
        ax.autoscale_view()
        ax.set_title(f"{split}/{row.trip_id}  ·  {row.n_nodes:,} nodes, {row.n_arcs:,} arcs")
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        ax.ticklabel_format(useOffset=False, style="plain")
        ax.grid(False)
    handles = [
        plt.Line2D([], [], color=MUTED, lw=1.4, label="road network"),
        plt.Line2D([], [], color=ROUTE, lw=2.0, label="reference route"),
        plt.Line2D([], [], color=OBS, marker="o", lw=0, ms=4, label="GPS observations"),
    ]
    axes.ravel()[0].legend(handles=handles, loc="best", fontsize=8.5)
    fig.suptitle(f"Road-network geometry on the {split} split", color=INK, fontsize=13, weight="semibold")
    fig.tight_layout()
    path = out / "eda_road_network.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default=str(results_dir() / "figures"))
    ap.add_argument("--dpi", type=int, default=180)
    ap.add_argument("--trips", type=int, default=6, help="GPS panels to draw")
    ap.add_argument("--only", default="", help="comma-separated: gps,network")
    args = ap.parse_args()

    _style(args.dpi)
    eda_path = results_dir() / "eda_trips.csv"
    if not eda_path.exists():
        raise SystemExit(f"Missing {eda_path}; run scripts/explore_data.py first")
    rows = pd.read_csv(eda_path, dtype={"trip_id": str})
    rows = rows[rows["split"] == args.split]
    if rows.empty:
        raise SystemExit(f"No rows for split {args.split} in {eda_path}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wanted = {name.strip() for name in args.only.split(",") if name.strip()}
    written = []
    if not wanted or "gps" in wanted:
        written.append(fig_gps(rows, args.split, out, args.trips))
    if not wanted or "network" in wanted:
        written.append(fig_network(rows, args.split, out))
    for path in written:
        print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
