"""Raw dataset parsing. The ONLY module that reads .track/.nodes/.arcs/.route.

Formats (see Dataset Exploration.pdf):
  .track  lon lat timestamp        (whitespace, no header, ~1 Hz)
  .nodes  lon lat                  (node id = row index)
  .arcs   from_node to_node        (arc id = row index)
  .route  arc_id                   (chronological reference route, where supplied)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import data_dir


def split_dir(split: str) -> Path:
    return data_dir() / split


def trip_ids(split: str) -> list[str]:
    d = split_dir(split)
    if not d.exists():
        raise FileNotFoundError(
            f"Split '{split}' not found under {d}. "
            "Set PATHSNAP_DATA_DIR to the folder containing train/ and student_test/."
        )
    return sorted(p.name for p in d.iterdir() if p.is_dir() and not p.name.startswith("."))


def trip_base(split: str, trip_id: str) -> Path:
    return split_dir(split) / trip_id / trip_id


def read_track(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["lon", "lat", "t"])
    df = df.dropna(subset=["lon", "lat"]).reset_index(drop=True)
    return df


def read_nodes(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=float, ndmin=2)


def read_arcs(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=float, ndmin=2).astype(np.int64)


def read_route(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    route = np.loadtxt(path, dtype=float, ndmin=1)
    if route.size == 0:
        return None
    return route.astype(np.int64)


def load_trip(split: str, trip_id: str, with_route: bool = True) -> dict:
    base = trip_base(split, trip_id)
    track = read_track(base.with_suffix(".track"))
    nodes = read_nodes(base.with_suffix(".nodes"))
    arcs = read_arcs(base.with_suffix(".arcs"))
    route = read_route(base.with_suffix(".route")) if with_route else None
    return {
        "trip_id": trip_id,
        "split": split,
        "track": track,
        "nodes": nodes,
        "arcs": arcs,
        "route": route,
    }
