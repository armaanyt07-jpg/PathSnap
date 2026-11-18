"""Glue + per-trip cache shared by the API, the map script and experiments."""
from __future__ import annotations

import numpy as np

from . import io as psio
from .config import load_config
from .features import build_trip_features

_TRIPS: dict = {}
_FEATURES: dict = {}


def get_trip(split: str, trip_id: str) -> dict:
    key = (split, trip_id)
    if key not in _TRIPS:
        _TRIPS[key] = psio.load_trip(split, trip_id)
    return _TRIPS[key]


def get_features(split: str, trip_id: str, cfg: dict | None = None, *, include_labels: bool = True, append_missing_labels: bool = True):
    cfg = cfg or load_config()
    key = (split, trip_id, cfg["k_candidates"], cfg["buffer_m"], include_labels, append_missing_labels)
    if key not in _FEATURES:
        trip = get_trip(split, trip_id)
        _FEATURES[key] = build_trip_features(
            trip,
            k=int(cfg["k_candidates"]),
            buffer_m=float(cfg["buffer_m"]),
            max_segments=int(cfg["network_prefilter"]["max_segments"]),
            include_labels=include_labels,
            append_missing_labels=append_missing_labels,
        )
    return _FEATURES[key]


def line_from_arcs(arc_ids, nodes: np.ndarray, arcs: np.ndarray) -> np.ndarray:
    """Connected display geometry: from-node of the first arc, then every to-node."""
    ids = np.asarray(arc_ids, dtype=np.int64)
    if ids.size == 0:
        return np.empty((0, 2))
    pts = [nodes[arcs[ids[0], 0]]]
    pts.extend(nodes[arcs[a, 1]] for a in ids)
    return np.asarray(pts, dtype=float)


def network_segments(trip: dict, cfg: dict | None = None):
    """Prefiltered network for drawing (lon/lat, shape (m,2,2))."""
    from .candidates import prefilter_segments
    from .geo import LocalProjection, SegmentIndex

    cfg = cfg or load_config()
    nodes, arcs = trip["nodes"], trip["arcs"]
    lon, lat = trip["track"]["lon"].to_numpy(float), trip["track"]["lat"].to_numpy(float)
    proj = LocalProjection(np.median(lon), np.median(lat))
    obs_xy = proj.xy(lon, lat)
    nodes_xy = proj.xy(nodes[:, 0], nodes[:, 1])
    a, b = nodes_xy[arcs[:, 0]], nodes_xy[arcs[:, 1]]
    mask = prefilter_segments(
        a, b, obs_xy, float(cfg["buffer_m"]), int(cfg["network_prefilter"]["max_segments"])
    )
    segs_xy = np.stack([a[mask], b[mask]], axis=1)
    return proj.lonlat(segs_xy.reshape(-1, 2)).reshape(-1, 2, 2)
