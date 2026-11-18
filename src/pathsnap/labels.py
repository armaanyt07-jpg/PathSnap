"""Map each GPS observation to the reference arc it most likely belongs to.

The supplied .route is a chronological list of arc ids, but GPS observations are
~1 Hz and there are many more observations than route arcs. We therefore assign
each observation the nearest arc *among the route's arcs*, and keep the position
of that arc in the route so monotonicity can be checked.

This is a heuristic label; validate it (labels.diagnostics) before trusting metrics.
"""
from __future__ import annotations

import numpy as np

from .geo import SegmentIndex


def map_observations_to_route(
    obs_xy: np.ndarray,
    route: np.ndarray,
    arc_a_xy: np.ndarray,
    arc_b_xy: np.ndarray,
) -> dict:
    """Return label_arc, distance to it, and first position in the route."""
    n = len(obs_xy)
    label = np.full(n, -1, dtype=np.int64)
    dist = np.full(n, np.inf)
    route_pos = np.full(n, -1, dtype=np.int64)

    if route is None or len(route) == 0:
        return {"label_arc": label, "dist_to_ref_m": dist, "route_pos": route_pos}

    uniq, first_index = np.unique(route, return_index=True)
    idx = SegmentIndex(arc_a_xy[uniq], arc_b_xy[uniq])

    for i, p in enumerate(obs_xy):
        d, _ = idx.point_dists(p)
        j = int(np.argmin(d))
        label[i] = uniq[j]
        dist[i] = d[j]
        route_pos[i] = first_index[j]
    return {"label_arc": label, "dist_to_ref_m": dist, "route_pos": route_pos}


def diagnostics(route_pos: np.ndarray, dist_to_ref_m: np.ndarray) -> dict:
    valid = route_pos >= 0
    mono = float("nan")
    if valid.sum() >= 2:
        r = route_pos[valid]
        mono = float(np.mean(np.diff(r) >= 0))
    return {
        "n_labeled": int(valid.sum()),
        "fraction_route_pos_nondecreasing": mono,
        "median_dist_to_ref_m": float(np.median(dist_to_ref_m[valid])) if valid.any() else float("nan"),
    }
