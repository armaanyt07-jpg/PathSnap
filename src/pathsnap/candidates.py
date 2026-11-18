"""k-nearest candidate arcs per observation, within a spatial prefilter."""
from __future__ import annotations

import numpy as np

from .geo import SegmentIndex


def prefilter_segments(
    a_all: np.ndarray,
    b_all: np.ndarray,
    obs_xy: np.ndarray,
    buffer_m: float,
    max_segments: int | None = None,
) -> np.ndarray:
    """Boolean mask of segments intersecting the trajectory bbox + buffer.

    If the mask is too large, the buffer shrinks until it fits. Shrinking (rather
    than random sampling) keeps every segment near the trajectory.
    """
    xmin, ymin = obs_xy.min(axis=0) - buffer_m
    xmax, ymax = obs_xy.max(axis=0) + buffer_m
    index = SegmentIndex(a_all, b_all)
    buf = float(buffer_m)
    mask = index.bbox_mask(xmin, xmax, ymin, ymax)
    while max_segments and mask.sum() > max_segments and buf > 20:
        buf *= 0.75
        mask = index.bbox_mask(xmin + (buffer_m - buf), xmax - (buffer_m - buf),
                               ymin + (buffer_m - buf), ymax - (buffer_m - buf))
    return mask


def candidate_arrays(
    obs_xy: np.ndarray,
    segment_index: SegmentIndex,
    segment_arc_ids: np.ndarray,
    k: int,
    max_dist_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (n,k) arrays: arc ids (-1 padded), distances (m), along-parameter t."""
    n = len(obs_xy)
    ids = np.full((n, k), -1, dtype=np.int64)
    dist = np.full((n, k), np.inf)
    tpos = np.zeros((n, k))
    for i, p in enumerate(obs_xy):
        d, t = segment_index.point_dists(p)
        if max_dist_m is not None:
            d = np.where(d <= max_dist_m, d, np.inf)
        valid = np.isfinite(d)
        if not valid.any():
            continue
        vidx = np.flatnonzero(valid)
        kk = min(k, vidx.size)
        local = np.argpartition(d[vidx], kk - 1)[:kk]
        sel = vidx[local]
        order = sel[np.argsort(d[sel])]
        ids[i, : order.size] = segment_arc_ids[order]
        dist[i, : order.size] = d[order]
        tpos[i, : order.size] = t[order]
    return ids, dist, tpos
