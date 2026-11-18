"""Geometry helpers: local projection, segment index, bearings.

At city scale an equirectangular projection around the trip centroid is accurate
enough (<0.1% distance error over a few km) and keeps the skeleton dependency-free.
If pyproj is available later, swap LocalProjection for a UTM CRS - features are in metres.
"""
from __future__ import annotations

import numpy as np

R_EARTH = 6_371_008.8  # metres


class LocalProjection:
    """Tangent-plane projection to metres around (lon0, lat0)."""

    def __init__(self, lon0: float, lat0: float):
        self.lon0 = float(lon0)
        self.lat0 = float(lat0)
        self.kx = R_EARTH * np.cos(np.deg2rad(self.lat0))
        self.ky = R_EARTH

    def xy(self, lon, lat) -> np.ndarray:
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        return np.column_stack(
            [np.deg2rad(lon - self.lon0) * self.kx, np.deg2rad(lat - self.lat0) * self.ky]
        )

    def lonlat(self, xy) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        lon = self.lon0 + np.rad2deg(xy[:, 0] / self.kx)
        lat = self.lat0 + np.rad2deg(xy[:, 1] / self.ky)
        return np.column_stack([lon, lat])


class SegmentIndex:
    """Vectorised point-to-segment queries over a fixed set of segments (metres)."""

    def __init__(self, a: np.ndarray, b: np.ndarray):
        self.a = np.ascontiguousarray(a, dtype=float)
        self.b = np.ascontiguousarray(b, dtype=float)
        self.ab = self.b - self.a
        self.denom = np.einsum("ij,ij->i", self.ab, self.ab)
        self.length = np.sqrt(self.denom)

    def __len__(self) -> int:
        return len(self.a)

    def point_dists(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Distances (m) and clamped projection parameter t in [0,1] for one point."""
        ap = p - self.a
        t = np.einsum("ij,ij->i", ap, self.ab) / np.maximum(self.denom, 1e-12)
        np.clip(t, 0.0, 1.0, out=t)
        proj = self.a + t[:, None] * self.ab
        return np.linalg.norm(p - proj, axis=1), t

    def nearest(self, p: np.ndarray, k: int = 10, max_dist: float | None = None):
        d, t = self.point_dists(p)
        if max_dist is not None:
            d = np.where(d <= max_dist, d, np.inf)
        finite = np.isfinite(d)
        if not finite.any():
            return np.empty(0, dtype=int), np.empty(0), np.empty(0)
        vidx = np.flatnonzero(finite)
        kk = min(k, vidx.size)
        local = np.argpartition(d[vidx], kk - 1)[:kk]
        sel = vidx[local]
        order = sel[np.argsort(d[sel])]
        return order, d[order], t[order]

    def bbox_mask(self, xmin: float, xmax: float, ymin: float, ymax: float) -> np.ndarray:
        ax, ay = self.a[:, 0], self.a[:, 1]
        bx, by = self.b[:, 0], self.b[:, 1]
        return (
            (np.maximum(ax, bx) >= xmin)
            & (np.minimum(ax, bx) <= xmax)
            & (np.maximum(ay, by) >= ymin)
            & (np.minimum(ay, by) <= ymax)
        )


def bearing_deg(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """Compass bearing (0=N, 90=E) from p0 to p1, element-wise over (n,2) arrays."""
    dx = p1[..., 0] - p0[..., 0]
    dy = p1[..., 1] - p0[..., 1]
    return np.mod(np.degrees(np.arctan2(dx, dy)), 360.0)


def angle_diff_abs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)
