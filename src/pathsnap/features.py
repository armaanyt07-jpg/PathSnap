"""Build the feature table defined in docs/contracts.md §1."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import candidates as cand
from .geo import LocalProjection, SegmentIndex, angle_diff_abs, bearing_deg
from .labels import diagnostics, map_observations_to_route

FEATURE_COLUMNS = [
    "cand_rank",
    "perp_dist_m",
    "along_m",
    "rel_pos",
    "arc_len_m",
    "dist_ratio",
    "heading_diff_deg",
    "gap_m",
    "gap_s",
    "speed_mps",
]


def _point_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    ab = b - a
    denom = float(ab @ ab)
    t = float(np.clip((p - a) @ ab / denom, 0.0, 1.0)) if denom > 0 else 0.0
    return float(np.linalg.norm(p - (a + t * ab))), t


def build_trip_features(
    trip: dict,
    k: int = 10,
    buffer_m: float = 300.0,
    max_segments: int = 30_000,
) -> tuple[pd.DataFrame, dict]:
    track = trip["track"]
    nodes = trip["nodes"]
    arcs = trip["arcs"]
    route = trip["route"]
    trip_id = trip["trip_id"]

    lon = track["lon"].to_numpy(dtype=float)
    lat = track["lat"].to_numpy(dtype=float)
    ts = track["t"].to_numpy(dtype=float)
    n = len(track)

    proj = LocalProjection(np.median(lon), np.median(lat))
    obs_xy = proj.xy(lon, lat)
    nodes_xy = proj.xy(nodes[:, 0], nodes[:, 1])
    a_all = nodes_xy[arcs[:, 0]]
    b_all = nodes_xy[arcs[:, 1]]

    if n == 0:
        return pd.DataFrame(columns=["trip_id", "obs_idx", *FEATURE_COLUMNS, "label"]), {
            "trip_id": trip_id,
            "n_obs": 0,
            "n_labeled": 0,
            "n_rows": 0,
            "candidate_miss_rate": float("nan"),
        }

    ab = b_all - a_all
    seg_len = np.linalg.norm(ab, axis=1)
    seg_bearing = bearing_deg(a_all, b_all)

    # 1. Reference labels
    if route is not None and len(route) > 0:
        lab = map_observations_to_route(obs_xy, route, a_all, b_all)
    else:
        lab = {
            "label_arc": np.full(n, -1, dtype=np.int64),
            "dist_to_ref_m": np.full(n, np.inf),
            "route_pos": np.full(n, -1, dtype=np.int64),
        }
    label_arc = lab["label_arc"]
    label_diag = diagnostics(lab["route_pos"], lab["dist_to_ref_m"])

    # 2. Candidate set from the spatially prefiltered network
    mask = cand.prefilter_segments(a_all, b_all, obs_xy, buffer_m, max_segments)
    sub_arc_ids = np.flatnonzero(mask)
    sub_index = SegmentIndex(a_all[mask], b_all[mask])
    ids, dist, tpos = cand.candidate_arrays(obs_xy, sub_index, sub_arc_ids, k)

    # 3. Observation-level kinematics
    if n >= 2:
        prev = np.vstack([obs_xy[0], obs_xy[:-1]])
        nxt = np.vstack([obs_xy[1:], obs_xy[-1]])
        steps = np.linalg.norm(np.diff(obs_xy, axis=0), axis=1)
        dts = np.diff(ts)
        gap_m = np.concatenate([[0.0], steps])
        gap_s = np.concatenate([[0.0], dts])
    else:
        prev = nxt = obs_xy
        gap_m = np.zeros(n)
        gap_s = np.zeros(n)
    obs_bearing = bearing_deg(prev, nxt)
    speed = np.where(gap_s > 0, gap_m / np.maximum(gap_s, 1e-9), 0.0)

    # 4. Flatten (obs, candidate) pairs
    valid = (ids >= 0).ravel()
    obs_idx = np.repeat(np.arange(n), k)[valid]
    cand_rank = np.tile(np.arange(k), n)[valid]
    arc_id = ids.ravel()[valid]
    perp = dist.ravel()[valid]
    rel = tpos.ravel()[valid]
    arc_len = seg_len[arc_id]
    label_obs = label_arc[obs_idx]

    frame = pd.DataFrame(
        {
            "trip_id": trip_id,
            "obs_idx": obs_idx,
            "lon": lon[obs_idx],
            "lat": lat[obs_idx],
            "t": ts[obs_idx],
            "arc_id": arc_id,
            "cand_rank": cand_rank,
            "perp_dist_m": perp,
            "along_m": rel * arc_len,
            "rel_pos": rel,
            "arc_len_m": arc_len,
            "dist_ratio": perp / np.maximum(arc_len, 1e-6),
            "heading_diff_deg": angle_diff_abs(obs_bearing[obs_idx], seg_bearing[arc_id]),
            "gap_m": gap_m[obs_idx],
            "gap_s": gap_s[obs_idx],
            "speed_mps": speed[obs_idx],
            "label": (label_obs >= 0) & (arc_id == label_obs),
            "label_arc": label_obs,
            "route_pos": lab["route_pos"][obs_idx],
        }
    )

    # 5. Ensure every labelled observation has its reference arc among the rows.
    #    Miss rate is measured on the candidate set *before* this append (contract §1).
    has_label = (ids == label_arc[:, None]).any(axis=1)
    missing = np.flatnonzero((label_arc >= 0) & ~has_label)
    n_labeled = int((label_arc >= 0).sum())

    if missing.size:
        extras = []
        for i in missing:
            arc = int(label_arc[i])
            p = obs_xy[i]
            d, t = _point_segment(p, a_all[arc], b_all[arc])
            length = float(seg_len[arc])
            extras.append(
                {
                    "trip_id": trip_id,
                    "obs_idx": int(i),
                    "lon": float(lon[i]),
                    "lat": float(lat[i]),
                    "t": float(ts[i]),
                    "arc_id": arc,
                    "cand_rank": k,
                    "perp_dist_m": d,
                    "along_m": t * length,
                    "rel_pos": t,
                    "arc_len_m": length,
                    "dist_ratio": d / max(length, 1e-6),
                    "heading_diff_deg": float(
                        angle_diff_abs(obs_bearing[i], seg_bearing[arc])
                    ),
                    "gap_m": float(gap_m[i]),
                    "gap_s": float(gap_s[i]),
                    "speed_mps": float(speed[i]),
                    "label": True,
                    "label_arc": arc,
                    "route_pos": int(lab["route_pos"][i]),
                }
            )
        frame = pd.concat([frame, pd.DataFrame(extras)], ignore_index=True)

    frame = frame.sort_values(["obs_idx", "cand_rank"], kind="mergesort").reset_index(drop=True)

    diag = {
        "trip_id": trip_id,
        "n_obs": n,
        "n_labeled": n_labeled,
        "n_rows": int(len(frame)),
        "candidate_miss_rate": float(len(missing) / n_labeled) if n_labeled else float("nan"),
        "n_reference_arcs_missing_from_candidates": int(len(missing)),
        "n_network_segments_used": int(mask.sum()),
        "median_candidates_per_obs": float(np.median((ids >= 0).sum(axis=1))) if n else 0.0,
        **label_diag,
    }
    return frame, diag
