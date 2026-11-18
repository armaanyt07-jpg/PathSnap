"""Folium maps (owner: A). Colours/markers per docs/contracts.md and report §6."""
from __future__ import annotations

import numpy as np
import folium

OBS_COLOR = "#1f77b4"
NET_COLOR = "#9aa0a6"
REF_COLOR = "#2ca02c"
PRED_COLOR = "#ff7f0e"

_LEGEND = """
<div style="position: fixed; bottom: 28px; left: 28px; z-index: 9999;
     background: rgba(255,255,255,.92); padding: 10px 12px; border-radius: 6px;
     font: 12px/1.5 -apple-system, Segoe UI, sans-serif; box-shadow: 0 1px 4px rgba(0,0,0,.25);">
  <b>PathSnap</b><br>
  <span style="color:{net};">&#9644;</span> road network &nbsp;
  <span style="color:{obs};">&#9679;</span> GPS observations<br>
  <span style="color:{ref};">&#9644;</span> reference route &nbsp;
  <span style="color:{pred};">&#9644;</span> predicted route
</div>
""".format(net=NET_COLOR, obs=OBS_COLOR, ref=REF_COLOR, pred=PRED_COLOR)


def _to_latlon(points) -> list[list[float]]:
    arr = np.asarray(points, dtype=float)
    if arr.size == 0:
        return []
    return [[float(lat), float(lon)] for lon, lat in arr]


def make_map(
    obs_lonlat: np.ndarray,
    network_segments_lonlat=None,
    pred_line=None,
    ref_line=None,
    center: tuple[float, float] | None = None,
    max_network_segments: int = 1500,
    zoom_start: int = 15,
    *,
    pred_lines=None,
):
    obs = np.asarray(obs_lonlat, dtype=float)
    if center is None:
        lon0, lat0 = obs.mean(axis=0) if len(obs) else (0.0, 0.0)
    else:
        lon0, lat0 = center

    m = folium.Map(location=[lat0, lon0], zoom_start=zoom_start, tiles="OpenStreetMap")

    if network_segments_lonlat is not None:
        segs = np.asarray(network_segments_lonlat, dtype=float)
        if segs.ndim == 3 and len(segs):
            if len(segs) > max_network_segments:
                pick = np.linspace(0, len(segs) - 1, max_network_segments).astype(int)
                segs = segs[pick]
            net = folium.FeatureGroup(name="Road network", show=True)
            for a, b in segs:
                folium.PolyLine(
                    _to_latlon([a, b]),
                    color=NET_COLOR,
                    weight=1,
                    opacity=0.5,
                ).add_to(net)
            net.add_to(m)

    obs_layer = folium.FeatureGroup(name="GPS observations", show=True)
    for lon, lat in obs:
        folium.CircleMarker(
            location=[lat, lon], radius=2, color=OBS_COLOR, weight=0,
            fill=True, fill_opacity=0.85,
        ).add_to(obs_layer)
    obs_layer.add_to(m)

    if ref_line is not None and len(ref_line):
        ref = folium.FeatureGroup(name="Reference route", show=True)
        folium.PolyLine(_to_latlon(ref_line), color=REF_COLOR, weight=4, opacity=0.85).add_to(ref)
        ref.add_to(m)

    lines = pred_lines if pred_lines is not None else [pred_line]
    lines = [line for line in lines if line is not None and len(line)]
    if lines:
        pred = folium.FeatureGroup(name="Predicted route", show=True)
        for line in lines:
            folium.PolyLine(_to_latlon(line), color=PRED_COLOR, weight=4, opacity=0.9).add_to(pred)
        pred.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(_LEGEND))
    return m
