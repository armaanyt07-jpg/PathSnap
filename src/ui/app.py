"""Streamlit UI (owner: D). Talks only to the API; never sees reference labels.

Run (Laptop 2):
    PYTHONPATH=src streamlit run src/ui/app.py
"""
from __future__ import annotations

import folium
import requests
import streamlit as st
from streamlit.components.v1 import html

st.set_page_config(page_title="PathSnap", layout="wide")

PRED_COLOR = "#ff7f0e"
OBS_COLOR = "#1f77b4"

st.title("PathSnap — route reconstruction")
st.caption("Select one of the ten reserved trips; the API reconstructs the route.")

with st.sidebar:
    api_base = st.text_input("API base URL", value="http://localhost:8000").rstrip("/")
    st.markdown("**Laptop 1:** `uvicorn api.main:app --host 0.0.0.0 --port 8000`")


@st.cache_data(ttl=60)
def fetch_trips(base: str):
    r = requests.get(f"{base}/trips", timeout=10)
    r.raise_for_status()
    return r.json()["trips"]


try:
    trips = fetch_trips(api_base)
except Exception as exc:  # noqa: BLE001
    st.error(f"Cannot reach the API at {api_base}: {exc}")
    st.stop()

trip_id = st.selectbox("Evaluation trip", trips)
run = st.button("Reconstruct route", type="primary")

if run:
    with st.spinner("Running the prediction pipeline…"):
        try:
            resp = requests.post(
                f"{api_base}/predict", json={"trip_id": trip_id}, timeout=300
            )
            resp.raise_for_status()
            st.session_state["result"] = resp.json()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Prediction failed: {exc}")

result = st.session_state.get("result")
if result and result.get("trip_id") == trip_id:
    m = result["metrics"]
    cols = st.columns(5)
    cols[0].metric("Top-1 accuracy", f"{m.get('top1_accuracy', float('nan')):.3f}")
    cols[1].metric("Macro F1", f"{m.get('macro_f1', float('nan')):.3f}")
    cols[2].metric("Route continuity", f"{m.get('route_continuity', float('nan')):.3f}")
    cols[3].metric("Candidate miss rate", f"{m.get('candidate_miss_rate', float('nan')):.3f}")
    cols[4].metric("Inference", f"{result['timings']['inference_ms']:.0f} ms")

    obs = result["observations"]
    fmap = folium.Map(
        location=[obs[len(obs) // 2][1], obs[len(obs) // 2][0]],
        zoom_start=15,
        tiles="OpenStreetMap",
    )
    for lon, lat in obs:
        folium.CircleMarker(
            [lat, lon], radius=2, color=OBS_COLOR, weight=0, fill=True, fill_opacity=0.85
        ).add_to(fmap)
    if result["predicted_line"]:
        folium.PolyLine(
            [[lat, lon] for lon, lat in result["predicted_line"]],
            color=PRED_COLOR,
            weight=4,
        ).add_to(fmap)

    html(fmap._repr_html_(), height=620)
    st.caption(
        f"decoder: **{result['decoder']}** · {result['n_observations']} observations · "
        f"features {result['timings']['features_ms']:.0f} ms · "
        f"total {result['timings']['total_ms']:.0f} ms"
    )
    with st.expander("Raw response"):
        st.json({k: v for k, v in result.items() if k not in ("observations", "predicted_line", "predicted_route")})
