"""Streamlit frontend for the documented PathSnap API.

Run on Laptop 2 with PATHSNAP_API_URL=http://<laptop-1-ip>:8000.
The browser receives only public prediction geometry and aggregate metrics.
"""
from __future__ import annotations

import os
from pathlib import Path

import folium
import pandas as pd
import requests
import streamlit as st
from streamlit.components.v1 import html

from pathsnap.comparison import COMPARISON_METRICS, read_experiments
from pathsnap.ui_contract import public_prediction_payload

st.set_page_config(page_title="PathSnap", page_icon="🗺️", layout="wide")

PRED_COLOR = "#f07818"
OBS_COLOR = "#1769aa"
EXPERIMENTS_PATH = Path(__file__).resolve().parents[2] / "results" / "experiments.csv"


def _api_message(response: requests.Response) -> str:
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        detail = response.text[:500]
    return str(detail or f"HTTP {response.status_code}")


def _format_metric(value, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:.{digits}f}" if pd.notna(number) else "—"


@st.cache_data(ttl=15, show_spinner=False)
def fetch_health(base: str) -> dict:
    response = requests.get(f"{base}/health", timeout=5)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or payload.get("status") != "ok":
        raise ValueError("API returned an unexpected /health response")
    return payload


@st.cache_data(ttl=30, show_spinner=False)
def fetch_trips(base: str) -> list[str]:
    response = requests.get(f"{base}/trips", timeout=10)
    response.raise_for_status()
    payload = response.json()
    trips = payload.get("trips") if isinstance(payload, dict) else None
    if not isinstance(trips, list) or not trips:
        raise ValueError("API returned no trips; check the student_test data path on Laptop 1")
    return [str(trip) for trip in trips]


st.title("PathSnap")
st.caption("GPS route reconstruction · predictions and aggregate evaluation from the FastAPI service")

with st.sidebar:
    default_api = os.environ.get("PATHSNAP_API_URL", "http://localhost:8000")
    api_base = st.text_input("FastAPI base URL", value=default_api).strip().rstrip("/")
    st.caption("For Laptop 2, use Laptop 1’s reachable IP address and port 8000.")
    st.code("uvicorn api.main:app --host 0.0.0.0 --port 8000", language="bash")
    if api_base:
        try:
            health = fetch_health(api_base)
            st.success("API connected")
            if not health.get("model_loaded"):
                st.warning("API is reachable, but no model bundle is loaded.")
        except (requests.RequestException, ValueError) as exc:
            st.error(f"API health check failed: {exc}")

if not api_base:
    st.error("Enter a FastAPI base URL in the sidebar.")
    st.stop()

try:
    trips = fetch_trips(api_base)
except requests.HTTPError as exc:
    detail = _api_message(exc.response) if exc.response is not None else str(exc)
    st.error(f"The API rejected the trips request: {detail}")
    st.stop()
except (requests.RequestException, ValueError) as exc:
    st.error(f"Cannot load trips from {api_base}: {exc}")
    st.stop()

trip_tab, compare_tab = st.tabs(["Trip reconstruction", "Model comparison"])

with trip_tab:
    trip_id = st.selectbox("Reserved trip", trips, help="The API supplies the available student-test trips.")
    run = st.button("Reconstruct route", type="primary")

    if run:
        st.session_state.pop("result", None)
        with st.spinner("Building features and reconstructing the route…"):
            try:
                response = requests.post(
                    f"{api_base}/predict", json={"trip_id": trip_id}, timeout=(5, 300)
                )
                response.raise_for_status()
                st.session_state["result"] = public_prediction_payload(response.json())
            except requests.HTTPError as exc:
                detail = _api_message(exc.response) if exc.response is not None else str(exc)
                st.error(f"Prediction request failed: {detail}")
            except requests.Timeout:
                st.error("Prediction timed out. The API may still be building trip features; retry shortly.")
            except requests.RequestException as exc:
                st.error(f"Could not reach the API: {exc}")
            except (ValueError, TypeError, KeyError) as exc:
                st.error(f"The API response did not match the PathSnap contract: {exc}")

    result = st.session_state.get("result")
    if result and result["trip_id"] == trip_id:
        m = result["metrics"]
        timings = result["timings"]
        cols = st.columns(5)
        cols[0].metric("Top-1 accuracy", _format_metric(m.get("top1_accuracy")))
        cols[1].metric("Macro F1", _format_metric(m.get("macro_f1")))
        cols[2].metric("Route continuity", _format_metric(m.get("route_continuity")))
        cols[3].metric("Candidate miss rate", _format_metric(m.get("candidate_miss_rate")))
        cols[4].metric("Inference", f"{_format_metric(timings.get('inference_ms'), 1)} ms")

        observations = result["observations"]
        predicted_line = result["predicted_line"]
        if not observations:
            st.warning("The API returned no GPS observations for this trip.")
        else:
            try:
                obs_points = [[float(p[0]), float(p[1])] for p in observations]
                pred_points = [[float(p[0]), float(p[1])] for p in predicted_line]
                center = obs_points[len(obs_points) // 2]
            except (TypeError, ValueError, IndexError) as exc:
                st.error(f"The API returned invalid map coordinates: {exc}")
            else:
                fmap = folium.Map(location=[center[1], center[0]], zoom_start=15, tiles="OpenStreetMap")
                observations_layer = folium.FeatureGroup(name="GPS observations", show=True)
                for lon, lat in obs_points:
                    folium.CircleMarker(
                        [lat, lon], radius=2, color=OBS_COLOR, weight=0,
                        fill=True, fill_opacity=0.85,
                    ).add_to(observations_layer)
                observations_layer.add_to(fmap)
                if pred_points:
                    folium.PolyLine(
                        [[lat, lon] for lon, lat in pred_points],
                        color=PRED_COLOR, weight=4, tooltip="Predicted route",
                    ).add_to(fmap)
                folium.LayerControl(collapsed=False).add_to(fmap)
                html(fmap._repr_html_(), height=620)

        st.caption(
            f"Decoder: **{result['decoder']}** · {result['n_observations']:,} observations · "
            f"feature build {_format_metric(timings.get('features_ms'), 1)} ms · "
            f"total {_format_metric(timings.get('total_ms'), 1)} ms"
        )
        with st.expander("Additional aggregate metrics"):
            st.json(m)

with compare_tab:
    st.subheader("Experiment results")
    st.caption("Rows below are read directly from results/experiments.csv; no values are estimated by the UI.")
    experiments = read_experiments(EXPERIMENTS_PATH)
    if not experiments:
        st.info(
            "No model comparison results are available on this machine yet. Run the experiment scripts "
            "and place their generated results/experiments.csv here."
        )
    else:
        frame = pd.DataFrame(experiments)
        available_metrics = [c for c in COMPARISON_METRICS if frame[c].notna().any()]
        display_columns = ["model", "run", *available_metrics]
        st.dataframe(frame[display_columns], use_container_width=True, hide_index=True)
        if not available_metrics:
            st.info("The experiment file has model rows but no recognized numeric comparison metrics.")
        else:
            chart_metric = st.selectbox(
                "Compare metric",
                available_metrics,
                format_func=lambda value: value.replace("_", " ").title(),
            )
            chart_data = frame[["model", "run", chart_metric]].dropna()
            if not chart_data.empty:
                chart_data = chart_data.assign(experiment=chart_data["model"] + " · " + chart_data["run"])
                st.bar_chart(chart_data.set_index("experiment")[chart_metric])
