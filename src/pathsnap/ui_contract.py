"""Allow-listed API response fields that the Streamlit frontend may retain."""
from __future__ import annotations


PUBLIC_METRICS = {
    "top1_accuracy",
    "macro_f1",
    "weighted_f1",
    "candidate_miss_rate",
    "route_continuity",
    "precision",
    "recall",
    "candidate_f1",
    "candidate_log_loss",
}
TIMING_FIELDS = {"features_ms", "inference_ms", "total_ms"}


def public_prediction_payload(payload: dict) -> dict:
    """Keep only documented prediction fields; reference labels never enter UI state."""
    if not isinstance(payload, dict):
        raise ValueError("Prediction response must be a JSON object")
    required = {"trip_id", "decoder", "n_observations", "observations", "predicted_route", "predicted_line", "metrics", "timings"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"Prediction response is missing fields: {', '.join(sorted(missing))}")
    if not isinstance(payload["metrics"], dict) or not isinstance(payload["timings"], dict):
        raise ValueError("Prediction metrics and timings must be JSON objects")
    return {
        "trip_id": str(payload["trip_id"]),
        "decoder": str(payload["decoder"]),
        "n_observations": int(payload["n_observations"]),
        "observations": payload["observations"],
        "predicted_route": payload["predicted_route"],
        "predicted_line": payload["predicted_line"],
        "metrics": {k: v for k, v in payload["metrics"].items() if k in PUBLIC_METRICS},
        "timings": {k: v for k, v in payload["timings"].items() if k in TIMING_FIELDS},
    }
