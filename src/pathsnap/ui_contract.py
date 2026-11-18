"""Allow-listed API response fields that the Streamlit frontend may retain."""
from __future__ import annotations

import math


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
TIMING_FIELDS = {"features_ms", "model_ms", "decode_ms", "inference_ms", "total_ms"}
DECODER_CONFIG_FIELDS = {"beta_m", "max_gap_s", "max_speed_mps", "distance_slack_m"}


def _coordinates(value, field: str, *, minimum: int = 0) -> list[list[float]]:
    if not isinstance(value, list) or len(value) < minimum:
        raise ValueError(f"Prediction field '{field}' must be a list with at least {minimum} coordinate(s)")
    result = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f"Prediction field '{field}' must contain [longitude, latitude] pairs")
        try:
            lon, lat = float(point[0]), float(point[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Prediction field '{field}' contains a non-numeric coordinate") from exc
        if not math.isfinite(lon) or not math.isfinite(lat) or not -180 <= lon <= 180 or not -90 <= lat <= 90:
            raise ValueError(f"Prediction field '{field}' contains a coordinate outside WGS84 bounds")
        result.append([lon, lat])
    return result


def public_prediction_payload(payload: dict) -> dict:
    """Keep only documented prediction fields; reference labels never enter UI state."""
    if not isinstance(payload, dict):
        raise ValueError("Prediction response must be a JSON object")
    required = {
        "trip_id", "decoder", "n_observations", "observations", "predicted_route",
        "predicted_line", "metrics", "timings",
    }
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"Prediction response is missing fields: {', '.join(sorted(missing))}")
    if not isinstance(payload["metrics"], dict) or not isinstance(payload["timings"], dict):
        raise ValueError("Prediction metrics and timings must be JSON objects")
    observations = _coordinates(payload["observations"], "observations")
    predicted_line = _coordinates(payload["predicted_line"], "predicted_line")
    if not isinstance(payload["predicted_route"], list):
        raise ValueError("Prediction field 'predicted_route' must be a list")
    try:
        n_observations = int(payload["n_observations"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Prediction field 'n_observations' must be a non-negative integer") from exc
    if n_observations < 0 or n_observations != payload["n_observations"]:
        raise ValueError("Prediction field 'n_observations' must be a non-negative integer")
    public = {
        "trip_id": str(payload["trip_id"]),
        "decoder": str(payload["decoder"]),
        "n_observations": n_observations,
        "observations": observations,
        "predicted_route": payload["predicted_route"],
        "predicted_line": predicted_line,
        "metrics": {k: v for k, v in payload["metrics"].items() if k in PUBLIC_METRICS},
        "timings": {k: v for k, v in payload["timings"].items() if k in TIMING_FIELDS},
    }
    baseline_metrics = payload.get("baseline_metrics")
    if isinstance(baseline_metrics, dict):
        public["baseline_metrics"] = {
            k: v for k, v in baseline_metrics.items() if k in PUBLIC_METRICS
        }
    predicted_lines = payload.get("predicted_lines")
    if isinstance(predicted_lines, list):
        public["predicted_lines"] = [
            _coordinates(line, "predicted_lines segment", minimum=2)
            for line in predicted_lines
        ]
    breaks = payload.get("breaks")
    if isinstance(breaks, list):
        public["breaks"] = [
            {key: item[key] for key in ("obs_idx", "reason") if key in item}
            for item in breaks
            if isinstance(item, dict)
        ]
    decoder_config = payload.get("decoder_config")
    if isinstance(decoder_config, dict):
        public["decoder_config"] = {
            k: v for k, v in decoder_config.items() if k in DECODER_CONFIG_FIELDS
        }
    return public


def validate_health_payload(payload: dict) -> dict:
    """Accept API readiness states while distinguishing connectivity from model availability."""
    if (
        not isinstance(payload, dict)
        or payload.get("status") not in {"ok", "not_ready"}
        or not isinstance(payload.get("model_loaded"), bool)
    ):
        raise ValueError("API returned an unexpected /health response")
    public = {"status": payload["status"], "model_loaded": payload["model_loaded"]}
    models = payload.get("models")
    if isinstance(models, list) and all(isinstance(name, str) for name in models):
        public["models"] = models
    default_model = payload.get("default_model")
    if isinstance(default_model, str):
        public["default_model"] = default_model
    return public
