"""Read actual experiment rows for the Streamlit model comparison view."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path


COMPARISON_METRICS = (
    "cv_top1",
    "cv_macro_f1",
    "test_top1",
    "test_macro_f1",
    "candidate_log_loss",
    "train_seconds",
    "inference_ms",
    "holdout_top1",
    "holdout_top1_covered",
    "holdout_macro_f1",
    "holdout_log_loss",
    "test_log_loss",
    "inference_ms_per_1k",
    "fit_seconds",
)
DECODER_METRICS = (
    "top1_accuracy",
    "macro_f1",
    "weighted_f1",
    "route_continuity",
    "candidate_miss_rate",
    "candidate_f1",
    "candidate_log_loss",
)


def _number(raw):
    try:
        value = float(raw) if raw not in (None, "") else None
    except (TypeError, ValueError):
        return None
    return value if value is None or math.isfinite(value) else None


def read_experiments(path: str | Path) -> list[dict]:
    """Read model/utc experiment CSV rows without inventing values."""
    path = Path(path)
    if not path.is_file():
        return []
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "model" not in reader.fieldnames:
                return []
            rows = []
            for row_number, row in enumerate(reader, start=1):
                model = (row.get("model") or "").strip()
                if not model:
                    continue
                clean = {"model": model, "run": row.get("utc") or str(row_number)}
                clean.update({metric: _number(row.get(metric)) for metric in COMPARISON_METRICS})
                rows.append(clean)
            return rows
    except (OSError, UnicodeError, csv.Error):
        return []


def read_model_comparison(path: str | Path) -> list[dict]:
    """Read B's saved family comparison table using only present numeric results."""
    path = Path(path)
    if not path.is_file():
        return []
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "family" not in reader.fieldnames:
                return []
            rows = []
            for row in reader:
                family = (row.get("family") or "").strip()
                if not family:
                    continue
                clean = {"model": family, "run": "model comparison"}
                for metric in COMPARISON_METRICS:
                    raw = row.get(metric)
                    if metric == "cv_top1" and raw in (None, ""):
                        raw = row.get("cv_top1_mean")
                    if metric == "candidate_log_loss" and raw in (None, ""):
                        raw = row.get("test_log_loss") or row.get("holdout_log_loss")
                    if metric == "train_seconds" and raw in (None, ""):
                        raw = row.get("fit_seconds")
                    clean[metric] = _number(raw)
                rows.append(clean)
            return rows
    except (OSError, UnicodeError, csv.Error):
        return []


def read_decoder_comparison(path: str | Path) -> list[dict]:
    """Read C's actual per-trip HMM and classifier-only comparison JSON."""
    path = Path(path)
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []

    rows = []
    for row_number, item in enumerate(payload, start=1):
        if not isinstance(item, dict) or not item.get("trip_id"):
            continue
        for method, key in (("HMM/Viterbi", "metrics"), ("Classifier argmax", "baseline_metrics")):
            metrics = item.get(key)
            if not isinstance(metrics, dict):
                continue
            row = {"trip_id": str(item["trip_id"]), "run": str(row_number), "method": method}
            for metric in DECODER_METRICS:
                row[metric] = _number(metrics.get(metric))
            if method == "HMM/Viterbi":
                timings = item.get("timings")
                raw_decode_ms = timings.get("decode_ms") if isinstance(timings, dict) else None
                row["decode_ms"] = _number(raw_decode_ms)
            else:
                row["decode_ms"] = None
            rows.append(row)
    return rows
