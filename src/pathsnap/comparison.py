"""Read actual experiment rows for the Streamlit model comparison view."""
from __future__ import annotations

import csv
from pathlib import Path


COMPARISON_METRICS = (
    "cv_top1",
    "cv_macro_f1",
    "test_top1",
    "test_macro_f1",
    "candidate_log_loss",
    "train_seconds",
    "inference_ms",
)


def read_experiments(path: str | Path) -> list[dict]:
    """Return parseable rows from an experiments CSV without inventing values."""
    path = Path(path)
    if not path.is_file():
        return []

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
            for metric in COMPARISON_METRICS:
                raw = row.get(metric)
                try:
                    value = float(raw) if raw not in (None, "") else None
                except (TypeError, ValueError):
                    value = None
                clean[metric] = value
            rows.append(clean)
        return rows
