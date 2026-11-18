#!/usr/bin/env bash
# Start the FastAPI service. For Laptop 1 set PATHSNAP_API_HOST=0.0.0.0.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON:-$ROOT/.venv/bin/python}"
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN=python3
API_HOST="${PATHSNAP_API_HOST:-0.0.0.0}"
API_PORT="${PATHSNAP_API_PORT:-8000}"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" scripts/validate_dataset.py
MODEL_PATH="${PATHSNAP_MODEL:-$ROOT/artifacts/model_bundle.joblib}"
if [[ ! -f "$MODEL_PATH" ]]; then
  echo "warning: no model bundle at $MODEL_PATH; API will report not_ready and /predict will return HTTP 503" >&2
fi
exec "$PYTHON_BIN" -m uvicorn api.main:app --host "$API_HOST" --port "$API_PORT"
