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
exec "$PYTHON_BIN" -m uvicorn api.main:app --host "$API_HOST" --port "$API_PORT"
