#!/usr/bin/env bash
# Start the Streamlit frontend. Set PATHSNAP_API_URL for a remote Laptop 1.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON:-$ROOT/.venv/bin/python}"
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN=python3
UI_HOST="${PATHSNAP_UI_HOST:-127.0.0.1}"
UI_PORT="${PATHSNAP_UI_PORT:-8501}"
export PATHSNAP_API_URL="${PATHSNAP_API_URL:-http://localhost:8000}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

exec "$PYTHON_BIN" -m streamlit run src/ui/app.py \
  --server.address "$UI_HOST" --server.port "$UI_PORT"
