#!/usr/bin/env bash
# Run API and UI on one laptop for a local demo.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON:-$ROOT/.venv/bin/python}"
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN=python3
API_PORT="${PATHSNAP_API_PORT:-8000}"
export PATHSNAP_API_URL="http://127.0.0.1:${API_PORT}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" scripts/validate_dataset.py
MODEL_PATH="${PATHSNAP_MODEL:-$ROOT/artifacts/model_bundle.joblib}"
if [[ ! -f "$MODEL_PATH" ]]; then
  echo "warning: no model bundle at $MODEL_PATH; API will report not_ready and /predict will return HTTP 503" >&2
fi

"$PYTHON_BIN" -m uvicorn api.main:app --host 127.0.0.1 --port "$API_PORT" &
API_PID=$!
cleanup() {
  kill "$API_PID" 2>/dev/null || true
  wait "$API_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

ready=0
for _ in $(seq 1 30); do
  if curl --silent --fail "${PATHSNAP_API_URL}/health" >/dev/null; then
    ready=1
    break
  fi
  sleep 1
done
if [[ "$ready" != 1 ]]; then
  echo "FastAPI did not become ready at ${PATHSNAP_API_URL}/health" >&2
  exit 1
fi

"$PYTHON_BIN" -m streamlit run src/ui/app.py \
  --server.address "${PATHSNAP_UI_HOST:-127.0.0.1}" \
  --server.port "${PATHSNAP_UI_PORT:-8501}"
