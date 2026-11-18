#!/usr/bin/env bash
# Phase-1 one-shot: inventory -> features -> baseline -> one map.
# Override with PATHSNAP_LIMIT=5 for a fast smoke run, NEG_FRAC=0.3 to speed training.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -z "${PATHSNAP_DATA_DIR:-}" ]]; then
  if [[ -d train && -d student_test ]]; then
    export PATHSNAP_DATA_DIR="$(pwd)"
  else
    export PATHSNAP_DATA_DIR="$(cd .. && pwd)/materials/student/Student folder"
  fi
fi
PY="${PYTHON:-.venv/bin/python}"
[[ -x "$PY" ]] || PY=python3
echo "python: $PY"
echo "data:   $PATHSNAP_DATA_DIR"

LIMIT="${PATHSNAP_LIMIT:-0}"
NEG_FRAC="${NEG_FRAC:-1.0}"

$PY scripts/inventory.py
$PY scripts/build_features.py --splits train student_test --limit "$LIMIT"
$PY scripts/train_baseline.py --neg-frac "$NEG_FRAC"
$PY scripts/make_map.py --split student_test --trip "${MAP_TRIP:-00000056}"

echo
echo "Next:"
echo "  PYTHONPATH=src uvicorn api.main:app --host 0.0.0.0 --port 8000"
echo "  PYTHONPATH=src streamlit run src/ui/app.py"
