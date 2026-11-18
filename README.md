# PathSnap — Phase 1 skeleton

Distributed map-matching & route reconstruction (PES UE24CS352A).

This repo is a **running skeleton**, not a finished solution: parse → candidates → features →
baseline classifier → Folium map, plus a FastAPI backend and a Streamlit UI. The HMM/Viterbi
decoder, the remaining models, Bayesian tuning and the report are the next blocks.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export PATHSNAP_DATA_DIR="/absolute/path/to/Student folder"   # contains train/ and student_test/
bash scripts/run_phase1.sh
```

`run_phase1.sh` runs inventory → features → baseline → one map. Artefacts land in
`results/` and `data/processed/`. If `PATHSNAP_DATA_DIR` is unset the code falls back to
`./data` and then to `../materials/student/Student folder`.

### API (Laptop 1)

```bash
PYTHONPATH=src uvicorn api.main:app --host 0.0.0.0 --port 8000
curl -s localhost:8000/health
curl -s -X POST localhost:8000/predict -H 'content-type: application/json' -d '{"trip_id":"00000056"}'
```

### UI (Laptop 2)

```bash
PYTHONPATH=src streamlit run src/ui/app.py
# sidebar: set API base to http://<laptop-1-ip>:8000
```

## Layout & ownership

```
src/pathsnap/io.py          A   parsing (the only place raw files are read)
src/pathsnap/geo.py         A   projection, segment index, bearings
src/pathsnap/labels.py      A   observation -> reference arc
src/pathsnap/candidates.py  A   k nearest arcs
src/pathsnap/features.py    A   feature table (contract §1)
src/pathsnap/viz.py         A   Folium maps
src/pathsnap/metrics.py     D   top-1, macro F1, miss rate, continuity (contract §3/§5)
src/pathsnap/baseline.py    B   reference model bundle (contract §2)
src/pathsnap/model_zoo.py   B   six model families behind predict_scores
src/pathsnap/evaluation.py  B   trip-level split, subsampling, observation-level metrics
src/pathsnap/pipeline.py    C/D glue + per-trip cache
src/api/main.py             C   FastAPI (contract §4)
src/ui/app.py               D   Streamlit
scripts/                    owners as named in each file header
```

## Modelling (workstream B)

```bash
# one family, identical folds for every family
python scripts/train_models.py --family hist_gbdt --neg-frac 0.5 --calibrate

# promote the winner to the bundle the API loads
python scripts/train_models.py --family hist_gbdt --neg-frac 0.5 --calibrate --set-champion

# comparison table for the slide (results/model_comparison.md)
python scripts/compare_models.py

# Bayesian tuning (needs optuna), then retrain with the best params
python scripts/tune_optuna.py --family hist_gbdt --trials 60 --timeout 720
python scripts/train_models.py --family hist_gbdt --params-file artifacts/best_params_hist_gbdt.json
```

Families: `logistic, extra_trees, hist_gbdt, nystroem_sgd, mlp, sgd_huber`.
Every run is appended to `results/model_runs.csv`; the dev/holdout trip split is frozen in
`artifacts/split.json` so all families are scored on the same trips.

## What is deliberately missing (do not demo it as done)

- HMM/Viterbi decoder — `src/api/main.py` marks the hook; C owns it.
- Two-laptop deployment hardening, screenshots, report — D.
- `candidate_miss_rate` is computed in the feature build but only surfaced by the API once
  C wires it into the response (it is already present in `results/feature_build_report.json`).
