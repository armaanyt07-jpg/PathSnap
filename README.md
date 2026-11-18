# PathSnap

Distributed map matching and route reconstruction for PES UE24CS352A.

## Current implementation status

This checkout includes Member A's route-blind features, Member B's six-family comparison harness, Member C's HMM/Viterbi decoder and FastAPI service, and Member D's metrics, dashboard integration, deployment scripts, and documentation. The UI uses real results from `results/experiments.csv` and `results/model_comparison.csv`; it also reads the optional per-trip HMM comparison JSON when available. The frontend keeps only allow-listed prediction fields and aggregate metrics; reference labels and geometry remain server-side.

The repository contains actual model comparison outputs in `results/`; the Optuna study has not been run, and trained model bundles are local artifacts that are not committed. A compatible bundle is required for live `/predict`. This checkout has no live demo screenshots or dataset-backed HMM comparison JSON. See [the report outline](docs/report_outline.md) for verified evidence and remaining requirements.

## Requirements and data

- Python 3.10+ on Linux or Windows PowerShell 5.1+ / PowerShell 7+.
- Dataset root containing `train/` and `student_test/`. This checkout includes 80 training trips and 10 reserved student-test trips.
- A trained bundle at `artifacts/model_bundle.joblib` before `/predict` can serve predictions.
- Install the application requirements in a virtual environment:

```bash
cd ~/projects/PathSnap
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PATHSNAP_DATA_DIR="$PWD"
```

On Windows PowerShell, create and activate the environment with:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PATHSNAP_DATA_DIR = (Get-Location).Path
```

`PATHSNAP_DATA_DIR` can point to a different dataset root. Without it, PathSnap searches `<repo>/data`, the repository root, then the team's local materials fallback. Setup, backend, and combined startup scripts validate the dataset layout, all ten reserved trips, and each trip's `.track`, `.nodes`, `.arcs`, and `.route` files.

## Start on one laptop

Linux (a model bundle is needed for predictions):

```bash
export PATHSNAP_DATA_DIR="$PWD"
bash scripts/run_all.sh
```

Open `http://localhost:8501`. The combined script starts the API on `127.0.0.1:8000`, checks `/health`, then starts Streamlit. Ctrl+C stops the services.

Windows PowerShell:

```powershell
.\scripts\start_all.ps1 -DataDir $PWD
```

Open `http://localhost:8501`. The combined launch binds both services to loopback, waits for `/health`, then runs Streamlit. Ctrl+C stops both services.

## Start on two laptops

On Laptop 1, which has the dataset and model bundle:

```bash
export PATHSNAP_DATA_DIR="$PWD"
export PATHSNAP_API_HOST=0.0.0.0
export PATHSNAP_API_PORT=8000
bash scripts/run_api.sh
```

Windows PowerShell equivalent:

```powershell
.\scripts\start_backend.ps1 -DataDir $PWD -HostAddress 0.0.0.0 -Port 8000
```

On Laptop 2, set Laptop 1's private LAN address:

```bash
export PATHSNAP_API_URL="http://<LAPTOP_1_PRIVATE_IP>:8000"
bash scripts/run_ui.sh
```

Windows PowerShell equivalent:

```powershell
.\scripts\start_frontend.ps1 -ApiUrl "http://<LAPTOP_1_PRIVATE_IP>:8000"
```

Open `http://localhost:8501` on Laptop 2. Restrict inbound TCP 8000 on Laptop 1 to Laptop 2's private address; do not expose the unauthenticated API publicly. Windows and Ubuntu UFW examples are in [docs/demo.md](docs/demo.md). The Streamlit server makes API calls from Laptop 2, so its configured API URL must be reachable from there.

## API contract

- `GET /health` reports readiness and model availability.
- `GET /models` returns the scoring models the server can use (`champion` plus each saved family).
- `GET /trips` returns the available trip IDs.
- `POST /predict` accepts `{"trip_id":"00000056"}` and returns observations, predicted route geometry, aggregate metrics, and timings. Add `"model":"mlp"` to score with a comparison model instead of the champion. HMM extensions include route segments and classifier-only aggregate metrics.

Reference labels and reference geometry stay on the server. For details see [docs/contracts.md](docs/contracts.md).

## Data and model build

`scripts/run_phase1.sh` runs inventory, feature generation, baseline training, and an offline map. It is a build command and can replace generated feature files and the baseline artifact; it also appends a row to `results/experiments.csv`. Do not run it over existing artifacts you need to preserve. The per-step commands are in the script headers and `PathSnap_Team_Plan.md`.

## Modelling (workstream B)

```bash
# one family, identical folds for every family
python scripts/train_models.py --family hist_gbdt --neg-frac 0.5 --calibrate

# promote the winner to the bundle the API loads
python scripts/train_models.py --family hist_gbdt --neg-frac 0.5 --calibrate --set-champion

# comparison table for the slide (results/model_comparison.md)
python scripts/compare_models.py

# figures for report §3.3 (results/figures/*.png)
python scripts/make_figures.py

# Bayesian tuning (needs optuna), then retrain with the best params
python scripts/tune_optuna.py --family hist_gbdt --trials 60 --timeout 720
python scripts/train_models.py --family hist_gbdt --params-file artifacts/best_params_hist_gbdt.json
```

Families: `logistic, extra_trees, hist_gbdt, nystroem_sgd, mlp, sgd_huber`.
Every run is appended to `results/model_runs.csv`; the dev/holdout trip split is frozen in
`artifacts/split.json` so all families are scored on the same trips. Top-1 is reported twice,
over all labelled observations and over observations whose reference arc is actually in the
candidate set, so candidate-generation misses are not charged to the classifier.

## Tests

Install test dependencies in the project virtual environment and run:

```bash
python -m pip install -r requirements-test.txt
PYTHONPATH=src python -m unittest discover -s tests -v
```

The API contract tests require the application dependencies; they are skipped when those are unavailable. `python -m compileall -q src scripts tests` and `bash -n scripts/run_api.sh scripts/run_ui.sh scripts/run_all.sh` provide syntax checks. PowerShell scripts are documented but have not been executed on Windows in this environment.

## Repository map

```text
src/pathsnap/       parsing, geometry, candidates, features, metrics, API glue
src/pathsnap/model_zoo.py    B   six model families behind predict_scores
src/pathsnap/evaluation.py   B   trip-level split, lattice filter, run log
src/hmm/            HMM/Viterbi decoder
src/api/            FastAPI service
src/ui/             Streamlit frontend
scripts/            feature/model pipeline and service startup scripts
docs/contracts.md   frozen feature, model, and API contracts
docs/demo.md        two-laptop setup and demo checklist
docs/report_outline.md  report sections and evidence checklist
```

## What is deliberately missing (do not demo it as done)

- The Optuna study has not been run; the harness and search spaces exist, but no tuned-vs-untuned row exists yet.
- Bayesian tuning has not been run. The cold-start two-laptop demo and real screenshots remain to be captured.
