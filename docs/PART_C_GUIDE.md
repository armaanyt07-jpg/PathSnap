# Your Part C: HMM/Viterbi + API

Work in this cloned repository, not the preliminary files in its parent directory.

## What you own

- `src/hmm/decoder.py`: candidate states, directed-network transitions, Viterbi, gap handling.
- `src/api/service.py`: A/B adapters, reference-free feature generation, aggregate scoring using D's metrics.
- `src/api/main.py`: FastAPI endpoints and startup model loading.
- `scripts/run_decode.py`: reproducible classifier-only versus HMM outputs.
- Report section 5 and the API half of section 7.

## First run (PowerShell, from this repository)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install httpx
$env:PYTHONPATH = "$PWD\src"
$env:PATHSNAP_DATA_DIR = "$PWD"
python scripts/run_decode.py --demo
python -m unittest discover -s tests -v
```

The synthetic demo needs only Python; its expected route is `[10, 11, 12]`.
It demonstrates bridging arc 11 even though neither GPS observation selects it.
It is not a measured result on the supplied dataset.

On this Codex machine, ordinary `python` resolves to a nonworking Windows Store alias.
For the already-installed local test environment, use:

```powershell
$pythonExe = "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$env:PYTHONPATH = "$PWD\src;$PWD\..\.deps"
& $pythonExe scripts/run_decode.py --demo
& $pythonExe -m unittest discover -s tests -v
```

## Connect B's model and run the API

B must provide the trusted `artifacts/model_bundle.joblib` and its matching library versions.
The existing `pathsnap.baseline.predict_scores(model, frame)` adapter selects FEATURE_COLUMNS
and returns one probability per candidate row. If B changes the model interface, change this adapter.

```powershell
$env:PYTHONPATH = "$PWD\src"  # append ..\.deps when using the Codex runtime above
$env:PATHSNAP_DATA_DIR = "$PWD"  # repo contains train and student_test
$env:PATHSNAP_CORS = "http://localhost:8501,http://LAPTOP2_IP:8501"
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

With the Codex runtime use `& $pythonExe -m uvicorn ...` instead of `python`.
Open http://localhost:8000/docs for interactive requests.
The model loads once at startup. A missing model produces model_loaded=false and HTTP 503 on prediction.
Restart after adding/replacing the model. Only load a joblib file from your own trusted team.

On Laptop 2, replace LAPTOP1_IP with the IPv4 address shown by `ipconfig` on Laptop 1:

```powershell
Invoke-RestMethod http://LAPTOP1_IP:8000/health
Invoke-RestMethod http://LAPTOP1_IP:8000/trips
Invoke-RestMethod -Method Post http://LAPTOP1_IP:8000/predict -ContentType application/json -Body '{"trip_id":"ACTUAL_ID_FROM_TRIPS"}'
```

Both laptops must be on a network allowing device-to-device traffic. If prompted, permit Python on the private network. Do not expose this unauthenticated classroom API publicly.
Take the successful response screenshot for section 7. Cross-laptop testing has not been performed here.

## Run actual comparisons after receiving the model

```powershell
python scripts/run_decode.py --split train --trip TRAIN_TRIP_ID
python scripts/run_decode.py --split student_test
```

Use held-out training trips to choose decoder parameters; do not tune on student-test trips.
The script writes per-trip JSON, timings, decoder settings, and a comparison summary to `results/part_c/`.
The baseline and HMM share the same candidates and model scores. No actual trained-model comparison
has been run as part of this implementation because the repository has no saved model bundle.

## Explain it in the viva

A GPS point can be close to two roads. The classifier might choose road A, then an unrelated road B,
then A again. Your decoder looks at the entire sequence and asks which physically reachable sequence
best agrees with the classifier scores.

1. **Hidden state:** a candidate directed road arc, with the GPS projection's position along it.
2. **Emission:** B's positive-class score, floored at 1e-12 then normalized within each observation.
   These are discriminative emission potentials; do not claim the classifier directly estimates a
   generative P(GPS | road). All-zero scores become uniform.
3. **Transition:** travel along the current arc, shortest directed road path, then along the next arc.
   Score `-|network_distance - GPS_displacement| / beta_m`, normalized over reachable next candidates.
   Paths exceeding `55 m/s * elapsed_seconds + 50 m` are rejected by default.
4. **Viterbi:** `V[t,j] = log(emission[t,j]) + max_i(V[t-1,i] + log(transition[i,j]))`.
   Store the maximizing predecessor, then backtrack. Logs avoid products underflowing to zero.
5. **Gaps:** missing candidates output an unmatched observation. Time gaps over 120 seconds or an
   entirely unreachable next layer finish the current segment and restart from emissions. At the
   restart, scores initially match classifier-only; later evidence can revise the choice.
6. **Route:** insert connecting arcs between selected states. Preserve segment boundaries instead
   of inventing a road across a gap. A repeated arc can legitimately recur after a directed loop.

Viterbi itself is O(T*K^2) time and O(T*K) backpointer memory; shortest-path searches add graph cost.
This first implementation uses bounded Dijkstra per candidate pair and has not been benchmarked on
all trips. If latency is high, cache bounded source-node searches or precompute local transitions.
The numeric defaults are starting choices, not tuned parameters.

## Handoff notes for A, B and D

- **A:** inference calls your feature builder with `route=None`. This prevents reference-arc injection.
  Server-side label mapping happens only after decoding. Candidate misses are retained as real errors.
  Your current label mapper remains a dependency; audit its monotonicity separately.
- **B:** deliver model + config and exact package versions. Confirm positive-class score ordering.
  Training's appended candidates may create a distribution shift because cand_rank is a feature;
  assess this on reference-free validation candidates.
- **D:** existing `/trips` and `/predict` fields remain available. Additional fields include predictions,
  baseline_metrics, route_segments, predicted_lines, breaks, and decoder_config.
  Draw each `predicted_lines` entry separately. Legacy `predicted_line` shows only the segment with
  most vertices when breaks exist, so the old UI does not draw a false straight connection.
  `predicted_route` flattens segments for compatibility; route_segments is authoritative for connectivity.
  Unavailable metrics are omitted so your `.get(..., nan)` display still works.
- **D:** metrics.classification_metrics currently computes macro F1 over the union of true and predicted
  classes, although docs/contracts.md describes reference classes only. The API labels the actual
  definition; reconcile the helper/report definition before final evaluation.
- **D:** route_continuity here measures adjacent observation choices, not inserted bridge arcs.
  Candidate F1/log loss describe the same classifier in both outputs, not HMM posterior probabilities.

## Report section 5 / section 7 worksheet

Describe the six algorithm points above in your own words. Add a table from actual logged runs:
trip, baseline top-1, HMM top-1, baseline macro F1, HMM macro F1, both continuity values,
decode milliseconds, gap count. Explain at least one improvement and one failure using real maps.
Do not copy the synthetic demo into the experimental-results table.

For section 7, document startup model loading, GET /health, GET /trips, POST /predict,
request validation, server-only labels, error responses, CORS origins, and the two-laptop IP/port setup.
Attach actual successful-response screenshots after the real model is connected.

FastAPI implementation follows its official lifespan and test-client documentation:
https://fastapi.tiangolo.com/advanced/events/
https://fastapi.tiangolo.com/tutorial/testing/
