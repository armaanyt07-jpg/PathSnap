# PathSnap — Work split for a team of 4

**What is actually graded:** ≥5 trained & compared models, Bayesian hyperparameter search, HMM/Viterbi decoder, FastAPI backend on Laptop 1, UI on Laptop 2, Folium maps, one consolidated PDF report, public GitHub repo. Every member must be able to explain every part in the viva.

**Dataset facts (checked):** 80 train trips, 10 student-test trips, 10 evaluator trips (not provided). ~208k GPS observations in train, 800–7,200 per trip. Each trip ships its own road network (~89k nodes, ~166k arcs — the network differs per trip, so build the spatial index per trip, don't assume one global graph). Files: `.track` (lon, lat, t), `.nodes` (lon, lat), `.arcs` (from_id, to_id, row index = arc_id), `.route` (arc_id sequence).

---

## 1. The split

| | Owner | Workstream | Code they own | Report sections |
|---|---|---|---|---|
| **A** | Data & spatial lead (strongest Python + geometry) | Parsing, label mapping, candidate generation, features, EDA, Folium | `src/io.py`, `geo.py`, `labels.py`, `candidates.py`, `features.py`, `viz.py` | §2.2 Dataset & EDA, §6 Folium |
| **B** | Modelling lead | ≥5 models, group-aware CV, comparison, Bayesian tuning | `src/models/`, `src/tuning/`, `scripts/run_experiments.py` | §3 Model comparison, §4 Bayesian |
| **C** | Sequence & backend lead | HMM + Viterbi, gap handling, FastAPI, server-side scoring | `src/hmm/`, `src/api/`, `scripts/run_decode.py` | §5 HMM/Viterbi, §7 API half |
| **D** | Integration lead | Metrics module, UI, 2-laptop deployment, screenshots, repo/report assembly, reproducibility | `src/metrics.py`, `src/ui/`, `scripts/run_all.sh`, README | §1 Overview, §7 UI half, assembly |

**Why this cut:** it follows the dependency chain data → features → models → decoder → API → UI. Each person owns a layer end-to-end (code + experiments + report text), so nobody writes about work they didn't do — which is exactly what the viva checks.

**Load balance note:** if one of you is much weaker at ML, give them **D** and have them co-own one model with B (so they can defend it). Do **not** give the UI to one person and all modelling to another — B already carries the heaviest analysis load.

---

## 2. Interfaces to lock in the first hour (this is what makes parallelism work)

Write these into `docs/contracts.md` before anyone codes deep.

1. **Feature table** — one row per (observation, candidate arc):
   `trip_id, obs_idx, timestamp, lon, lat, arc_id, cand_rank, perp_dist_m, along_arc_m, heading_diff_deg, arc_len_m, prev_gap_m, prev_speed_mps, label`
   Delivered by A as `features_{split}.parquet`. B and C never parse raw files themselves.
2. **Model bundle** — B exposes `predict_proba(features_df) -> float[n_obs, n_cands]`, saved as `artifacts/model_bundle.pkl` + `config.json`. C consumes only this.
3. **Prediction output** — per observation: chosen `arc_id`, candidate ids, probabilities; plus `route` = ordered arc sequence. C returns this same shape from `/predict`.
4. **API contract** — `GET /trips` → `[{trip_id}]`; `POST /predict {trip_id}` → `{observations, predicted_route, baseline_route, metrics, timings}`. D builds the UI against a stub from minute one.

**Two decisions to settle in Block 0** (document them in the report, don't discover them at 9 pm):
- **Label mapping:** `.route` has far fewer arcs than GPS points (e.g. 525 arcs / 1315 points), so each observation must be assigned to a reference arc (nearest arc *among the reference route*, checked for monotonic progress along the route). Getting this wrong silently ruins every metric.
- **"Macro F1":** with thousands of arc-id classes, macro F1 over all classes is near-zero and meaningless. Pick a defensible definition (macro over arcs present in the test trips, or candidate-level binary F1), report both, and justify it in §3.

---

## 3. Timeline (one-day hackathon; if you have more days, each block becomes a day)

**Block 0 — 0:00–1:15, all four together**
Repo + folder skeleton + `requirements.txt` + empty report doc. Agree the four contracts. Each person gets a *vertical smoke test* running:
A: parse all 4 file types for 3 trips + label sanity check. B: harness runs 2 fast models on dummy features. C: Viterbi runs on uniform emissions for 1 trip. D: `metrics.py` + Streamlit UI against a stub API.

**Gate 1 (1:15):** features v0 exist for 3 trips; the dummy pipeline runs end to end. If not, cut scope now, not later.

**Block 1 — 1:15–3:30, parallel build**
A: full feature build for 80 + 10 trips, candidate miss-rate report, EDA plots. B: 5 models with `GroupKFold` on trip id, results logged to CSV. C: HMM with real emissions (use B's first LogReg), gap/disconnection handling. D: metrics wired in, UI ↔ real API, **two-laptop networking tested with the real IP/port**.

**Gate 2 (≈3:30, first-phase evaluation):** one trip demoed end-to-end — UI → API → model → HMM → Folium map. A thin complete slice beats a perfect model here.

**Block 2 — 3:30–5:30**
B: finish comparison table + Optuna (time-boxed, ~30–50 trials) + tuned-vs-untuned. C: classifier-only vs HMM comparison + API hardening. A: Folium maps for §6, label audit. D: screenshots, EDA sections into the report, integration fixes.

**Block 3 — 5:30–7:30**
Freeze code. Full run on the 10 student-test trips → CSVs, figures, tables. Everyone writes their own report sections; D assembles and patches the seams. Repo cleanup: no hardcoded paths, no secrets, seed + configs + saved weights committed, README with run instructions.

**Block 4 — final 60–90 min**
Dry-run the demo twice on the two laptops (including a cold start). 15-minute teach-back: each person explains one *other* person's component. Export the PDF, verify the repo is public and self-contained, submit.

---

## 4. Rules that prevent the usual 4-person failure modes

- **Never block on the data layer.** A ships features for 3 trips within the first hour; B, C and D code against explicit schemas, never against "when A is done".
- **Folder ownership.** You may read anyone's code, but only edit your own folder; cross-cutting changes go through the owner. Avoids merge hell on one shared branch.
- **Small commits continuously**, conventional messages, everyone must have a visible commit history in their area (the repo history is part of the integrity check).
- **Write report sections as the work happens**, not at the end. D's assembly pass is formatting and consistency, not authoring.
- **Log every experiment** (config, seed, metrics, timing) into `results/` from the first run — §3, §4 and the comparison table are generated from those files, not retyped.
- **Do not touch the 10 student-test trips for anything except model comparison**, and never cross-contaminate folds by observation (group by trip).

## 5. Report ownership map

| Section | Author | Needs from |
|---|---|---|
| §1 Overview / system diagram | D | all |
| §2.2 Dataset & EDA | A | — |
| §3 Model comparison (5 subsections + charts) | B | A's features, D's metrics |
| §4 Bayesian optimisation | B | B |
| §5 HMM / Viterbi | C | B's probabilities, A's candidates |
| §6 Folium visualisation | A | C's predicted routes |
| §7 API / UI / deployment | C (API) + D (UI) | each other |
| Assembly, formatting, screenshots | D | all |

## 6. Risks

| Risk | Owner | Mitigation |
|---|---|---|
| Label mapping turns out ambiguous → all metrics invalid | A | Validate in Block 0: monotonic route progression, candidate miss rate ≈ 0 on train |
| SVM/MLP too slow on ~2.5M candidate rows | B | Subsample negatives or rows per family, applied identically to all models; document it |
| HMM produces disconnected jumps | C | Transition = network adjacency + distance; fall back to classifier-only for that step; report both |
| Demo breaks on the second laptop | D | Test with real IP + CORS in Block 1; fixed port, model pre-loaded at startup; second dry run |
| Report written at the last minute | D | Skeleton in Block 0, sections filled per block, freeze at Block 3 |
| Viva: someone can't explain their own section | all | Gate 2 and Block 4 teach-backs; each person reviews one other section |
