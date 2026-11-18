# Workstream B — Modelling implementation plan

Owner: Armaan (B). Status: planning only. Nothing in this document is built yet, and nothing is committed or pushed.

Current facts on disk (checked, not assumed):

- `src/pathsnap/baseline.py`, `scripts/train_baseline.py`, `src/pathsnap/metrics.py` exist and are committed as `048aef9`.
- `data/processed/` and `artifacts/` are **empty** — the feature build has not been run in this clone. Step 0 depends on the 10:25 kickoff build.
- Installed: numpy 2.0.2, pandas 2.3.3, scikit-learn 1.6.1, scipy 1.13.1, Python 3.9.6.
- **Not** installed: optuna, lightgbm, xgboost. The model zoo is therefore scikit-learn-only unless a dependency is approved.

---

## 1. Boundaries

B may create or edit:

- `src/pathsnap/model_zoo.py` (new), `src/pathsnap/evaluation.py` (new)
- `scripts/train_models.py`, `scripts/tune_optuna.py`, `scripts/compare_models.py` (new)
- `scripts/train_baseline.py`, `src/pathsnap/baseline.py`, `src/pathsnap/metrics.py` (existing, B-owned)
- `configs/*.json` model section, `docs/report_B.md` (new)
- its own result files: `results/model_runs.csv`, `results/model_comparison.csv`, `results/baseline_metrics.json`

B must not edit:

- `src/pathsnap/features.py` and the candidate/label code — A owns these; a change there invalidates every number B reports.
- `src/api/main.py` — C owns the decoder hook; B only produces the artifact it loads.
- `src/ui/app.py` — D owns it.

**Hard interface B must preserve:** `artifacts/model_bundle.joblib` stays loadable by `baseline.load()` and scoreable by `baseline.predict_scores(model, df) -> np.ndarray`, because `src/api/main.py` already calls exactly that. Whatever wins the comparison, the champion is written to that path in that shape. Losers go to `artifacts/candidates/{name}.joblib`.

---

## 2. Definition of done

**By 11:30 (phase-1 demo):**

1. `results/baseline_metrics.json` exists with honest grouped CV, per-trip breakdown, train time and inference ms.
2. A stated primary metric, written down, with the numbers that justify it.
3. Two model families compared on identical folds, not one — enough to prove the comparison harness works and that "≥5 models" is a real pipeline rather than a promise.
4. One comparison-table row handed to D for the slide, with a one-line interpretation B can defend in viva.
5. `artifacts/model_bundle.joblib` present so C and D can demo against a real model.

**After 11:30 (final phase):**

6. ≥5 diverse model families through one harness, one table, paired results.
7. Optuna study per promising family, with a tuned-vs-untuned comparison.
8. Champion calibrated, because C's Viterbi multiplies probabilities.
9. `docs/report_B.md` with method, results, threats to validity, and the AI-assistance disclosure.

---

## 3. Phase-1 run sheet

Times assume the 10:25 kickoff build finished; shift by whatever it overran.

| Time | Step | Gate |
|---|---|---|
| 10:32–10:38 | Load `features_train.parquet`, sanity-check: rows, trips, label balance, labelled-observation count, candidate miss rate. Compare against A's build report. | Counts and miss rate match A's. If not, stop and diagnose before training — every number downstream is suspect. |
| 10:38–10:50 | Run the baseline harness with `--neg-frac 0.5`. Read CV per fold, per-trip table, train time, inference ms. | CV and student-test gap is explainable; no fold is an outlier without a reason. |
| 10:50–11:02 | Second family (HistGradientBoosting if it fits in memory, else ExtraTrees) through the same folds and the same subsample rule. | Both models scored on identical validation rows. If the second family is still running at 11:02, kill it — a partial second row is worth less than a clean first row. |
| 11:02–11:10 | Freeze. Write the slide row and three bullets of interpretation. | B can say what the number means and what it does not. |
| 11:10–11:20 | Support: confirm the API still loads the artifact after the champion is written. | `/predict` returns 200 against the final bundle. |
| 11:20–11:30 | Demo. | — |

Phase 1 deliberately stops at two families. Five families before 11:30 means five half-verified numbers, and the first question in viva will be "why is this one better" — which needs a harness, not more rows.

---

## 4. Evaluation protocol (settled before any tuning)

This is the part that has to be right, because it is what B defends.

**Split design.** Outer split of the 80 training trips into 64 dev / 16 held-out, grouped by `trip_id`, seeded, written once to disk so it cannot drift. All model selection, feature-set questions and Optuna trials happen on the 64. The 16 are scored once per frozen candidate. Then the champion is refit on all 80 for the shipped artifact. `student_test` (10 trips) is touched once per frozen candidate and every touch is logged — repeated peeking turns it into a training set and the final number becomes fiction.

**Why not just GroupKFold on all 80.** Tuning on the same folds you report from inflates the reported score. With 64/16 the report is honest and it costs one extra column in the table, not extra hours.

**Negative subsampling.** Negatives are thinned inside the training fold only, never inside validation. This is already correct in `train_baseline.py`; keep it that way in the new harness and make it a shared helper so the rule cannot drift between scripts.

**Pairing.** Every model sees identical rows in identical folds in identical order. That makes "model A beats model B" a paired comparison, and lets the table include a per-trip win count (on how many of the 16 held-out trips A beats B) — cheap, and far more convincing than a third decimal place on an aggregate.

**Metric policy.**

- Primary: **observation-level Top-1** — the chosen arc equals the reference arc, aggregated over labelled observations, one vote per observation rather than per candidate row. Rows are not the unit; observations are.
- Report `top1_all` and `top1_covered` separately. The gap between them is A's candidate miss rate, not B's error, and conflating them lets B take credit or blame for a data problem.
- Secondary: **candidate log loss**. It is the metric that matters for what C consumes: Viterbi multiplies probabilities along a path, so a well-ranked but badly-scaled score produces a bad route even at good Top-1.
- Reported but not used for selection: macro F1, weighted F1, candidate precision/recall at 0.5. Macro F1 over thousands of arc classes with a handful of instances each is unstable by construction; state that once in the report and move on rather than trying to make it look healthy.
- Not B's to optimise: candidate miss rate. It is a property of A's candidate generation. Report it, never tune against it.

**Leakage checklist to run before publishing any number:** folds grouped by trip, no trip in two folds; scaler and calibrator fit inside the fold; subsampling only on training rows; no target-derived feature; held-out trips never seen during selection.

---

## 5. Phase 2 — model zoo

Five families that differ in *inductive bias*, not five settings of one model. That is the honest reading of "compare at least five models".

| # | Model | Family / bias | Why it belongs | Expected cost | Calibration |
|---|---|---|---|---|---|
| 1 | Logistic regression | linear, additive | baseline, interpretable coefficients, already shipped | seconds | reasonable |
| 2 | ExtraTrees | bagged trees, axis-aligned | captures thresholds in `perp_dist_m`, `rel_pos` that a linear model cannot | moderate | poor, needs it |
| 3 | HistGradientBoosting | boosted trees | usually the accuracy ceiling on tabular data of this size | moderate | poor near 0/1 |
| 4 | Nystroem + SGD (or kNN on a subsample) | kernel / instance-based | local similarity structure; genuinely different failure mode | moderate | needs it |
| 5 | MLP | neural, smooth interactions | tests whether interactions matter beyond trees | slow, needs scaling | moderate |
| 6 (optional) | SGD `modified_huber` or Gaussian NB | cheap diversity | a sixth row for near-zero cost if time allows | seconds | varies |

Each exposes the same `predict_scores(df)` contract so the harness, the API and the comparison table are family-agnostic. No model gets a bespoke script.

**Diversity is the deliverable, not the winner.** If five families land within one point of each other, that is a finding and it gets reported as one: the problem is candidate-generation-limited, not model-limited. That is a much stronger viva answer than "the boosted tree won by 0.4".

---

## 6. Phase 2 — tuning

- Optuna TPE, seeded, with `MedianPruner` on intermediate fold scores; study persisted to `artifacts/optuna_{family}.db` so a crash costs a restart, not the run.
- Objective: mean candidate log loss on the 64 dev trips with 3 grouped inner folds. Not Top-1 — Top-1 is a step function of the scores and gives the sampler almost no gradient; log loss is smooth and is the quantity Viterbi actually depends on.
- Per-family search space kept narrow and justified in the report: regularisation for linear; depth/leaf size/`max_features` for trees; learning rate/iterations/leaf count/L2 for boosting; hidden sizes/alpha for the MLP; gamma/components/alpha for the kernel model.
- Hard wall-clock cap per family, and a trials cap. A timeout that stops the study is a feature: it keeps the post-demo window bounded.
- Tuned-vs-untuned is a required row pair in the table. If tuning does not help, saying so with evidence is worth more than hiding it.
- Tuning happens on dev only. The 16 held-out trips score the tuned model exactly once.
- Dependency note: optuna is not installed. It is one new package, added to `requirements.txt` and installed with the project venv, only with your go-ahead. No other new dependency is planned; the zoo is otherwise pure scikit-learn.

---

## 7. Phase 2 — calibration

Needed because of what sits downstream. C's Viterbi sums log-probabilities across roughly a thousand observations; a tree ensemble that outputs 0.02 and 0.98 for the same evidence as a logistic model's 0.3 and 0.7 will produce a different, worse route even with identical rankings.

- Fit the calibrator on out-of-fold scores only, never on rows the model trained on.
- Compare log loss and Brier before/after, per family.
- Note in the report: monotone calibration (Platt, isotonic) cannot change the argmax, so Top-1 is unchanged by construction. Calibration buys route quality for C, not accuracy for B's table. That distinction is exactly the kind of thing a viva examiner probes.
- Ship the champion calibrated, and record the calibrator inside the bundle so `predict_scores` still returns calibrated probabilities with no API change.

---

## 8. Deliverables and files

New:

- `src/pathsnap/model_zoo.py` — registry of families behind one interface.
- `src/pathsnap/evaluation.py` — split construction, subsampling helper, scoring, paired comparison.
- `scripts/train_models.py` — one command, any family, identical folds.
- `scripts/tune_optuna.py` — study per family, resumable.
- `scripts/compare_models.py` — writes `results/model_comparison.csv` + the markdown table for the slide.
- `docs/report_B.md` — method, results, threats to validity, AI-assistance disclosure.

Result files:

- `results/model_runs.csv` — one row per run, explicit schema, never mutated mid-run.
- `results/model_comparison.csv` — the table that drives the slide and the report.
- `results/baseline_metrics.json` — kept as the reference point.

Do **not** change the schema of the existing `results/experiments.csv` while appending to it; a new file with a fixed schema is cheaper than a broken history.

---

## 9. Risks

| Risk | Signal | Response |
|---|---|---|
| Features late or inconsistent | parquet missing, or counts differ from A's report | Do not train on a suspect table. Diagnose with A first; a wrong number is worse than a late one. |
| Second family too slow for phase 1 | still fitting at 11:02 | Kill it. Ship one honest row. |
| Memory pressure on ~2M rows | swap, or a killed process | Drop to a 12-trip subsample for *development* only, and say so; never report a subsample as the final number. |
| Optuna overruns | study still running after the cap | Ship untuned champion, report the unfinished study as such. |
| Tuning leaks | held-out score much worse than dev | Expected to some degree; if the gap is large, the split is wrong, not the model. Recheck fold grouping before touching hyperparameters. |
| Champion swap breaks the API | `/predict` errors after artifact write | The interface is frozen for exactly this reason; verify the endpoint after every bundle write. |

---

## 10. Open decisions

1. Does A's kickoff build land before 10:35, and do the counts match across members?
2. Approve `optuna` as the one new dependency (it is post-11:30 work either way).
3. Confirm the student-test touch budget: one scoring per frozen candidate, logged. If the team wants more freedom there, the honest alternative is to stop calling it held-out.
4. Confirm the slide row format with D so B's numbers drop straight into the deck without transcription.

---

## Appendix — commands, to be run only when authorised

Nothing below has been run.

```
# after the shared kickoff build, in this clone
.venv/bin/python scripts/train_baseline.py --neg-frac 0.5

# later, phase 2
.venv/bin/pip install optuna
.venv/bin/python scripts/train_models.py --family extra_trees --neg-frac 0.5
.venv/bin/python scripts/tune_optuna.py --family hist_gbdt --trials 60 --timeout 720
.venv/bin/python scripts/compare_models.py
```
