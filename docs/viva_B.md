# Part B — modelling: what is implemented, and how to defend it

Owner: Armaan. Scope: everything between the feature table and the decoder — model
families, training protocol, evaluation, comparison, tuning harness, champion export.

All numbers below come from one full run on 2026-11-18 over all 80 training trips and
the 10 student-test trips, and are reproducible from `results/model_runs.csv`.

---

## 1. Where this sits in the system

```
raw .track/.nodes/.arcs/.route
  -> A: candidate generation + feature table      (2.1M candidate rows)
  -> B: per-candidate probability                 (this part)
  -> C: HMM/Viterbi decoding                      (sequence choice)
  -> D: API + Streamlit + maps
```

B does not decode routes and does not build features. B answers one question:
*given an observation and one candidate arc, what is the probability that this arc is
the right one?* C turns thousands of those probabilities into one continuous route.

That separation matters at viva: if the route is wrong, the failure can be in candidate
generation (the true arc was never offered), in the classifier (it ranked the true arc
low), or in decoding (it ranked it first but the path was incoherent). The metrics are
designed so those three are separable.

---

## 2. The five families

Five deliberately different inductive biases, not five settings of one model. That is
what "compare at least five models" should mean.

| # | Family | Syllabus category | Configuration used | Why it is in the comparison |
|---|---|---|---|---|
| 1 | Hist Gradient Boosting | Boosting | `learning_rate 0.1`, `max_iter 250`, `max_leaf_nodes 31`, `min_samples_leaf 50`, `l2 1.0`, `class_weight balanced` | Usually the accuracy ceiling on tabular data at this scale; histogram binning keeps 1M+ rows tractable without one-hot encoding |
| 2 | MLP | Neural networks | `(64, 32)` hidden, `alpha 1e-3`, `batch 4096`, `lr 1e-3`, 20 epochs, standardised inputs | Tests whether smooth feature interactions beat axis-aligned splits; a genuinely different function class |
| 3 | Extra Trees | Tree-based / ensemble | 200 trees, `max_depth 32`, `min_samples_leaf 25`, `max_features sqrt`, `class_weight balanced_subsample` | Bagging instead of boosting: high variance reduction, no sequential dependency. Leaf size and depth are capped to keep the serialised forest in the low hundreds of MB — unconstrained 300-tree forests ran to ~2 GB |
| 4 | Nystroem + SGD | Kernel-based / SGD | 64 Nystroem components, `gamma 0.1`, SGD `log_loss`, `alpha 1e-4`, `average=True` | Kernel methods do not scale to 1M rows directly; random-feature approximation maps them into 64 dimensions, then a linear model in that space. Tests local similarity structure |
| 5 | Logistic Regression | Classification baseline | StandardScaler + L2, `C 1.0`, `class_weight balanced`, 400 iterations | Interpretable reference point, near-instant to fit, and the model that was shipped in the phase-1 skeleton |

A sixth, cheap `SGDClassifier(loss="modified_huber")` was also run as a floor check. It
is the same linear family as #5, so it is reported in `model_runs.csv` but left out of
the five-family table.

Every family is behind one interface: `fit(frame) -> estimator` and
`predict_proba(X)[:, 1]`. Because of that, the API, the comparison harness and the
decoder never branch on model type — the contract in `docs/contracts.md` §2 holds for
all five.

---

## 3. Training protocol

**Data.** 80 training trips, 208,417 observations, 2,099,018 candidate rows at
`k = 10`. Student-test: 10 trips, 19,160 observations, 191,614 rows.

**Class imbalance.** Roughly 1 positive per 9 negatives. Two changes: `class_weight
balanced` inside the model, and negatives thinned to 50% *inside training folds only*.
Thinning is for speed; `class_weight` keeps the effective weighting honest; calibration
afterwards repairs the probability scale, which matters because C multiplies them.

**Lattice rule.** The feature build appends a synthetic row at `cand_rank == 10` when
candidate generation misses the reference arc. Those rows never exist at inference, so
they are excluded from both training and scoring; observations with no positive
candidate are dropped from training only (they carry no "which arc" signal) and kept in
validation (where they must count as errors). See §6, finding 4.

**Split.** The 80 trips are split 64 dev / 16 held-out, grouped by `trip_id`, generated
once and frozen in `artifacts/split.json`. All model selection, hyperparameters and
comparisons happen on the 64. The 16 are scored once per frozen candidate. The champion
is then refit on all 80 for the shipped artifact. Student-test is used only for the
final report row.

**Cross-validation.** `GroupKFold`, 5 folds on the dev trips. Grouping by trip is
non-negotiable: two trips share no road network, but a random split would still leak
trip-specific geometry and inflate fold scores.

**Seeds.** 42 throughout; the split, the subsample and every estimator share it, so a
rerun reproduces the table.

---

## 4. What is measured

| Metric | Definition | Role |
|---|---|---|
| `top1_accuracy` | fraction of labelled observations whose highest-scoring candidate is the reference arc | **Primary.** Observation-level, not row-level |
| `top1_covered` | same, restricted to observations whose reference arc is actually among the 10 candidates | Model quality, isolated from candidate-generation quality |
| `coverage` | share of observations whose reference arc is in the candidate set | Diagnostic. 1 − miss rate. A's property, not B's |
| `candidate_log_loss` | mean binary cross-entropy over candidate rows | Probability quality; what C's Viterbi consumes |
| `macro_f1` | unweighted mean F1 over arc classes present in the truth | Required by the brief, reported, not used for selection |
| `route_continuity` | fraction of consecutive predicted arcs that are identical or connected | Computed at the API level; a decoding metric, not a classifier metric |

Two deliberate exclusions: row-level accuracy is meaningless when 90% of rows are
negative, and `candidate_miss_rate` is never optimised by B — tuning against it would
mean tuning the candidate generator, which is A's component.

---

## 5. Results (full run, 80 training trips + 10 student-test trips)

| Family | CV Top-1 (dev, 5 folds) | Holdout Top-1 | Holdout Top-1 covered | Holdout log loss (raw → calibrated) | Student-test Top-1 | Inference ms / 1k rows |
|---|---|---|---|---|---|---|
| Hist Gradient Boosting | 0.7355 ± 0.171 | **0.7485** | **0.8622** | 0.1658 → **0.0906** | **0.8375** | 1.27 |
| MLP | 0.7312 ± 0.170 | 0.7426 | 0.8554 | 0.1698 → 0.0999 | 0.8263 | 0.08 |
| Extra Trees | 0.7303 ± 0.171 | 0.7399 | 0.8522 | 0.2107 → 0.0929 | 0.8266 | 2.37 |
| Nystroem + SGD | 0.7212 ± 0.166 | 0.7313 | 0.8424 | 0.2423 → 0.1175 | 0.8062 | 0.22 |
| Logistic Regression | 0.7090 ± 0.162 | 0.7276 | 0.8381 | 0.2854 → 0.1315 | 0.8110 | 0.04 |

Champion: **Hist Gradient Boosting**, calibrated, written to
`artifacts/model_bundle.joblib` and loaded by C's API unchanged.

How to read it in one breath: the best model is right on **83.8% of student-test
observations**, and on the held-out training trips it is right on **86.2% of the
observations it was actually offered the correct arc for**. The five families span three
points — the choice is worth less than the candidate generator.

---

## 6. Four findings worth leading with

**1. Candidate generation, not model choice, sets the ceiling.** One held-out trip,
`00000033`, has a 65.8% candidate miss rate over 5,844 observations. It alone accounts
for almost the entire gap between 0.7485 (all observations) and 0.8622 (covered): every
other held-out trip sits at 0.1–0.2%. A 66% miss rate is not GPS noise at `k = 10`; it
looks like a label/network mismatch and is worth A investigating. The honest headline is
therefore two numbers, never one.

**2. Five families land within three points.** Boosting wins by 0.6 points over the MLP
while costing 15× more per observation to score. If the project needed the cheapest
defensible model, the MLP or even logistic regression would be the defensible choice.
Saying that explicitly is stronger than pretending the winner is a landslide.

**3. Calibration halves log loss and cannot change Top-1.** Isotonic regression fitted
on out-of-fold dev scores is monotone, so per-observation argmax is preserved (a
1e-9·p term breaks the artificial ties that a piecewise-constant map creates). The
Top-1 columns are identical before and after; the log-loss column roughly halves. This
matters because C's Viterbi sums log-probabilities along a path — badly scaled scores
produce a different, worse route even when the argmax per observation is unchanged.

**4. A leak was found and removed.** Because the feature build injects the reference arc
as a synthetic candidate, scoring the full frame let the classifier pick the true arc
out of a candidate the generator never produced. Restricting scoring to the real lattice
moved one family's apparent holdout accuracy from 0.879 to 0.7485. The evaluation code
now asserts that Top-1 over all observations can never exceed Top-1 over covered
observations — that inequality is a cheap, permanent tripwire for this class of bug.

---

## 7. Numbers to quote, and numbers to avoid

Quote: 83.8% student-test Top-1; 86.2% held-out covered Top-1; 45–54% log-loss reduction
from calibration; 2.1M candidate rows; 64/16 trip split; 5 grouped folds; 1.3 ms per
1,000 candidate rows.

Avoid: any single per-trip accuracy as a headline (variance across trips is large — CV
std is ±0.17); the smoke-run numbers; anything from the evaluator split, which does not
exist for students.

---

## 8. Likely viva questions

**Why observation-level Top-1 rather than row accuracy?**
Every observation contributes 10 rows and exactly one is positive, so row accuracy has a
90% floor for a model that always says "no". Accuracy per observation is the quantity
the route actually depends on.

**Why does `top1_covered` exist?**
It separates "the model ranked the right arc low" from "the right arc was never offered".
Without it, a candidate-generation regression looks like a model regression, and B would
be defending A's numbers.

**Why GroupKFold and a 64/16 split instead of plain 5-fold?**
Grouping stops a trip appearing in both train and validation. The separate 16 keeps
model selection from happening on the folds we report, which is the usual way a
cross-validated score becomes optimistic.

**Why is the held-out gap (0.75) so much larger than the covered number (0.86)?**
One trip with a 65.8% candidate miss rate. Removing it brings coverage to ~99.5% and the
two numbers together.

**Why log loss as the tuning objective and not Top-1?**
Top-1 is a step function of the scores: a trial that improves every probability slightly
but changes no argmax scores identically. Log loss is smooth and is exactly what the
Viterbi decoder consumes.

**Why calibrate, and why can't it change accuracy?**
Tree ensembles output vote fractions, not probabilities. Isotonic regression fitted on
out-of-fold scores maps them to observed frequencies. It is monotone, so it cannot
reorder candidates within an observation.

**Does class weighting or negative thinning distort the probabilities?**
Yes — that is precisely why calibration is fitted afterwards, on out-of-fold scores, and
why the reported log loss is the calibrated one.

**Why is macro-F1 ~0.70 when accuracy is ~0.84?**
There are thousands of arc classes, many with a handful of observations. Macro-F1 averages
per-class F1, so rare arcs the model never predicts drag it down. It is reported because
the brief asks for it, and explicitly not used for selection.

**Why these five and not SVM, kNN or XGBoost?**
Plain SVM does not scale to a million rows; kNN at 10 candidates per observation over a
million training rows is too slow at inference. XGBoost and LightGBM are not installed,
and Hist Gradient Boosting is the same algorithm family with histogram binning. The five
chosen span linear, bagged, boosted, kernel and neural — that is what "five families"
should mean.

**How do you know there is no leakage?**
Folds grouped by trip; scaler and calibrator fitted inside the fold; negative thinning
only on training rows; synthetic reference rows excluded from scoring; the split frozen
to disk; student-test scored only for the final row. And the assertion in §6.4 catches
the specific leak that this pipeline was vulnerable to.

**What is the bottleneck, and what would you do with another day?**
Candidate generation. Better candidates (larger buffer, network-constrained search
around the previous observation) would lift the ceiling more than any model. After that:
Optuna on the two leading families, trip-level aggregate features, and sequence decoding,
which is C's part.

**What did you not finish?**
The Optuna harness is written and persisted to SQLite but the study has not been run —
`optuna` is not installed in the venv. Tuned-vs-untuned is therefore an open row in the
report, not a claimed result. Say that plainly rather than implying tuning happened.

---

## 9. Files

| Path | What it is |
|---|---|
| `src/pathsnap/model_zoo.py` | Five families + calibration wrapper behind one interface |
| `src/pathsnap/evaluation.py` | Frozen split, lattice filter, subsampling, metrics, run log |
| `scripts/train_models.py` | One command per family; identical folds; champion export |
| `scripts/tune_optuna.py` | Bayesian search, log-loss objective, resumable study |
| `scripts/compare_models.py` | Comparison table for the slide and report |
| `results/model_runs.csv` | Every run with its parameters and metrics |
| `results/model_comparison.md` | The table in §5 |
| `artifacts/model_bundle.joblib` | Champion consumed by C's API |

## 10. Reproduce

```
python scripts/build_features.py --splits train student_test     # ~200 s
python scripts/train_models.py --family all --neg-frac 0.5 --calibrate   # ~4 min
python scripts/compare_models.py
```
