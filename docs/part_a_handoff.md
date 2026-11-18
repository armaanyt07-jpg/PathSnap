# Part A handoff

## Current implementation

- `src/pathsnap/features.py` now accepts `include_labels` and `append_missing_labels` options. Defaults preserve labeled training feature generation.
- `src/pathsnap/pipeline.py` forwards those options.
- `/predict` requests route-blind candidates and does not append a reference arc when it is absent. Reference labels can still be used for aggregate server-side metrics after prediction.
- Label assignment remains nearest reference arc with first route occurrence. Route progression is diagnostic, not enforced.

## Interfaces and use

The model feature columns remain unchanged. Rebuild artifacts with `python scripts/build_features.py --splits train student_test`: training includes mapped labels and appends missed reference rows; student-test features retain labels for offline metrics but never append them to candidates. B's model consumes only `FEATURE_COLUMNS`, so labels do not enter model inputs. The API predicts from route-blind candidates, then loads route labels separately for aggregate metrics.

Run `python scripts/explore_data.py` for partition summaries and `python scripts/make_map.py --split student_test --trip <trip_id>` for an interactive Folium map. Maps show observations, prefiltered network, reference route, and a baseline predicted route when a saved model is available. EDA files are `results/eda_trips.csv` and `results/eda_summary.json`.

The Part A report-material scaffold is `docs/part_a_report_material.md`. Populate claims from generated result files and review them as a team; do not invent results or submit AI-generated analysis in place of the team's own interpretation. Evaluator-testing data must remain untouched.
