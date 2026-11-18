# Part A report material: dataset understanding and maps

This file is a factual scaffold for report sections 2.2 and 6. Populate it only after generating the linked outputs. Review and write the team's interpretation from its own inspection; do not report uncomputed values.

## 2.2 Dataset understanding and exploratory analysis

Run `python scripts/inventory.py` and `python scripts/explore_data.py`. Cite `results/inventory_summary.json`, `results/eda_summary.json`, and `results/eda_trips.csv` for partition counts, file health, observation and network sizes, timestamp gaps/order, coordinate extents, isolated nodes, weakly connected components, and route availability. State the partition used for each table or figure. Describe any anomalous trips using trip IDs from `eda_trips.csv`.

The raw formats are whitespace-separated `.track` (longitude, latitude, seconds), `.nodes` (longitude, latitude; row index is node ID), `.arcs` (directed from/to node IDs; row index is arc ID), and `.route` (chronological reference arc IDs). The feature pipeline projects coordinates into a local tangent plane around the trip median coordinate and computes point-to-segment distances in metres. Features include candidate rank, perpendicular and along-segment position, segment length, heading difference, observation gap distance/time, and speed. Explain which are useful for matching noisy observations to nearby roads, and discuss directionality and limitations of nearest-route labels.

Reference routes are supplied evaluation references, not independently verified physical truth. Labels use the geometrically nearest arc among unique route arcs and `route_pos` records that arc's first occurrence. Backward route-position changes are diagnostic and are not forcibly corrected. Training may append missed reference arcs; student-test and prediction candidates never do. Report candidate miss rates before any training-only append.

## 6. Geographical visualisation

Generate interactive maps with `python scripts/make_map.py --split student_test --trip <trip_id>`. The HTML output under `results/maps/` contains GPS observations, relevant road-network segments, a reference route when present, a predicted baseline route when a model bundle is available, a legend, and layer controls. Select a representative map from the permitted student-test partition for a screenshot/export in the consolidated PDF. Do not use evaluator-testing data or manually alter routes.

## Generated evidence

- `results/inventory.csv` and `results/inventory_summary.json`
- `results/eda_trips.csv` and `results/eda_summary.json`
- `results/feature_build_report.json` for candidate misses and route-label diagnostics
- `results/maps/<trip_id>.html` for interactive Folium maps
