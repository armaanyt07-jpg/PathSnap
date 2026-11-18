# PathSnap — interface contracts

Freeze these before parallel work starts. If a schema changes, change it here first and tell the team.

## 1. Feature table (owner: A)

One row per **(observation, candidate arc)** pair. Produced by `scripts/build_features.py`,
consumed by every model and by the API. Never parse raw `.track/.nodes/.arcs` outside `pathsnap/io.py`.

| column | type | meaning |
|---|---|---|
| `trip_id` | str | e.g. `00000056` |
| `obs_idx` | int | 0-based row index in `.track` |
| `lon`, `lat` | float | observation coordinates |
| `t` | float | timestamp (s) |
| `arc_id` | int | candidate road segment (row index in `.arcs`) |
| `cand_rank` | int | 0 = nearest candidate for that observation |
| `perp_dist_m` | float | perpendicular distance observation → segment, metres |
| `along_m` | float | position of the projection along the segment, metres |
| `rel_pos` | float | `along_m / arc_len_m`, in [0, 1] |
| `arc_len_m` | float | segment length, metres |
| `dist_ratio` | float | `perp_dist_m / arc_len_m` |
| `heading_diff_deg` | float | abs difference between GPS heading and arc direction, 0–180 |
| `gap_m` | float | distance from the previous observation (0 for the first) |
| `gap_s` | float | time from the previous observation |
| `speed_mps` | float | `gap_m / gap_s` |
| `label` | bool | True if `arc_id` is the reference arc for that observation |
| `label_arc` | int | reference arc for the observation (all rows of an obs share it) |
| `route_pos` | int | first index of `label_arc` in `.route`; `-1` if absent |

Rules:
- Observations whose reference arc could not be mapped get `label_arc = -1`; they are **excluded from training**. Report how many.
- Training feature builds may append a missed reference arc with `cand_rank = k` so each mapped training observation has a positive. Always compute candidate miss rate before appending.
- Prediction candidate generation must be route-blind: call `build_trip_features(..., include_labels=False, append_missing_labels=False)`. Reference labels may be loaded separately after predictions for aggregate scoring; they must not change the candidate set or scores.
- Coordinates are projected with a local tangent plane (`geo.LocalProjection`), metres. Document this in §2.2.

## 2. Model bundle (owner: B)

`artifacts/model_bundle.joblib` + `artifacts/model_config.json`.

```python
predict_scores(df: pd.DataFrame) -> np.ndarray   # one score per row, aligned with df
```

Scoring is row-wise so C can apply any decoder on top. `scripts/train_baseline.py` shows the reference implementation.

## 3. Prediction output (owner: C)

The `/predict` response (see §4) must contain, per trip:

```json
{
  "trip_id": "00000056",
  "decoder": "argmax-baseline | hmm-viterbi",
  "n_observations": 1234,
  "observations": [[lon, lat], ...],
  "predicted_route": [arc_id, ...],
  "predicted_line": [[lon, lat], ...],
  "metrics": {"top1_accuracy": 0.0, "macro_f1": 0.0, "candidate_miss_rate": 0.0, "route_continuity": 0.0},
  "timings": {"features_ms": 0, "inference_ms": 0, "total_ms": 0}
}
```

**Ground truth never leaves the server.** `predicted_route` is arc ids; `predicted_line` is
display geometry. Metrics are aggregate only — that is allowed by the brief and is what the UI shows.
The offline `scripts/make_map.py` may draw the reference route for report figures. Map candidate generation remains route-blind.

## 4. HTTP contract (owner: C, consumer: D)

| endpoint | request | response |
|---|---|---|
| `GET /health` | — | `{"status":"ok","model_loaded":true}` |
| `GET /trips` | — | `{"trips":["00000056", ...]}` (the 10 student-test trips) |
| `POST /predict` | `{"trip_id":"00000056"}` | object from §3 |

CORS enabled for the UI origin; API listens on `0.0.0.0` so Laptop 2 can reach it.

## 5. Decisions to justify in the report

- **Macro F1 with thousands of arc-id classes** is near-zero and unstable. We report
  `macro_f1` over classes present in the reference labels **and** candidate-level binary F1,
  and state which one drives model selection.
- **Direction:** arcs are directed. A reversed traversal shows `heading_diff_deg ≈ 180`; if the
  road network is bidirectional this feature must not be read as "wrong side".
- **Duplicated arc ids** in `.route` are normal; `route_pos` uses the first occurrence.
