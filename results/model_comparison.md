# Model comparison (workstream B)

Ranked by held-out Top-1 over labelled observations. `holdout_top1_covered`
removes observations whose reference arc is missing from the candidate set,
so the gap between the two columns is the candidate miss rate, not model error.

| family | calibrated | cv_top1_mean | cv_top1_std | holdout_top1 | holdout_top1_covered | holdout_macro_f1 | holdout_log_loss | test_top1 | test_log_loss | fit_seconds | inference_ms_per_1k |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hist_gbdt | yes | 0.7355 | 0.1711 | 0.7485 | 0.8622 | 0.8170 | 0.1658 | 0.8375 | 0.0980 | 4.5000 | 1.1700 |
| mlp | yes | 0.7312 | 0.1702 | 0.7426 | 0.8554 | 0.7958 | 0.1698 | 0.8263 | 0.1066 | 3.2000 | 0.0800 |
| extra_trees | yes | 0.7303 | 0.1708 | 0.7399 | 0.8522 | 0.7882 | 0.2107 | 0.8266 | 0.1017 | 20.7000 | 2.3700 |
| nystroem_sgd | yes | 0.7212 | 0.1664 | 0.7313 | 0.8424 | 0.7689 | 0.2423 | 0.8062 | 0.1224 | 2.0000 | 0.2200 |
| logistic | yes | 0.7090 | 0.1621 | 0.7276 | 0.8381 | 0.7625 | 0.2854 | 0.8110 | 0.1328 | 0.4000 | 0.0400 |
| sgd_huber | yes | 0.7121 | 0.1638 | 0.7163 | 0.8251 | 0.7280 | 0.3039 | 0.7956 | 0.1374 | 0.3000 | 0.0300 |

Runs compared: 6 families. Source: results/model_runs.csv.
