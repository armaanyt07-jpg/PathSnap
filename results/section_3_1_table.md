| # | Model | Algorithm family | Key configuration | CV macro F1 | Top-1 acc. (holdout) | Route continuity | Student-test macro F1 | Log loss (holdout) | Inference |
|---|-------|------------------|-------------------|-------------|----------------------|------------------|----------------------|--------------------|-----------|
| 1 | Hist GBDT (champion) | Gradient-boosted trees | `lr=0.1, iters=250, leaves=31, min_leaf=50, l2=1.0, weight=balanced` | 0.702 | 0.749 | 0.956 | 0.885 | 0.091 | 1.17 ms / 1k rows |
| 2 | MLP | Feed-forward neural net | `hidden=64-32, alpha=0.001, lr0=0.001, batch=4096, iters=20` | 0.674 | 0.743 | 0.954 | 0.877 | 0.100 | 0.08 ms / 1k rows |
| 3 | Extra Trees | Extremely randomised trees | `trees=200, depth=32, min_leaf=25, feat=sqrt, weight=balanced_subsample` | 0.668 | 0.740 | 0.951 | 0.875 | 0.093 | 2.37 ms / 1k rows |
| 4 | Nystroem SGD | Nystroem RBF + SGD | `components=64, gamma=0.1, alpha=0.0001, avg=True` | 0.661 | 0.731 | 0.952 | 0.872 | 0.118 | 0.22 ms / 1k rows |
| 5 | Logistic | L2 logistic regression | `C=1.0, weight=balanced` | 0.642 | 0.728 | 0.953 | 0.863 | 0.131 | 0.04 ms / 1k rows |
| 6 | SGD Huber | Linear SGD (Huber) | `alpha=0.0001, weight=balanced` | 0.644 | 0.716 | 0.953 | 0.830 | 0.137 | 0.03 ms / 1k rows |
