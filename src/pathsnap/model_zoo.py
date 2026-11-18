"""Model families for workstream B (owner: B).

Every family is an sklearn-compatible estimator whose ``predict_proba(X)`` returns
two columns, so the frozen contract in docs/contracts.md §2 keeps working:

    joblib.load(artifacts/model_bundle.joblib)  ->  model
    pathsnap.baseline.predict_scores(model, df) ->  scores in [0, 1], shape (n,)

Nothing here imports optuna; tuning lives in scripts/tune_optuna.py so the model
zoo stays installable with the base requirements.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS

# Families that exist in the comparison. Keep the keys stable: they are written
# to results/model_runs.csv and used as artifact filenames.
FAMILIES = ("logistic", "extra_trees", "hist_gbdt", "nystroem_sgd", "mlp", "sgd_huber")

# Defaults are deliberately conservative so a family runs without tuning first.
DEFAULT_PARAMS = {
    "logistic": {"C": 1.0, "class_weight": "balanced"},
    "extra_trees": {
        # Leaf size and depth keep the serialised forest in the low hundreds of MB
        # on the full training set; 300 unconstrained trees ran to ~2 GB.
        "n_estimators": 200,
        "max_depth": 32,
        "min_samples_leaf": 25,
        "max_features": "sqrt",
        "class_weight": "balanced_subsample",
    },
    "hist_gbdt": {
        "learning_rate": 0.1,
        "max_iter": 250,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 50,
        "l2_regularization": 1.0,
        "class_weight": "balanced",
    },
    "nystroem_sgd": {
        "n_components": 64,
        "gamma": 0.1,
        "alpha": 1e-4,
        "average": True,
    },
    "mlp": {
        "hidden_layer_sizes": (64, 32),
        "alpha": 1e-3,
        "learning_rate_init": 1e-3,
        "batch_size": 4096,
        "max_iter": 20,
    },
    "sgd_huber": {"alpha": 1e-4, "class_weight": "balanced"},
}


def _logistic(p: dict, seed: int):
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=float(p["C"]),
                    max_iter=400,
                    class_weight=p.get("class_weight", "balanced"),
                ),
            ),
        ]
    )


def _extra_trees(p: dict, seed: int):
    return ExtraTreesClassifier(
        n_estimators=int(p["n_estimators"]),
        max_depth=None if p.get("max_depth") in (None, 0) else int(p["max_depth"]),
        min_samples_leaf=int(p["min_samples_leaf"]),
        max_features=p["max_features"],
        class_weight=p.get("class_weight", "balanced_subsample"),
        n_jobs=-1,
        random_state=seed,
    )


def _hist_gbdt(p: dict, seed: int):
    return HistGradientBoostingClassifier(
        learning_rate=float(p["learning_rate"]),
        max_iter=int(p["max_iter"]),
        max_leaf_nodes=int(p["max_leaf_nodes"]),
        min_samples_leaf=int(p["min_samples_leaf"]),
        l2_regularization=float(p["l2_regularization"]),
        class_weight=p.get("class_weight", "balanced"),
        early_stopping=False,  # random early-stopping splits can leak across trips
        random_state=seed,
    )


def _nystroem_sgd(p: dict, seed: int):
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "kernel",
                Nystroem(
                    gamma=float(p["gamma"]),
                    n_components=int(p["n_components"]),
                    random_state=seed,
                ),
            ),
            (
                "clf",
                SGDClassifier(
                    loss="log_loss",
                    alpha=float(p["alpha"]),
                    max_iter=20,
                    tol=None,
                    average=bool(p["average"]),
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )


def _mlp(p: dict, seed: int):
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                MLPClassifier(
                    hidden_layer_sizes=tuple(p["hidden_layer_sizes"]),
                    alpha=float(p["alpha"]),
                    learning_rate_init=float(p["learning_rate_init"]),
                    batch_size=int(p["batch_size"]),
                    max_iter=int(p["max_iter"]),
                    early_stopping=False,
                    random_state=seed,
                ),
            ),
        ]
    )


def _sgd_huber(p: dict, seed: int):
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                SGDClassifier(
                    loss="modified_huber",
                    alpha=float(p["alpha"]),
                    max_iter=20,
                    tol=None,
                    class_weight=p.get("class_weight", "balanced"),
                    random_state=seed,
                ),
            ),
        ]
    )


_BUILDERS = {
    "logistic": _logistic,
    "extra_trees": _extra_trees,
    "hist_gbdt": _hist_gbdt,
    "nystroem_sgd": _nystroem_sgd,
    "mlp": _mlp,
    "sgd_huber": _sgd_huber,
}


def build(family: str, params: dict | None = None, seed: int = 42):
    """Instantiate an unfitted estimator for *family*."""
    if family not in _BUILDERS:
        raise KeyError(f"Unknown family '{family}'. Choose from {', '.join(FAMILIES)}.")
    merged = dict(DEFAULT_PARAMS[family])
    merged.update(params or {})
    return _BUILDERS[family](merged, seed)


def fit(family: str, df: pd.DataFrame, params: dict | None = None, seed: int = 42):
    """Fit *family* on a feature frame; returns the fitted estimator."""
    model = build(family, params, seed)
    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int8)

    if family == "nystroem_sgd":
        # Nystroem needs fewer components than samples; shrink on small frames.
        k = min(int(model.named_steps["kernel"].n_components), max(8, len(X) // 4))
        model.set_params(kernel__n_components=k)

    with warnings.catch_warnings():
        # Same cosmetic NumPy 2 matmul warning baseline.fit suppresses.
        warnings.filterwarnings("ignore", message=".*in matmul", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message=".*mall number of iterations", category=UserWarning)
        model.fit(X, y)
    return model


def fit_calibrator(scores: np.ndarray, labels: np.ndarray) -> IsotonicRegression:
    """Monotone score -> probability mapping, fitted on out-of-fold scores only.

    Isotonic regression is non-decreasing, so per-observation argmax is unchanged;
    calibration buys log loss (and therefore better Viterbi emissions for C), not
    Top-1.
    """
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
    iso.fit(np.asarray(scores, dtype=float), np.asarray(labels, dtype=float))
    return iso


class CalibratedScorer:
    """Estimator wrapper that keeps the ``predict_proba`` contract after calibration.

    Isotonic regression is monotone but piecewise constant, so two candidates with
    different raw scores can land on the same calibrated value and change the
    argmax on an artificial tie. The tiny ``1e-9 * p`` term restores the original
    ordering among collapsed scores without materially changing the probability,
    so calibration buys log loss and leaves Top-1 untouched — which is exactly the
    claim the report makes.
    """

    def __init__(self, base, calibrator: IsotonicRegression):
        self.base_ = base
        self.calibrator_ = calibrator

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = np.asarray(self.base_.predict_proba(X))[:, 1]
        q = self.calibrator_.predict(p) + 1e-9 * p
        q = np.clip(q, 1e-6, 1 - 1e-6)
        return np.column_stack([1.0 - q, q])

    @property
    def classes_(self):
        return getattr(self.base_, "classes_", np.array([0, 1]))
