"""Evaluation metrics shared by experiments and the API (contract §3/§5).

Note on macro F1: arc ids are classes with thousands of levels and very few
occurrences each, so a plain macro average is unstable and near-zero. We report
  * macro_f1          - sklearn macro over classes present in y_true
  * binary_f1         - candidate-level: is the chosen arc the reference arc?
and state in the report which one drives selection.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mask = y_true >= 0
    if mask.sum() == 0:
        return {"n": 0, "top1_accuracy": float("nan"), "macro_f1": float("nan")}
    yt, yp = y_true[mask], y_pred[mask]
    return {
        "n": int(mask.sum()),
        "top1_accuracy": float(accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(yt, yp, average="weighted", zero_division=0)),
    }


def candidate_prf(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict:
    """Candidate-level precision/recall/F1: is the candidate scored above threshold?"""
    if labels.size == 0:
        return {"precision": float("nan"), "recall": float("nan"), "candidate_f1": float("nan")}
    y = labels.astype(bool)
    p = scores >= threshold
    return {
        "precision": float(precision_score(y, p, zero_division=0)),
        "recall": float(recall_score(y, p, zero_division=0)),
        "candidate_f1": float(f1_score(y, p, zero_division=0)),
    }


def candidate_log_loss(labels: np.ndarray, scores: np.ndarray) -> float:
    """Binary log loss over candidate rows (label = is-reference-arc)."""
    if labels.size == 0:
        return float("nan")
    p = np.clip(scores.astype(float), 1e-6, 1 - 1e-6)
    y = labels.astype(float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def route_continuity(pred_arcs: np.ndarray, arcs: np.ndarray) -> float:
    """Fraction of consecutive predictions that are the same arc or connected by an arc."""
    if pred_arcs is None or len(pred_arcs) < 2:
        return float("nan")
    a, b = pred_arcs[:-1], pred_arcs[1:]
    connected = (a == b) | (arcs[a, 1] == arcs[b, 0])
    return float(np.mean(connected))


def candidate_miss_rate(ids: np.ndarray, label_arc: np.ndarray) -> float:
    n = int((label_arc >= 0).sum())
    if n == 0:
        return float("nan")
    hit = (ids == label_arc[:, None]).any(axis=1) & (label_arc >= 0)
    return float(1.0 - hit.sum() / n)
