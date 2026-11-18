"""Evaluation metrics shared by experiments and the API (contract §3/§5).

Arc-level macro F1 averages only classes present in mapped reference labels.
Candidate-level precision, recall, and F1 treat each candidate row as a binary
reference-arc decision at a score threshold of 0.5.
"""
from __future__ import annotations

import numpy as np


def _class_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-class F1 and support for the supplied labels."""
    true_class_idx = np.searchsorted(labels, y_true)
    support = np.bincount(true_class_idx, minlength=len(labels)).astype(float)

    true_positive = np.bincount(
        true_class_idx[y_true == y_pred], minlength=len(labels)
    ).astype(float)
    predicted_class_idx = np.searchsorted(labels, y_pred)
    within_bounds = predicted_class_idx < len(labels)
    predicted_is_reference_class = np.zeros(len(y_pred), dtype=bool)
    predicted_is_reference_class[within_bounds] = (
        labels[predicted_class_idx[within_bounds]] == y_pred[within_bounds]
    )
    predicted_count = np.bincount(
        predicted_class_idx[predicted_is_reference_class], minlength=len(labels)
    ).astype(float)
    denom = support + predicted_count
    f1 = np.divide(2 * true_positive, denom, out=np.zeros_like(support), where=denom > 0)
    return f1, support


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Score chosen arc ids, averaging macro F1 over reference-present arcs.

    Predictions for arcs absent from the reference are still false positives for
    any reference-present class they displace, but do not create extra classes in
    the macro average. This follows the definition frozen in contracts.md.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1 or y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must be one-dimensional arrays of equal length")

    mask = y_true >= 0
    if mask.sum() == 0:
        return {"n": 0, "top1_accuracy": float("nan"), "macro_f1": float("nan")}
    yt, yp = y_true[mask], y_pred[mask]
    reference_classes = np.unique(yt)
    class_f1, support = _class_f1(yt, yp, reference_classes)
    return {
        "n": int(mask.sum()),
        "top1_accuracy": float(np.mean(yt == yp)),
        "macro_f1": float(np.mean(class_f1)),
        "weighted_f1": float(np.average(class_f1, weights=support)),
    }


def candidate_prf(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict:
    """Candidate-level precision/recall/F1: is the candidate scored above threshold?"""
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if labels.ndim != 1 or scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("labels and scores must be one-dimensional arrays of equal length")
    if labels.size == 0:
        return {"precision": float("nan"), "recall": float("nan"), "candidate_f1": float("nan")}
    try:
        scores = scores.astype(float)
        numeric_labels = labels.astype(float)
        threshold = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("labels, scores, and threshold must be numeric") from exc
    if not np.isfinite(scores).all() or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("scores and threshold must be finite probabilities in [0, 1]")
    if not np.isin(numeric_labels, (0.0, 1.0)).all():
        raise ValueError("candidate labels must contain only 0/1 values")
    y = numeric_labels.astype(bool)
    p = scores >= threshold
    tp = int(np.count_nonzero(y & p))
    fp = int(np.count_nonzero(~y & p))
    fn = int(np.count_nonzero(y & ~p))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "candidate_f1": float(f1),
    }


def candidate_log_loss(labels: np.ndarray, scores: np.ndarray) -> float:
    """Binary log loss over candidate rows (label = is-reference-arc)."""
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if labels.ndim != 1 or scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("labels and scores must be one-dimensional arrays of equal length")
    if labels.size == 0:
        return float("nan")
    try:
        p = scores.astype(float)
        y = labels.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("labels and scores must be numeric") from exc
    if not np.isfinite(p).all() or not np.isfinite(y).all() or not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("candidate labels and scores must be finite, with labels in {0, 1}")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("candidate scores must be probabilities in [0, 1]")
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def route_continuity(pred_arcs: np.ndarray, arcs: np.ndarray) -> float:
    """Fraction of consecutive predictions that are the same arc or connected by an arc."""
    if pred_arcs is None or len(pred_arcs) < 2:
        return float("nan")
    pred_arcs = np.asarray(pred_arcs, dtype=np.int64)
    arcs = np.asarray(arcs)
    if pred_arcs.ndim != 1 or arcs.ndim != 2 or arcs.shape[1] < 2:
        raise ValueError("pred_arcs must be 1-D and arcs must have at least two columns")
    valid = (pred_arcs >= 0) & (pred_arcs < len(arcs))
    a, b = pred_arcs[:-1], pred_arcs[1:]
    pair_valid = valid[:-1] & valid[1:]
    connected = np.zeros(len(a), dtype=bool)
    connected[pair_valid] = (a[pair_valid] == b[pair_valid]) | (
        arcs[a[pair_valid], 1] == arcs[b[pair_valid], 0]
    )
    return float(np.mean(connected))


def candidate_miss_rate(ids: np.ndarray, label_arc: np.ndarray) -> float:
    ids = np.asarray(ids)
    label_arc = np.asarray(label_arc)
    if ids.ndim != 2 or label_arc.ndim != 1 or ids.shape[0] != label_arc.shape[0]:
        raise ValueError("ids must be (n_observations, n_candidates); label_arc must be (n_observations,)")
    n = int((label_arc >= 0).sum())
    if n == 0:
        return float("nan")
    hit = (ids == label_arc[:, None]).any(axis=1) & (label_arc >= 0)
    return float(1.0 - hit.sum() / n)
