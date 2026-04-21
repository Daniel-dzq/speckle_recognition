"""Authentication-centric metrics: ROC / AUC / EER / confusion / top-k."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .basic import pairwise_euclidean, pairwise_ncc


def roc_curve(
    scores: Sequence[float],
    labels: Sequence[int],
    score_higher_is_genuine: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a ROC curve from a 1-D score + binary label array.

    * ``labels`` is expected to be ``0`` for impostor, ``1`` for genuine.
    * Returns ``(fpr, tpr, thresholds)`` sorted by increasing FPR.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.int64).ravel()
    if s.size != y.size:
        raise ValueError("scores and labels must have matching length")

    order = np.argsort(-s if score_higher_is_genuine else s)
    s = s[order]
    y = y[order]
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return (
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([float("inf"), float("-inf")]),
        )

    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)

    # Deduplicate threshold ties (keep last occurrence).
    distinct = np.r_[np.where(np.diff(s) != 0)[0], s.size - 1]
    tpr = tp[distinct] / n_pos
    fpr = fp[distinct] / n_neg
    thr = s[distinct]

    tpr = np.r_[0.0, tpr, 1.0]
    fpr = np.r_[0.0, fpr, 1.0]
    thr = np.r_[thr[0] + 1.0, thr, thr[-1] - 1.0]
    return fpr, tpr, thr


def auc_score(fpr: np.ndarray, tpr: np.ndarray) -> float:
    fpr = np.asarray(fpr, dtype=np.float64)
    tpr = np.asarray(tpr, dtype=np.float64)
    order = np.argsort(fpr)
    return float(np.trapz(tpr[order], fpr[order]))


def equal_error_rate(
    fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray
) -> Tuple[float, float]:
    """Return ``(eer, threshold_at_eer)`` using linear interpolation."""
    fpr = np.asarray(fpr, dtype=np.float64)
    tpr = np.asarray(tpr, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if fpr.size == 0 or tpr.size == 0:
        return float("nan"), float("nan")
    fnr = 1.0 - tpr
    diff = fpr - fnr
    sign_change = np.where(np.diff(np.sign(diff)) != 0)[0]
    if sign_change.size == 0:
        eer = float(np.min(np.minimum(fpr, fnr)))
        idx = int(np.argmin(np.abs(diff)))
        thr_val = thresholds[idx]
        return eer, float(thr_val) if np.isfinite(thr_val) else float("nan")
    i = int(sign_change[0])
    x0, x1 = diff[i], diff[i + 1]
    denom = x1 - x0 if x1 - x0 != 0 else 1e-12
    t = -x0 / denom
    eer = float(fpr[i] + t * (fpr[i + 1] - fpr[i]))
    thr_i = thresholds[i]
    thr_ip1 = thresholds[i + 1]
    if np.isfinite(thr_i) and np.isfinite(thr_ip1):
        thr_eer = float(thr_i + t * (thr_ip1 - thr_i))
    else:
        thr_eer = float("nan")
    return eer, thr_eer


def confusion_matrix(
    y_true: Sequence[int], y_pred: Sequence[int], n_classes: int
) -> np.ndarray:
    yt = np.asarray(y_true, dtype=np.int64).ravel()
    yp = np.asarray(y_pred, dtype=np.int64).ravel()
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (yt, yp), 1)
    return cm


def top_k_accuracy(scores: np.ndarray, y_true: Sequence[int], k: int = 1) -> float:
    """Scores is ``(n_probes, n_classes)``; higher = better."""
    yt = np.asarray(y_true, dtype=np.int64).ravel()
    scores = np.asarray(scores)
    if scores.size == 0:
        return float("nan")
    topk = np.argpartition(-scores, kth=min(k, scores.shape[1] - 1), axis=1)[:, :k]
    correct = np.any(topk == yt[:, None], axis=1)
    return float(np.mean(correct))


def nearest_neighbor_identify(
    probes: np.ndarray,
    probe_labels: Sequence[str],
    templates: np.ndarray,
    template_labels: Sequence[str],
    metric: str = "ncc",
) -> Dict[str, object]:
    """
    Nearest-neighbour identification.

    Returns a dict with per-probe predictions, accuracy, and the full
    similarity / distance matrix for downstream inspection.
    """
    probes = np.asarray(probes)
    templates = np.asarray(templates)
    if metric == "ncc":
        S = pairwise_ncc(probes, templates)          # higher is better
        nn_idx = np.argmax(S, axis=1)
    elif metric == "euclidean":
        D = pairwise_euclidean(probes, templates)     # lower is better
        S = -D
        nn_idx = np.argmin(D, axis=1)
    else:
        raise ValueError(f"Unknown metric: {metric!r}")

    preds = [template_labels[i] for i in nn_idx]
    correct = [int(a == b) for a, b in zip(preds, probe_labels)]
    acc = float(np.mean(correct)) if correct else float("nan")
    return {
        "predictions": preds,
        "accuracy": acc,
        "nn_indices": nn_idx.tolist(),
        "similarity": S,
    }


__all__ = [
    "roc_curve",
    "auc_score",
    "equal_error_rate",
    "confusion_matrix",
    "top_k_accuracy",
    "nearest_neighbor_identify",
]
