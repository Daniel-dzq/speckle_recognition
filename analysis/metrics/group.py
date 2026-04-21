"""Intra / inter class separability metrics."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .basic import pairwise_euclidean, pairwise_ncc


def _labels_to_array(labels: Sequence[str]) -> np.ndarray:
    arr = np.asarray(list(labels))
    return arr


def _upper_tri(N: int) -> tuple[np.ndarray, np.ndarray]:
    iu = np.triu_indices(N, k=1)
    return iu


def _pair_mask(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean masks ``(same_upper, diff_upper)`` over the upper triangle."""
    N = labels.shape[0]
    iu = _upper_tri(N)
    same = labels[iu[0]] == labels[iu[1]]
    return same, ~same


def intra_class_distance(X: np.ndarray, labels: Sequence[str]) -> float:
    """Mean pairwise L2 distance among samples that share a label."""
    labels = _labels_to_array(labels)
    if labels.size < 2:
        return float("nan")
    D = pairwise_euclidean(X)
    same, _ = _pair_mask(labels)
    iu = _upper_tri(D.shape[0])
    d = D[iu][same]
    return float(np.mean(d)) if d.size else float("nan")


def inter_class_distance(X: np.ndarray, labels: Sequence[str]) -> float:
    """Mean pairwise L2 distance between samples with different labels."""
    labels = _labels_to_array(labels)
    if labels.size < 2:
        return float("nan")
    D = pairwise_euclidean(X)
    _, diff = _pair_mask(labels)
    iu = _upper_tri(D.shape[0])
    d = D[iu][diff]
    return float(np.mean(d)) if d.size else float("nan")


def intra_inter_ratio(X: np.ndarray, labels: Sequence[str]) -> Dict[str, float]:
    """Return intra, inter, and the inter/intra ratio in one vectorised pass."""
    labels = _labels_to_array(labels)
    if labels.size < 2:
        return {"intra": float("nan"), "inter": float("nan"), "ratio": float("nan")}
    D = pairwise_euclidean(X)
    iu = _upper_tri(D.shape[0])
    vals = D[iu]
    same, diff = _pair_mask(labels)
    intra = float(np.mean(vals[same])) if same.any() else float("nan")
    inter = float(np.mean(vals[diff])) if diff.any() else float("nan")
    ratio = inter / intra if intra and intra == intra and intra > 0 else float("nan")
    return {"intra": intra, "inter": inter, "ratio": ratio}


def within_class_similarity(X: np.ndarray, labels: Sequence[str]) -> Dict[str, float]:
    """Mean NCC similarity within each class; returns per-class dict."""
    labels = _labels_to_array(labels)
    S = pairwise_ncc(X)
    unique = np.unique(labels)
    per_class: Dict[str, float] = {}
    for lab in unique:
        idx = np.where(labels == lab)[0]
        if idx.size < 2:
            per_class[str(lab)] = float("nan")
            continue
        sub = S[np.ix_(idx, idx)]
        iu = np.triu_indices(sub.shape[0], k=1)
        per_class[str(lab)] = float(np.mean(sub[iu]))
    return per_class


__all__ = [
    "intra_class_distance",
    "inter_class_distance",
    "intra_inter_ratio",
    "within_class_similarity",
]
