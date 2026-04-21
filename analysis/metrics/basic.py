"""Core similarity / distance / information metrics (vectorised NumPy)."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _flat(X: np.ndarray) -> np.ndarray:
    """Ensure 2-D ``(N, D)`` layout."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        return X[None, :]
    if X.ndim == 2:
        return X
    return X.reshape(X.shape[0], -1)


# ---------------------------------------------------------------------------
# Distance metrics
# ---------------------------------------------------------------------------


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    return float(np.linalg.norm(a - b))


def pairwise_euclidean(X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
    """Vectorised pairwise L2 distance matrix."""
    X = _flat(X)
    if Y is None:
        Y = X
    else:
        Y = _flat(Y)
    x2 = np.sum(X * X, axis=1, keepdims=True)
    y2 = np.sum(Y * Y, axis=1, keepdims=True).T
    d2 = x2 + y2 - 2.0 * X @ Y.T
    np.maximum(d2, 0, out=d2)
    return np.sqrt(d2, dtype=np.float32)


def correlation_coefficient(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    if a.size < 2:
        return float("nan")
    c = np.corrcoef(a, b)
    return float(c[0, 1])


def normalized_cross_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Zero-mean NCC in the range [-1, 1]."""
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    am = a - a.mean()
    bm = b - b.mean()
    denom = np.linalg.norm(am) * np.linalg.norm(bm)
    if denom < 1e-12:
        return 0.0
    return float(am @ bm / denom)


def pairwise_ncc(X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
    """Pairwise zero-mean NCC matrix."""
    X = _flat(X)
    Y = X if Y is None else _flat(Y)

    Xc = X - X.mean(axis=1, keepdims=True)
    Yc = Y - Y.mean(axis=1, keepdims=True)
    xn = np.linalg.norm(Xc, axis=1, keepdims=True)
    yn = np.linalg.norm(Yc, axis=1, keepdims=True).T
    num = Xc @ Yc.T
    denom = xn @ yn
    denom = np.where(denom < 1e-12, 1.0, denom)
    out = num / denom
    return out.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Dispersion / information
# ---------------------------------------------------------------------------


def coefficient_of_variation(values: Sequence[float]) -> float:
    x = np.asarray(list(values), dtype=np.float32)
    if x.size == 0:
        return float("nan")
    m = float(np.mean(x))
    if abs(m) < 1e-12:
        return float("inf")
    return float(np.std(x, ddof=0) / abs(m))


def shannon_entropy(frame: np.ndarray, bins: int = 256, normalize: bool = True) -> float:
    """Histogram-based pixel entropy in bits."""
    f = np.asarray(frame, dtype=np.float32).ravel()
    if f.size == 0:
        return 0.0
    if normalize:
        mn, mx = float(f.min()), float(f.max())
        rng = mx - mn
        if rng < 1e-12:
            return 0.0
        f = (f - mn) / rng
    hist, _ = np.histogram(f, bins=bins, range=(0.0, 1.0) if normalize else (float(f.min()), float(f.max())))
    total = hist.sum()
    if total == 0:
        return 0.0
    p = hist[hist > 0] / total
    return float(-np.sum(p * np.log2(p)))


def transmission_loss_db(p_in: float, p_out: float) -> float:
    """Return -10 * log10(p_out / p_in). Non-positive inputs yield ``inf``."""
    if p_in is None or p_out is None:
        return float("nan")
    if p_in <= 0 or p_out <= 0:
        return float("inf")
    return float(-10.0 * np.log10(float(p_out) / float(p_in)))


__all__ = [
    "euclidean_distance",
    "pairwise_euclidean",
    "correlation_coefficient",
    "normalized_cross_correlation",
    "pairwise_ncc",
    "coefficient_of_variation",
    "shannon_entropy",
    "transmission_loss_db",
]
