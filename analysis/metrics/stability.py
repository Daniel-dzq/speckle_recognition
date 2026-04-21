"""Aggregation, bootstrap CIs, and temporal stability helpers."""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence

import numpy as np

from .basic import normalized_cross_correlation


def aggregate_mean_std(values: Sequence[float]) -> Dict[str, float]:
    """Mean / std / min / max / count for a 1-D numeric sequence."""
    v = np.asarray([x for x in values if x is not None and not np.isnan(float(x))], dtype=np.float64)
    if v.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"),
                "max": float("nan"), "count": 0}
    return {
        "mean": float(np.mean(v)),
        "std": float(np.std(v, ddof=0)),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "count": int(v.size),
    }


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 1000,
    confidence: float = 0.95,
    rng: np.random.Generator | None = None,
) -> Dict[str, float]:
    """Bootstrap percentile CI for an arbitrary scalar statistic."""
    v = np.asarray(list(values), dtype=np.float64)
    if v.size < 2:
        return {"point": float(statistic(v)) if v.size else float("nan"),
                "low": float("nan"), "high": float("nan"), "n_boot": 0}
    rng = rng if rng is not None else np.random.default_rng(0)
    draws = rng.integers(0, v.size, size=(n_boot, v.size))
    samples = np.fromiter(
        (statistic(v[d]) for d in draws),
        dtype=np.float64,
        count=n_boot,
    )
    alpha = (1.0 - confidence) / 2.0
    lo = float(np.quantile(samples, alpha))
    hi = float(np.quantile(samples, 1.0 - alpha))
    return {"point": float(statistic(v)), "low": lo, "high": hi, "n_boot": int(n_boot)}


def temporal_stability_score(series: Sequence[np.ndarray]) -> Dict[str, float]:
    """
    Given a list of feature vectors captured at increasing time indices,
    return the mean consecutive NCC and the mean NCC against the first sample.
    """
    arr = [np.asarray(x).ravel() for x in series]
    if len(arr) < 2:
        return {"consecutive_ncc": float("nan"), "vs_first_ncc": float("nan"),
                "min_consecutive_ncc": float("nan")}
    consec: List[float] = []
    vs_first: List[float] = []
    for i in range(1, len(arr)):
        consec.append(normalized_cross_correlation(arr[i - 1], arr[i]))
        vs_first.append(normalized_cross_correlation(arr[0], arr[i]))
    return {
        "consecutive_ncc": float(np.mean(consec)),
        "vs_first_ncc": float(np.mean(vs_first)),
        "min_consecutive_ncc": float(np.min(consec)),
    }


__all__ = [
    "aggregate_mean_std",
    "bootstrap_ci",
    "temporal_stability_score",
]
