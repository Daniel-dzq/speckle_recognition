"""Radial intensity profile extraction + Gaussian fitting."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np


def radial_intensity_profile(
    image: np.ndarray,
    center: Optional[Tuple[float, float]] = None,
    bins: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return ``(r, I)`` — the radial mean intensity profile.

    ``r`` is expressed as normalised radius (0..1) relative to the maximum
    in-image distance from ``center``. ``center`` defaults to the centroid.
    """
    img = np.asarray(image, dtype=np.float32)
    if img.ndim != 2:
        raise ValueError("radial_intensity_profile expects a 2-D image")
    h, w = img.shape
    ys, xs = np.indices((h, w), dtype=np.float32)
    if center is None:
        total = img.sum()
        if total <= 0:
            cx, cy = w / 2.0, h / 2.0
        else:
            cx = float((xs * img).sum() / total)
            cy = float((ys * img).sum() / total)
    else:
        cx, cy = float(center[0]), float(center[1])
    X = xs - cx
    Y = ys - cy
    R = np.sqrt(X * X + Y * Y)
    R_max = float(R.max()) or 1.0
    R_norm = R / R_max

    edges = np.linspace(0.0, 1.0, bins + 1, dtype=np.float32)
    idx = np.clip(np.digitize(R_norm.ravel(), edges) - 1, 0, bins - 1)
    sum_per_bin = np.bincount(idx, weights=img.ravel(), minlength=bins)
    count_per_bin = np.bincount(idx, minlength=bins)
    with np.errstate(divide="ignore", invalid="ignore"):
        profile = np.where(count_per_bin > 0, sum_per_bin / count_per_bin, 0.0)
    r_centers = 0.5 * (edges[:-1] + edges[1:])
    return r_centers, profile.astype(np.float32)


def fit_gaussian_profile(r: np.ndarray, I: np.ndarray) -> Dict[str, float]:
    """Fit ``I(r) = A * exp(-((r - r0) / sigma)^2) + c``. Falls back to SciPy."""
    r = np.asarray(r, dtype=np.float64)
    I = np.asarray(I, dtype=np.float64)
    A0 = float(np.max(I) - np.min(I))
    c0 = float(np.min(I))
    if I.sum() > 0:
        r0 = float(np.sum(r * I) / np.sum(I))
    else:
        r0 = 0.0
    sigma0 = 0.25
    result = {
        "A": A0,
        "r0": r0,
        "sigma": sigma0,
        "c": c0,
        "rmse": float("nan"),
        "success": False,
    }
    try:
        from scipy.optimize import curve_fit  # type: ignore
    except ImportError:  # pragma: no cover
        return result

    def model(rr, A, r0, sigma, c):
        return A * np.exp(-((rr - r0) / max(sigma, 1e-6)) ** 2) + c

    try:
        popt, _ = curve_fit(
            model, r, I,
            p0=[A0, r0, sigma0, c0],
            maxfev=2000,
        )
    except Exception:
        return result
    A, r0_fit, sigma_fit, c_fit = [float(v) for v in popt]
    pred = model(r, *popt)
    rmse = float(np.sqrt(np.mean((pred - I) ** 2)))
    return {
        "A": A,
        "r0": r0_fit,
        "sigma": abs(sigma_fit),
        "c": c_fit,
        "rmse": rmse,
        "success": True,
    }


def profile_width(r: np.ndarray, I: np.ndarray, level: float = 0.5) -> float:
    """Width of the profile at a fractional level (default FWHM)."""
    r = np.asarray(r, dtype=np.float64)
    I = np.asarray(I, dtype=np.float64)
    if I.size < 2:
        return float("nan")
    Imin, Imax = float(I.min()), float(I.max())
    if Imax - Imin < 1e-12:
        return float("nan")
    peak = int(np.argmax(I))

    def interp_cross(start: int, direction: int) -> float:
        threshold = Imin + level * (Imax - Imin)
        i = start
        while 0 < i + direction < I.size and 0 < i + direction >= 0:
            if direction > 0 and i + direction >= I.size:
                return float("nan")
            if direction < 0 and i + direction < 0:
                return float("nan")
            y0, y1 = I[i], I[i + direction]
            if (y0 - threshold) * (y1 - threshold) <= 0:
                denom = (y1 - y0) if (y1 - y0) != 0 else 1e-12
                t = (threshold - y0) / denom
                return float(r[i] + t * (r[i + direction] - r[i]))
            i += direction
        return float("nan")

    left = interp_cross(peak, -1)
    right = interp_cross(peak, 1)
    if np.isnan(left) or np.isnan(right):
        return float("nan")
    return float(abs(right - left))


__all__ = [
    "radial_intensity_profile",
    "fit_gaussian_profile",
    "profile_width",
]
