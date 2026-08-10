"""
wtwin.common.math_utils
==============================
Robust statistical primitives for W-Twin monitor.
"""

import numpy as np
from typing import Sequence


def mad_scale(x: Sequence[float], window: int | None = None) -> float:
    """
    Median Absolute Deviation (MAD), scaled to be a consistent estimator
    of sigma for a normal distribution (scale factor 1.4826).

    Parameters
    ----------
    x       : sequence of floats
    window  : if given, use only the last `window` elements

    Returns
    -------
    Robust estimate of sigma (float). Returns 1e-8 if degenerate.
    """
    arr = np.asarray(x, dtype=float)
    if window is not None:
        arr = arr[-window:]
    if len(arr) < 2:
        return 1e-8
    mad = np.median(np.abs(arr - np.median(arr)))
    sigma = 1.4826 * mad
    return float(sigma) if sigma > 1e-10 else 1e-8


def rolling_stats(
    x: Sequence[float], window: int
) -> tuple[float, float]:
    """
    Rolling mean and std over the last `window` elements.

    Returns
    -------
    (mean, std) tuple of floats
    """
    arr = np.asarray(x[-window:], dtype=float)
    return float(np.mean(arr)), float(np.std(arr, ddof=1)) if len(arr) > 1 else (float(np.mean(arr)), 1e-8)


def fit_power_law(
    steps: Sequence[float], losses: Sequence[float]
) -> tuple[float, float, float]:
    """
    Fit L(t) = a * t^(-b) + c via linearisation in log space.

    Parameters
    ----------
    steps  : training step indices (1-indexed, warmup excluded)
    losses : corresponding observed loss values

    Returns
    -------
    (a, b, c) — power-law coefficients
    Raises ValueError if fit fails, input is too short, or fit is degenerate.

    Notes
    -----
    Requires b > 0 (loss must be decreasing). If the fit produces b ≤ 0,
    falls back to c=mean(losses), a=0, b=1 (flat baseline). This prevents
    degenerate fits when the calibration window is nearly flat (late training)
    or when min(L) is very close to mean(L) making log-space ill-conditioned.
    """
    steps = np.asarray(steps, dtype=float)
    losses = np.asarray(losses, dtype=float)

    if len(steps) < 5:
        raise ValueError("Need at least 5 data points for power-law fit.")
    if np.any(losses <= 0) or np.any(steps <= 0):
        raise ValueError("Steps and losses must be positive for log transform.")

    # Guard: if loss range is too small, the log-space fit is ill-conditioned.
    # Use a flat baseline (c = mean, a = 0) instead.
    loss_range = losses.max() - losses.min()
    loss_mean = losses.mean()
    if loss_range < 0.05 * loss_mean:
        # Nearly flat window — return flat baseline
        return 0.0, 1.0, float(loss_mean)

    # Estimate c as slightly below min loss to ensure (L - c) > 0
    # Use a more conservative estimate to avoid log-space instability
    c_est = max(0.0, losses.min() * 0.80)
    y = np.log(np.maximum(losses - c_est, 1e-6))
    x = np.log(steps)

    # Linear regression in log space: log(L - c) = log(a) - b*log(t)
    coeffs = np.polyfit(x, y, 1)
    b = -coeffs[0]
    a = np.exp(coeffs[1])
    c = c_est

    # Validate: b must be positive (loss must decrease over time)
    # If b ≤ 0, the fit is degenerate — fall back to flat baseline
    if b <= 0.05:
        return 0.0, 1.0, float(losses.mean())

    return float(a), float(b), float(c)


def power_law_predict(
    t: float | np.ndarray, a: float, b: float, c: float
) -> float | np.ndarray:
    """
    Evaluate L_pred(t) = a * t^(-b) + c.
    """
    return a * np.power(np.maximum(t, 1e-8), -b) + c
