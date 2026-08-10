"""
wtwin.monitor.baseline
==============================
W-Twin — baseline predictor layer.

The baseline predictor is any causal forecasting model F fit on the
first k% of observed steps. This module provides:

  - PowerLawBaseline  (default — used in experiments)
  - BaseBaseline      (abstract interface for custom predictors)

Custom predictors (GP, Kalman, RNN, etc.) can be plugged in by
subclassing BaseBaseline and implementing fit() + predict().
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from wtwin.common.math_utils import fit_power_law, power_law_predict


class BaseBaseline(ABC):
    """Abstract interface for all baseline predictors."""

    @abstractmethod
    def fit(self, steps: np.ndarray, losses: np.ndarray) -> None:
        """Fit predictor on observed (steps, losses)."""

    @abstractmethod
    def predict(self, t: float | np.ndarray) -> float | np.ndarray:
        """Return L_pred(t) for given step(s)."""

    @property
    @abstractmethod
    def fit_mse(self) -> float:
        """MSE of the fit on the calibration window (used for Q)."""

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """True after a successful fit()."""


@dataclass
class PowerLawBaseline(BaseBaseline):
    """
    Default baseline: L_pred(t) = a * t^(-b) + c

    Fit via log-space linear regression on the calibration window.
    Used as the reference implementation in all W-Twin experiments.

    Parameters
    ----------
    calibration_frac : float
        Fraction of total steps used for fitting (default 0.10 = 10%).
        Warmup steps are excluded before this fraction is applied.
    warmup_steps : int
        Number of initial steps to exclude (LR warmup phase).
    """

    calibration_frac: float = 0.10
    warmup_steps: int = 50

    _a: float = field(default=0.0, init=False, repr=False)
    _b: float = field(default=0.0, init=False, repr=False)
    _c: float = field(default=0.0, init=False, repr=False)
    _fit_mse: float = field(default=float("inf"), init=False, repr=False)
    _fitted: bool = field(default=False, init=False, repr=False)

    def fit(self, steps: np.ndarray, losses: np.ndarray) -> None:
        """
        Fit power-law on calibration window.

        Parameters
        ----------
        steps  : full step array seen so far (1-indexed)
        losses : corresponding observed losses
        """
        steps = np.asarray(steps, dtype=float)
        losses = np.asarray(losses, dtype=float)

        # Exclude warmup
        mask = steps > self.warmup_steps
        steps_clean = steps[mask]
        losses_clean = losses[mask]

        if len(steps_clean) < 5:
            raise ValueError(
                f"After warmup exclusion only {len(steps_clean)} steps remain. "
                "Need ≥5 for fit. Increase data or reduce warmup_steps."
            )

        # Use calibration fraction of the cleaned data
        n_cal = max(5, int(len(steps_clean) * self.calibration_frac))
        # Take first n_cal points of clean data (earliest part of stable training)
        cal_steps = steps_clean[:n_cal]
        cal_losses = losses_clean[:n_cal]

        self._a, self._b, self._c = fit_power_law(cal_steps, cal_losses)

        # Compute MSE on the calibration window
        preds = power_law_predict(cal_steps, self._a, self._b, self._c)
        self._fit_mse = float(np.mean((cal_losses - preds) ** 2))
        self._fitted = True

    def predict(self, t: float | np.ndarray) -> float | np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")
        return power_law_predict(t, self._a, self._b, self._c)

    @property
    def fit_mse(self) -> float:
        return self._fit_mse

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def coefficients(self) -> dict[str, float]:
        """Return fitted coefficients as dict."""
        return {"a": self._a, "b": self._b, "c": self._c}
