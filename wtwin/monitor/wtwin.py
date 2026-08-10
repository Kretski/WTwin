"""
wtwin.monitor.wtwin
===========================
W-Twin — online training trajectory deviation monitor.

Algorithm (v2):
    W(t) = Q(t) · (D(t) − T(t))

Where:
    D(t) = (L_obs(t) − L_pred(t)) / MAD_local(t)     [normalised deviation]
    Q(t) = exp(−MSE_fit / τ)                           [baseline confidence]
    T(t) = μ*_D + α·σ*_D                               [adaptive threshold,
                                                         frozen during alert]

Alert condition:
    W(t) > 0  for  n_consec  consecutive steps

Reference:
    Kretski, D. (2026). W-Twin: Forecast-Based Detection of Progressive Neural Network Training Degradation.
    https://github.com/Kretski/WTwin
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import NamedTuple

from wtwin.monitor.baseline import BaseBaseline, PowerLawBaseline
from wtwin.common.math_utils import mad_scale


class WTwinState(NamedTuple):
    """Snapshot of W-Twin at a single step."""
    step: int
    l_obs: float
    l_pred: float
    D: float        # normalised deviation
    Q: float        # baseline confidence
    T: float        # adaptive threshold
    W: float        # final score
    alert: bool     # True if currently in alert state


@dataclass
class WTwinMonitor:
    """
    Online W-Twin monitor.

    Parameters
    ----------
    baseline : BaseBaseline
        Predictor for L_pred(t). Defaults to PowerLawBaseline.
    mad_window : int
        Window for local MAD (robust sigma estimate). Default 50.
    threshold_window : int
        Rolling window for adaptive threshold T(t). Default 200.
    alpha : float
        Threshold sensitivity (T = μ + α·σ). Default 2.0.
    tau : float
        Q decay scale (Q = exp(−MSE/τ)). Default 1e-3.
    n_consec : int
        Consecutive steps above zero W required for alert. Default 5.
    calibration_frac : float
        Fraction of steps used to fit baseline (passed to baseline). Default 0.10.
    warmup_steps : int
        Steps excluded from fit (LR warmup). Default 50.

    Usage
    -----
    monitor = WTwinMonitor()
    for step, loss in training_loop():
        state = monitor.update(step, loss)
        if state.alert:
            print(f"ALERT at step {step}: W={state.W:.3f}")
    """

    baseline: BaseBaseline = field(default_factory=PowerLawBaseline)
    mad_window: int = 50
    threshold_window: int = 200
    alpha: float = 2.0
    tau: float = 1e-3
    n_consec: int = 5
    calibration_frac: float = 0.10
    warmup_steps: int = 50
    use_adaptive_T: bool = False  # False = fixed alpha threshold (more robust)

    # Internal state
    _steps: list[int] = field(default_factory=list, init=False, repr=False)
    _losses: list[float] = field(default_factory=list, init=False, repr=False)
    _D_history: list[float] = field(default_factory=list, init=False, repr=False)
    _history: list[WTwinState] = field(default_factory=list, init=False, repr=False)

    # Threshold freeze state
    _T_frozen: float = field(default=0.0, init=False, repr=False)
    _T_frozen_active: bool = field(default=False, init=False, repr=False)
    _consec_count: int = field(default=0, init=False, repr=False)
    _in_alert: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        # Propagate calibration params to default baseline if it's PowerLawBaseline
        if isinstance(self.baseline, PowerLawBaseline):
            self.baseline.calibration_frac = self.calibration_frac
            self.baseline.warmup_steps = self.warmup_steps

    def update(self, step: int, l_obs: float) -> WTwinState:
        """
        Ingest one new (step, loss) observation and return current state.

        The baseline is (re)fit once enough calibration data is available.
        Before that, returns a neutral state with W=0 and alert=False.

        Parameters
        ----------
        step  : current training step (1-indexed)
        l_obs : observed loss value

        Returns
        -------
        WTwinState with all computed values at this step.
        """
        self._steps.append(step)
        self._losses.append(l_obs)

        steps_arr = np.array(self._steps, dtype=float)
        losses_arr = np.array(self._losses, dtype=float)

        # ── 1. Fit baseline once calibration window is complete ──────────────
        # Require a minimum absolute window (not just a fraction of n_clean so far),
        # to avoid degenerate fits on 5-point windows.
        n_clean = int(np.sum(steps_arr > self.warmup_steps))
        # n_cal_needed = fraction of TOTAL expected steps, with a hard floor of 100
        n_cal_needed = max(100, int(n_clean * self.calibration_frac))

        if not self.baseline.is_fitted and n_clean >= n_cal_needed:
            try:
                self.baseline.fit(steps_arr, losses_arr)
            except ValueError:
                # Not enough data yet — return neutral state
                return self._neutral_state(step, l_obs)

        if not self.baseline.is_fitted:
            return self._neutral_state(step, l_obs)

        # ── 2. L_pred ────────────────────────────────────────────────────────
        l_pred = float(self.baseline.predict(float(step)))

        # ── 3. Q(t) = exp(−MSE_fit / τ) ─────────────────────────────────────
        Q = float(np.exp(-self.baseline.fit_mse / self.tau))
        Q = np.clip(Q, 0.0, 1.0)

        # ── 4. D(t) — MAD-normalised deviation ───────────────────────────────
        # Use MAD of the PREVIOUS window (exclude current step) so that
        # a sudden spike doesn't collapse sigma to near-zero and distort D.
        sigma_local = mad_scale(self._losses[:-1], window=self.mad_window) \
                      if len(self._losses) > 1 else 1e-8
        D = (l_obs - l_pred) / sigma_local
        # Clip D to prevent numerical explosion (e.g. after weight corruption)
        D = float(np.clip(D, -1e4, 1e4))

        # ── 5. T(t) — hybrid threshold ────────────────────────────────────────
        # For spike detection: use fixed alpha directly (D must exceed alpha)
        # For drift detection: use adaptive rolling threshold
        # Hybrid: T = min(alpha, adaptive_T) — alert fires on whichever is easier
        # When alert is active, T is frozen at the value that triggered it.
        if self._T_frozen_active:
            T = self._T_frozen
        else:
            if self.use_adaptive_T and len(self._D_history) >= 10:
                T_adaptive = self._compute_T()
                # Only use adaptive T if it's lower than fixed alpha
                # (adaptive T helps for drift; fixed alpha catches spikes)
                T = min(T_adaptive, self.alpha)
            else:
                T = self.alpha

        self._D_history.append(D)

        # ── 6. W(t) = Q · (D − T) ─────────────────────────────────────────────
        W = float(Q * (D - T))

        # ── 7. Alert logic — require n_consec consecutive positives ───────────
        if W > 0:
            self._consec_count += 1
        else:
            self._consec_count = 0

        alert = self._consec_count >= self.n_consec

        # Freeze T when alert starts; thaw when W drops below 0
        if alert and not self._in_alert:
            self._T_frozen = T
            self._T_frozen_active = True
            self._in_alert = True
        elif not alert and self._in_alert:
            self._T_frozen_active = False
            self._in_alert = False

        state = WTwinState(
            step=step,
            l_obs=l_obs,
            l_pred=l_pred,
            D=float(D),
            Q=float(Q),
            T=float(T),
            W=float(W),
            alert=alert,
        )
        self._history.append(state)
        return state

    def _compute_T(self) -> float:
        """Compute adaptive threshold from D history."""
        if len(self._D_history) < 2:
            return 0.0
        window = self._D_history[-self.threshold_window:]
        arr = np.array(window)
        mu = float(np.mean(arr))
        sigma = float(np.std(arr, ddof=1)) if len(arr) > 1 else 1e-8
        return mu + self.alpha * sigma

    def _neutral_state(self, step: int, l_obs: float) -> WTwinState:
        """Return a zero-W state before baseline is fitted."""
        return WTwinState(
            step=step, l_obs=l_obs, l_pred=float("nan"),
            D=0.0, Q=0.0, T=0.0, W=0.0, alert=False
        )

    @property
    def history(self) -> list[WTwinState]:
        """Full history of WTwinState objects."""
        return self._history

    def first_alert_step(self) -> int | None:
        """Return step number of first alert, or None if no alert occurred."""
        for s in self._history:
            if s.alert:
                return s.step
        return None

    def reset(self) -> None:
        """Reset monitor state (keeps baseline parameters)."""
        self._steps.clear()
        self._losses.clear()
        self._D_history.clear()
        self._history.clear()
        self._consec_count = 0
        self._in_alert = False
        self._T_frozen_active = False
        self._T_frozen = 0.0
        if hasattr(self.baseline, '_fitted'):
            object.__setattr__(self.baseline, '_fitted', False)
