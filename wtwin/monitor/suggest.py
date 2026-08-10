"""
wtwin.monitor.suggest
=============================
Failure classification and structured suggestion after W-Twin alert.

suggest() analyzes the W-Twin history and returns a structured
recommendation. It does NOT modify the optimizer, learning rate,
or any training state. All actions are advisory only.

Failure types detected:
    abrupt_spike   — sudden large D(t) jump (e.g. weight corruption,
                     hardware fault, bad batch)
    gradual_drift  — slow monotonic increase in D(t) over many steps
                     (e.g. label noise, data pipeline drift)
    uncertain      — signal is ambiguous; manual investigation recommended

Suggestions:
    consider_rollback      — reload last good checkpoint
    consider_lr_reduction  — reduce learning rate (try 0.5x)
    manual_review          — inspect data, hardware, and training config

Status: EXPERIMENTAL
    The classifier uses heuristics derived from synthetic experiments.
    It has NOT been validated on a labeled dataset of real training
    failures. Treat suggestions as informational, not authoritative.

Reference:
    Kretski, D. (2026). ScalePredict. https://github.com/Kretski/WTwin
    DOI: 10.5281/zenodo.21851340
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np


# ── Types ─────────────────────────────────────────────────────────────────────

FailureType = Literal["abrupt_spike", "gradual_drift", "uncertain"]
SuggestionType = Literal[
    "consider_rollback",
    "consider_lr_reduction",
    "manual_review",
]


@dataclass(frozen=True)
class Suggestion:
    """
    Structured recommendation returned by suggest().

    All fields are informational. No training state is modified.

    Attributes
    ----------
    alert          : True if W-Twin has fired an alert
    failure_type   : Classified failure pattern (or 'uncertain')
    confidence     : Classifier confidence in [0, 1]
    suggestion     : Recommended action keyword
    action         : Always 'manual_review' — human must decide
    reasoning      : Human-readable explanation
    w_at_alert     : W(t) score at first alert step
    first_alert_step : Step number of first alert (None if no alert)
    d_slope        : Estimated slope of D(t) over recent window
    d_spike        : Peak D(t) in the window around the alert

    Warning
    -------
    This classifier is EXPERIMENTAL and based on heuristics from
    synthetic fault-injection experiments. It has not been validated
    on a labeled real-failure dataset.
    """

    alert: bool
    failure_type: FailureType | None
    confidence: float
    suggestion: SuggestionType | None
    action: str
    reasoning: str
    w_at_alert: float | None
    first_alert_step: int | None
    d_slope: float | None
    d_spike: float | None

    def as_dict(self) -> dict:
        """Return as plain dict (e.g. for JSON serialization)."""
        return {
            "alert": self.alert,
            "failure_type": self.failure_type,
            "confidence": round(self.confidence, 3) if self.confidence else None,
            "suggestion": self.suggestion,
            "action": self.action,
            "reasoning": self.reasoning,
            "w_at_alert": round(self.w_at_alert, 3) if self.w_at_alert else None,
            "first_alert_step": self.first_alert_step,
            "d_slope": round(self.d_slope, 4) if self.d_slope is not None else None,
            "d_spike": round(self.d_spike, 3) if self.d_spike is not None else None,
            "_warning": (
                "EXPERIMENTAL: classifier trained on synthetic data only. "
                "Not validated on real labeled failures."
            ),
        }

    def __str__(self) -> str:
        if not self.alert:
            return "No alert — training appears normal."
        lines = [
            f"⚠  W-Twin Alert at step {self.first_alert_step}",
            f"   Failure type : {self.failure_type}  (confidence: {self.confidence:.0%})",
            f"   Suggestion   : {self.suggestion}",
            f"   Action       : {self.action}",
            f"   Reasoning    : {self.reasoning}",
            f"   [EXPERIMENTAL — validate before acting]",
        ]
        return "\n".join(lines)


# ── Classifier ─────────────────────────────────────────────────────────────────

def _estimate_d_slope(d_history: list[float], window: int = 50) -> float:
    """
    Estimate the linear slope of D(t) over the last `window` steps.
    Positive slope = D is increasing (drift signal).
    """
    arr = np.array(d_history[-window:])
    if len(arr) < 5:
        return 0.0
    x = np.arange(len(arr), dtype=float)
    coeffs = np.polyfit(x, arr, 1)
    return float(coeffs[0])


def _classify(
    d_history: list[float],
    w_history: list[float],
    alert_idx: int,
    window_before: int = 80,
    spike_ratio_threshold: float = 20.0,
    drift_slope_threshold: float = 0.03,
) -> tuple[FailureType, float, float, float]:
    """
    Classify the failure type from D(t) history.

    Returns
    -------
    (failure_type, confidence, d_slope, d_spike)

    Heuristics
    ----------
    abrupt_spike:
        |D(t)| jumps to >> median(|D|) in the tight pre-alert window.
        Uses MAD-based ratio to be robust against pre-existing large D.

    gradual_drift:
        D(t) has a sustained positive slope over the window before alert.
        Slope > drift_slope_threshold.

    uncertain:
        Neither pattern is clearly dominant.
    """
    arr = np.array(d_history)
    n = len(arr)

    # Tight pre-alert window (last 30 steps before alert) — for spike detection
    tight_start = max(0, alert_idx - 30)
    pre_tight = arr[tight_start:alert_idx] if alert_idx > 0 else arr[:1]

    # Wide pre-alert window — for slope/drift detection
    wide_start = max(0, alert_idx - window_before)
    pre_wide = arr[wide_start:alert_idx] if alert_idx > 0 else arr[:1]

    # Post-alert window
    post_alert = arr[alert_idx:min(alert_idx + 10, n)]

    # ── Spike detection (MAD-ratio based) ────────────────────────────────────
    pre_median_abs = float(np.median(np.abs(pre_tight))) if len(pre_tight) > 0 else 1.0
    pre_median_abs = max(pre_median_abs, 0.1)
    d_spike = float(np.max(np.abs(post_alert))) if len(post_alert) > 0 else 0.0
    spike_ratio = d_spike / pre_median_abs

    # ── Drift detection (slope based) ─────────────────────────────────────────
    d_slope = _estimate_d_slope(list(pre_wide), window=min(window_before, len(pre_wide)))
    drift_score = max(0.0, d_slope / drift_slope_threshold)

    # ── Classify ──────────────────────────────────────────────────────────────
    if spike_ratio > spike_ratio_threshold:
        failure_type: FailureType = "abrupt_spike"
        raw_conf = min(1.0, math.log10(max(spike_ratio, 1)) / math.log10(spike_ratio_threshold * 10))
    elif drift_score > 1.0:
        failure_type = "gradual_drift"
        raw_conf = min(1.0, drift_score / 4.0)
    else:
        failure_type = "uncertain"
        raw_conf = 0.5

    confidence = float(np.clip(raw_conf, 0.50, 0.95))
    return failure_type, confidence, d_slope, d_spike


def _make_suggestion(
    failure_type: FailureType,
    confidence: float,
) -> tuple[SuggestionType, str]:
    """Map failure type to suggestion and reasoning text."""

    if failure_type == "abrupt_spike":
        suggestion: SuggestionType = "consider_rollback"
        reasoning = (
            "D(t) shows a sharp, sudden jump typical of abrupt failures "
            "(hardware fault, bad batch, weight corruption). "
            "Rolling back to the last good checkpoint may recover training."
        )
    elif failure_type == "gradual_drift":
        suggestion = "consider_lr_reduction"
        reasoning = (
            "D(t) shows a sustained positive slope — training is gradually "
            "deviating from the expected trajectory. "
            "This may indicate label noise, data pipeline drift, or an "
            "LR schedule issue. Reducing LR by 50% is a conservative first step."
        )
    else:
        suggestion = "manual_review"
        reasoning = (
            "The deviation pattern is ambiguous — neither a clear spike nor "
            "a sustained drift. Inspect data quality, hardware telemetry, "
            "gradient norms, and LR schedule before taking action."
        )

    return suggestion, reasoning


# ── Public API ─────────────────────────────────────────────────────────────────

def suggest(monitor) -> Suggestion:
    """
    Analyse W-Twin monitor history and return a structured suggestion.

    Parameters
    ----------
    monitor : WTwinMonitor
        A WTwinMonitor instance that has processed at least some steps.

    Returns
    -------
    Suggestion
        A frozen dataclass with failure classification and recommendation.
        Does NOT modify any training state.

    Example
    -------
    >>> from wtwin.monitor import WTwinMonitor
    >>> from wtwin.monitor.suggest import suggest
    >>>
    >>> monitor = WTwinMonitor()
    >>> for step, loss in training_loop():
    ...     state = monitor.update(step, loss)
    >>>
    >>> s = suggest(monitor)
    >>> print(s)
    >>> if s.alert:
    ...     print(s.as_dict())

    Warning
    -------
    EXPERIMENTAL. Not validated on real labeled failures.
    """
    history = monitor.history
    first_alert = monitor.first_alert_step()

    # No alert — return clean suggestion
    if not first_alert or not history:
        return Suggestion(
            alert=False,
            failure_type=None,
            confidence=1.0,
            suggestion=None,
            action="none",
            reasoning="No alert detected. Training appears to be on track.",
            w_at_alert=None,
            first_alert_step=None,
            d_slope=None,
            d_spike=None,
        )

    # Find alert index in history
    alert_idx = next(
        (i for i, s in enumerate(history) if s.step == first_alert),
        len(history) - 1,
    )

    d_history = [s.D for s in history if s.D != 0.0]
    w_history = [s.W for s in history]

    if len(d_history) < 5:
        return Suggestion(
            alert=True,
            failure_type="uncertain",
            confidence=0.5,
            suggestion="manual_review",
            action="manual_review",
            reasoning="Insufficient history for classification. Inspect manually.",
            w_at_alert=history[alert_idx].W if alert_idx < len(history) else None,
            first_alert_step=first_alert,
            d_slope=None,
            d_spike=None,
        )

    # Classify
    failure_type, confidence, d_slope, d_spike = _classify(
        d_history, w_history, min(alert_idx, len(d_history) - 1)
    )

    # Map to suggestion
    suggestion_key, reasoning = _make_suggestion(failure_type, confidence)

    w_at_alert = history[alert_idx].W if alert_idx < len(history) else None

    return Suggestion(
        alert=True,
        failure_type=failure_type,
        confidence=confidence,
        suggestion=suggestion_key,
        action="manual_review",  # always manual — no auto-action
        reasoning=reasoning,
        w_at_alert=w_at_alert,
        first_alert_step=first_alert,
        d_slope=d_slope,
        d_spike=d_spike,
    )
