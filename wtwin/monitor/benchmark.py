"""
wtwin.monitor.benchmark
================================
Benchmark harness: Threshold vs CUSUM vs W-Twin on synthetic runs.

Generates controlled training failure scenarios and measures:
  - Detection step (how early alert fires)
  - False alarm rate (alerts on clean runs)
  - Saved compute fraction (steps_remaining / total when alert fires)

Usage
-----
    from wtwin.monitor.benchmark import run_benchmark
    results = run_benchmark(n_clean=50, n_failure=50, seed=42)
    print(results.summary())
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable

from wtwin.monitor.wtwin import WTwinMonitor


# ── Synthetic run generators ──────────────────────────────────────────────────

def generate_clean_run(
    total_steps: int = 1000,
    a: float = 2.0,
    b: float = 0.5,
    c: float = 0.1,
    noise_std: float = 0.02,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a clean power-law training run with Gaussian noise.

    Returns (steps, losses).
    """
    rng = rng or np.random.default_rng()
    steps = np.arange(1, total_steps + 1, dtype=float)
    losses = a * steps ** (-b) + c + rng.normal(0, noise_std, total_steps)
    losses = np.maximum(losses, c)  # physical floor
    return steps, losses


def generate_failure_run(
    total_steps: int = 1000,
    failure_step: int = 600,
    failure_type: str = "spike",  # "spike" | "drift" | "diverge"
    a: float = 2.0,
    b: float = 0.5,
    c: float = 0.1,
    noise_std: float = 0.02,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Generate a training run with injected failure at `failure_step`.

    Failure types:
      spike   — sudden large loss jump, then recovery
      drift   — gradual upward deviation
      diverge — exponential loss growth (NaN precursor)

    Returns (steps, losses, true_failure_step).
    """
    rng = rng or np.random.default_rng()
    steps, losses = generate_clean_run(total_steps, a, b, c, noise_std, rng)

    if failure_type == "spike":
        losses[failure_step:failure_step + 20] += 0.5

    elif failure_type == "drift":
        drift = np.linspace(0, 0.4, total_steps - failure_step)
        losses[failure_step:] += drift

    elif failure_type == "diverge":
        n = total_steps - failure_step
        diverge = 0.01 * np.exp(np.linspace(0, 4, n))
        losses[failure_step:] += diverge

    return steps, losses, failure_step


# ── Baseline detectors for comparison ────────────────────────────────────────

class ThresholdDetector:
    """Naive fixed z-score threshold detector."""

    def __init__(self, threshold: float = 3.0, window: int = 50, n_consec: int = 5):
        self.threshold = threshold
        self.window = window
        self.n_consec = n_consec

    def detect(self, steps: np.ndarray, losses: np.ndarray) -> int | None:
        """Return step of first alert, or None."""
        consec = 0
        for i in range(self.window, len(losses)):
            window = losses[i - self.window:i]
            z = (losses[i] - np.mean(window)) / (np.std(window) + 1e-8)
            if z > self.threshold:
                consec += 1
            else:
                consec = 0
            if consec >= self.n_consec:
                return int(steps[i])
        return None


class CUSUMDetector:
    """CUSUM (Cumulative Sum) change-point detector."""

    def __init__(self, k: float = 0.5, h: float = 5.0, window: int = 100):
        self.k = k
        self.h = h
        self.window = window

    def detect(self, steps: np.ndarray, losses: np.ndarray) -> int | None:
        """Return step of first alert, or None."""
        if len(losses) < self.window:
            return None
        # Estimate mu and sigma from first window
        mu = np.mean(losses[:self.window])
        sigma = np.std(losses[:self.window]) + 1e-8

        S_pos = 0.0
        for i in range(self.window, len(losses)):
            x = (losses[i] - mu) / sigma
            S_pos = max(0.0, S_pos + x - self.k)
            if S_pos > self.h:
                return int(steps[i])
        return None


# ── Benchmark runner ──────────────────────────────────────────────────────────

@dataclass
class BenchmarkResults:
    """Results from a full benchmark run."""
    n_clean: int
    n_failure: int
    total_steps: int

    # Detection step relative to true failure step (negative = early, positive = late)
    wtwin_detection_delays: list[int | None]
    threshold_detection_delays: list[int | None]
    cusum_detection_delays: list[int | None]

    # False alarm rates on clean runs
    wtwin_far: float
    threshold_far: float
    cusum_far: float

    def summary(self) -> str:
        def _mean_delay(delays):
            valid = [d for d in delays if d is not None]
            if not valid:
                return "No detections"
            return f"{np.mean(valid):.1f} steps (detected {len(valid)}/{len(delays)})"

        lines = [
            "=" * 55,
            "  W-Twin Benchmark — Detection Performance",
            "=" * 55,
            f"  Failure runs : {self.n_failure}",
            f"  Clean runs   : {self.n_clean}",
            f"  Total steps  : {self.total_steps}",
            "",
            "  Mean detection delay after failure injection:",
            f"    W-Twin     : {_mean_delay(self.wtwin_detection_delays)}",
            f"    Threshold  : {_mean_delay(self.threshold_detection_delays)}",
            f"    CUSUM      : {_mean_delay(self.cusum_detection_delays)}",
            "",
            "  False Alarm Rate (on clean runs):",
            f"    W-Twin     : {self.wtwin_far:.1%}",
            f"    Threshold  : {self.threshold_far:.1%}",
            f"    CUSUM      : {self.cusum_far:.1%}",
            "=" * 55,
        ]
        return "\n".join(lines)


def run_benchmark(
    n_clean: int = 30,
    n_failure: int = 30,
    total_steps: int = 1000,
    failure_step: int = 600,
    failure_types: list[str] | None = None,
    seed: int = 42,
    wtwin_kwargs: dict | None = None,
) -> BenchmarkResults:
    """
    Run comparative benchmark across n_failure failure runs and n_clean clean runs.

    Parameters
    ----------
    n_clean       : Number of clean runs (for FAR calculation)
    n_failure     : Number of failure runs (equally split across failure_types)
    total_steps   : Steps per run
    failure_step  : True failure injection point
    failure_types : List of failure types to cycle through. Default: all three.
    seed          : Random seed for reproducibility
    wtwin_kwargs  : Optional overrides for WTwinMonitor parameters

    Returns
    -------
    BenchmarkResults with detection delays and false alarm rates.
    """
    rng = np.random.default_rng(seed)
    failure_types = failure_types or ["spike", "drift", "diverge"]
    wtwin_kwargs = wtwin_kwargs or {}

    threshold_det = ThresholdDetector()
    cusum_det = CUSUMDetector()

    # ── Failure runs ──────────────────────────────────────────────────────────
    wtwin_delays, thresh_delays, cusum_delays = [], [], []

    for i in range(n_failure):
        ftype = failure_types[i % len(failure_types)]
        steps, losses, true_step = generate_failure_run(
            total_steps=total_steps,
            failure_step=failure_step,
            failure_type=ftype,
            rng=rng,
        )

        # W-Twin
        monitor = WTwinMonitor(**wtwin_kwargs)
        for s, l in zip(steps, losses):
            state = monitor.update(int(s), float(l))
        alert_step = monitor.first_alert_step()
        wtwin_delays.append(
            (alert_step - true_step) if alert_step is not None else None
        )

        # Threshold
        thresh_delays.append(
            (threshold_det.detect(steps, losses) or 0) - true_step
            if threshold_det.detect(steps, losses) else None
        )

        # CUSUM
        cusum_delays.append(
            (cusum_det.detect(steps, losses) or 0) - true_step
            if cusum_det.detect(steps, losses) else None
        )

    # ── Clean runs (False Alarm Rate) ─────────────────────────────────────────
    wtwin_fa, thresh_fa, cusum_fa = 0, 0, 0

    for _ in range(n_clean):
        steps, losses = generate_clean_run(total_steps=total_steps, rng=rng)

        monitor = WTwinMonitor(**wtwin_kwargs)
        for s, l in zip(steps, losses):
            monitor.update(int(s), float(l))
        if monitor.first_alert_step() is not None:
            wtwin_fa += 1

        if threshold_det.detect(steps, losses) is not None:
            thresh_fa += 1

        if cusum_det.detect(steps, losses) is not None:
            cusum_fa += 1

    return BenchmarkResults(
        n_clean=n_clean,
        n_failure=n_failure,
        total_steps=total_steps,
        wtwin_detection_delays=wtwin_delays,
        threshold_detection_delays=thresh_delays,
        cusum_detection_delays=cusum_delays,
        wtwin_far=wtwin_fa / n_clean,
        threshold_far=thresh_fa / n_clean,
        cusum_far=cusum_fa / n_clean,
    )
