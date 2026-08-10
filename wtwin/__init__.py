"""
W-Twin — Forecast-Based Detection of Progressive Neural Network Training Degradation
=====================================================================================

Compares your training loss against a scaling-law baseline at every step.
When the trajectory drifts from what is expected, W-Twin raises an alert —
before classical threshold and CUSUM detectors notice anything.

Usage:
    from wtwin import WTwinMonitor, suggest

    monitor = WTwinMonitor()
    for step, loss in training_loop():
        state = monitor.update(step, loss)
        if state.alert:
            print(f"⚠ Degradation at step {step}  (W={state.W:.2f})")

Paper:
    https://doi.org/10.5281/zenodo.21851340

Repository:
    https://github.com/Kretski/WTwin
"""

from wtwin.monitor import (
    WTwinMonitor,
    WTwinState,
    PowerLawBaseline,
    BaseBaseline,
    run_benchmark,
    suggest,
    Suggestion,
)

__version__ = "1.2.0"
__author__ = "Dimitar Kretski"
__doi__ = "10.5281/zenodo.21851340"

__all__ = [
    "WTwinMonitor",
    "WTwinState",
    "PowerLawBaseline",
    "BaseBaseline",
    "run_benchmark",
    "suggest",
    "Suggestion",
]
