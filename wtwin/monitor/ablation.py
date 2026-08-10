"""
wtwin.monitor.ablation
==============================
Configurable W-Twin for ablation study.

Single implementation, components enabled/disabled via flags:
    use_q              : include Q(t) confidence weighting
    use_adaptive_T     : adaptive rolling threshold vs fixed alpha*sigma_global
    n_consec           : consecutive steps required (set to 1 to disable)

Variants tested:
    D only             : use_q=False, use_adaptive_T=False, n_consec=1
    D + T (adaptive)   : use_q=False, use_adaptive_T=True
    D + Q              : use_q=True,  use_adaptive_T=False
    Full W-Twin        : use_q=True,  use_adaptive_T=True  ← reference
"""

from __future__ import annotations

import time
import numpy as np
from dataclasses import dataclass, field
from scipy import stats as scipy_stats

from wtwin.common.math_utils import (
    fit_power_law, power_law_predict, mad_scale
)
from wtwin.monitor.benchmark import (
    generate_clean_run, generate_failure_run,
    ThresholdDetector, CUSUMDetector,
)


# ── Configurable W-Twin ───────────────────────────────────────────────────────

@dataclass
class ConfigurableWTwin:
    """
    Single W-Twin implementation with ablation flags.

    Parameters
    ----------
    use_q           : weight deviation by baseline confidence Q(t)
    use_adaptive_T  : adaptive rolling threshold; if False, use fixed T = alpha
    warmup_steps    : steps excluded from baseline fit
    cal_min         : minimum calibration window (absolute steps after warmup)
    mad_window      : window for local MAD
    threshold_window: rolling window for adaptive T
    alpha           : sensitivity (T = μ + α·σ if adaptive, else T = α directly)
    tau             : Q decay scale
    n_consec        : consecutive above-zero W steps required for alert
    """
    use_q: bool = True
    use_adaptive_T: bool = True
    warmup_steps: int = 50
    cal_min: int = 100
    mad_window: int = 50
    threshold_window: int = 150
    alpha: float = 2.0
    tau: float = 1e-3
    n_consec: int = 5

    def run(self, steps: np.ndarray, losses: np.ndarray) -> dict:
        """
        Run monitor on a complete (steps, losses) trajectory.

        Returns
        -------
        dict with keys:
            first_alert_step : int or None
            update_times_us  : list of per-step update times in microseconds
            w_at_alert       : float or None (W score at first alert)
        """
        steps = np.asarray(steps, dtype=float)
        losses = np.asarray(losses, dtype=float)

        a = b = c = None
        fit_mse = 1.0
        D_history: list[float] = []
        consec = 0
        in_alert = False
        T_frozen = 0.0
        first_alert = None
        w_at_alert = None
        update_times: list[float] = []

        for i, (s, l) in enumerate(zip(steps, losses)):
            t0 = time.perf_counter()

            # ── Fit baseline once cal_min clean steps available ───────────────
            n_clean = int(np.sum(steps[:i+1] > self.warmup_steps))
            if a is None and n_clean >= self.cal_min:
                lb = 0
                cal_s = steps[lb:i+1]
                cal_l = losses[lb:i+1]
                mask = cal_s > self.warmup_steps
                if mask.sum() >= 20:
                    try:
                        a, b, c = fit_power_law(cal_s[mask], cal_l[mask])
                        preds = power_law_predict(cal_s[mask], a, b, c)
                        fit_mse = float(np.mean((cal_l[mask] - preds) ** 2))
                    except Exception:
                        pass

            if a is None:
                update_times.append((time.perf_counter() - t0) * 1e6)
                D_history.append(0.0)
                continue

            # ── L_pred ────────────────────────────────────────────────────────
            l_pred = float(power_law_predict(s, a, b, c))

            # ── Q(t) ──────────────────────────────────────────────────────────
            Q = float(np.clip(np.exp(-fit_mse / self.tau), 0.0, 1.0)) \
                if self.use_q else 1.0

            # ── D(t) — MAD-normalised deviation ──────────────────────────────
            sigma = mad_scale(losses[max(0, i - self.mad_window):i + 1])
            D = (l - l_pred) / sigma
            D_history.append(D)

            # ── T(t) ──────────────────────────────────────────────────────────
            if in_alert:
                T = T_frozen
            elif self.use_adaptive_T:
                dw = np.array(D_history[-self.threshold_window:])
                T = float(np.mean(dw) + self.alpha *
                          (np.std(dw, ddof=1) if len(dw) > 1 else 1e-8))
            else:
                # Fixed threshold: T = alpha (in z-score units)
                T = self.alpha

            # ── W(t) = Q · (D − T) ───────────────────────────────────────────
            W = Q * (D - T)

            if W > 0:
                consec += 1
            else:
                consec = 0

            if consec >= self.n_consec and first_alert is None:
                first_alert = int(s)
                w_at_alert = float(W)
                in_alert = True
                T_frozen = T

            update_times.append((time.perf_counter() - t0) * 1e6)

        return {
            "first_alert_step": first_alert,
            "update_times_us": update_times,
            "w_at_alert": w_at_alert,
        }


# ── Metrics computation ───────────────────────────────────────────────────────

def _bootstrap_ci(values: list[float], n_boot: int = 1000,
                  ci: float = 0.95) -> tuple[float, float]:
    """Bootstrap confidence interval for the mean."""
    if not values:
        return (float("nan"), float("nan"))
    arr = np.array(values)
    boot_means = [np.mean(arr[np.random.randint(0, len(arr), len(arr))])
                  for _ in range(n_boot)]
    lo = np.percentile(boot_means, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return float(lo), float(hi)


@dataclass
class VariantResult:
    """Full metrics for one ablation variant."""
    name: str
    n_failure: int
    n_clean: int

    detection_delays: list[int | None]   # steps after true failure (None = missed)
    false_alarms: int
    w_at_alert_values: list[float]
    update_times_us: list[float]         # all per-step times across all runs

    @property
    def detection_rate(self) -> float:
        detected = sum(1 for d in self.detection_delays if d is not None)
        return detected / self.n_failure if self.n_failure else 0.0

    @property
    def false_alarm_rate(self) -> float:
        return self.false_alarms / self.n_clean if self.n_clean else 0.0

    @property
    def detected_delays(self) -> list[int]:
        return [d for d in self.detection_delays if d is not None]

    @property
    def precision(self) -> float:
        """TP / (TP + FP) — of all alerts, fraction on real failures."""
        tp = len(self.detected_delays)
        fp = self.false_alarms
        return tp / (tp + fp) if (tp + fp) > 0 else float("nan")

    @property
    def recall(self) -> float:
        """= detection rate."""
        return self.detection_rate

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if p + r == 0 or np.isnan(p):
            return float("nan")
        return 2 * p * r / (p + r)

    def delay_stats(self) -> dict:
        d = self.detected_delays
        if not d:
            return {"mean": None, "median": None, "std": None, "ci95": (None, None)}
        lo, hi = _bootstrap_ci(d)
        return {
            "mean": float(np.mean(d)),
            "median": float(np.median(d)),
            "std": float(np.std(d, ddof=1)) if len(d) > 1 else 0.0,
            "ci95": (lo, hi),
        }

    def timing_stats(self) -> dict:
        t = self.update_times_us
        if not t:
            return {"mean_us": None, "p99_us": None}
        return {
            "mean_us": float(np.mean(t)),
            "p99_us": float(np.percentile(t, 99)),
        }


# ── Ablation runner ───────────────────────────────────────────────────────────

VARIANTS: dict[str, dict] = {
    "D only":        {"use_q": False, "use_adaptive_T": False, "n_consec": 1},
    "D + Q":         {"use_q": True,  "use_adaptive_T": False, "n_consec": 5},
    "D + T":         {"use_q": False, "use_adaptive_T": True,  "n_consec": 5},
    "Full W-Twin":   {"use_q": True,  "use_adaptive_T": True,  "n_consec": 5},
}


def run_ablation(
    n_failure: int = 30,
    n_clean: int = 30,
    total_steps: int = 1000,
    failure_step: int = 600,
    failure_types: list[str] | None = None,
    seed: int = 42,
    include_baselines: bool = True,
) -> dict[str, dict[str, VariantResult]]:
    """
    Run full ablation across all variants and failure types.

    Returns
    -------
    results[failure_type][variant_name] = VariantResult
    """
    failure_types = failure_types or ["spike", "drift", "diverge"]
    rng_fail = np.random.default_rng(seed)
    rng_clean = np.random.default_rng(seed + 1000)

    # Pre-generate all runs to ensure identical data across variants
    failure_runs: dict[str, list] = {ft: [] for ft in failure_types}
    for ft in failure_types:
        for _ in range(n_failure):
            steps, losses, true_step = generate_failure_run(
                total_steps=total_steps, failure_step=failure_step,
                failure_type=ft, rng=rng_fail,
            )
            failure_runs[ft].append((steps, losses, true_step))

    clean_runs = [
        generate_clean_run(total_steps=total_steps, rng=rng_clean)
        for _ in range(n_clean)
    ]

    results: dict[str, dict[str, VariantResult]] = {}

    for ft in failure_types:
        results[ft] = {}

        # ── W-Twin variants ───────────────────────────────────────────────────
        for vname, vkwargs in VARIANTS.items():
            monitor = ConfigurableWTwin(**vkwargs)
            delays, fa, w_vals, times = [], 0, [], []

            for steps, losses, true_step in failure_runs[ft]:
                out = monitor.run(steps, losses)
                alert = out["first_alert_step"]
                delays.append((alert - true_step) if alert else None)
                if out["w_at_alert"] is not None:
                    w_vals.append(out["w_at_alert"])
                times.extend(out["update_times_us"])

            for steps, losses in clean_runs:
                out = monitor.run(steps, losses)
                if out["first_alert_step"] is not None:
                    fa += 1
                times.extend(out["update_times_us"])

            results[ft][vname] = VariantResult(
                name=vname, n_failure=n_failure, n_clean=n_clean,
                detection_delays=delays, false_alarms=fa,
                w_at_alert_values=w_vals, update_times_us=times,
            )

        # ── External baselines ────────────────────────────────────────────────
        if include_baselines:
            thresh = ThresholdDetector()
            cusum = CUSUMDetector()

            for bname, detector in [("Threshold", thresh), ("CUSUM", cusum)]:
                delays, fa, times = [], 0, []

                for steps, losses, true_step in failure_runs[ft]:
                    t0 = time.perf_counter()
                    alert = detector.detect(steps, losses)
                    elapsed = (time.perf_counter() - t0) * 1e6 / len(steps)
                    delays.append((alert - true_step) if alert else None)
                    times.append(elapsed)

                for steps, losses in clean_runs:
                    t0 = time.perf_counter()
                    alert = detector.detect(steps, losses)
                    elapsed = (time.perf_counter() - t0) * 1e6 / len(steps)
                    if alert is not None:
                        fa += 1
                    times.append(elapsed)

                results[ft][bname] = VariantResult(
                    name=bname, n_failure=n_failure, n_clean=n_clean,
                    detection_delays=delays, false_alarms=fa,
                    w_at_alert_values=[], update_times_us=times,
                )

    return results


# ── Report formatter ──────────────────────────────────────────────────────────

def print_ablation_report(results: dict[str, dict[str, VariantResult]]) -> None:
    """Print formatted ablation report to stdout."""

    VARIANT_ORDER = ["D only", "D + Q", "D + T", "Full W-Twin", "Threshold", "CUSUM"]

    print("\n" + "=" * 80)
    print("  W-TWIN ABLATION STUDY")
    print("=" * 80)

    for ft, variants in results.items():
        print(f"\n  Failure type: {ft.upper()}")
        print(f"  {'Variant':<16} {'Det%':>6} {'Delay μ':>8} {'Delay σ':>8} "
              f"{'CI95 lo':>8} {'CI95 hi':>8} {'FAR':>6} {'Prec':>6} "
              f"{'F1':>6} {'μs/step':>8}")
        print(f"  {'-'*88}")

        for vname in VARIANT_ORDER:
            if vname not in variants:
                continue
            r = variants[vname]
            ds = r.delay_stats()
            ts = r.timing_stats()

            det = f"{r.detection_rate:.0%}"
            delay_mean = f"{ds['mean']:.1f}" if ds['mean'] is not None else "—"
            delay_std = f"{ds['std']:.1f}" if ds['std'] is not None else "—"
            ci_lo = f"{ds['ci95'][0]:.1f}" if ds['ci95'][0] is not None else "—"
            ci_hi = f"{ds['ci95'][1]:.1f}" if ds['ci95'][1] is not None else "—"
            far = f"{r.false_alarm_rate:.0%}"
            prec = f"{r.precision:.2f}" if not np.isnan(r.precision) else "—"
            f1 = f"{r.f1:.2f}" if not np.isnan(r.f1) else "—"
            timing = f"{ts['mean_us']:.2f}" if ts['mean_us'] is not None else "—"

            print(f"  {vname:<16} {det:>6} {delay_mean:>8} {delay_std:>8} "
                  f"{ci_lo:>8} {ci_hi:>8} {far:>6} {prec:>6} "
                  f"{f1:>6} {timing:>8}")

    print("\n" + "=" * 80)
    print("  Det%    = Detection Rate  (fraction of failure runs detected)")
    print("  Delay   = Steps after true failure step (lower = earlier detection)")
    print("  CI95    = 95% bootstrap confidence interval on mean delay")
    print("  FAR     = False Alarm Rate on clean runs")
    print("  Prec    = Precision: TP / (TP + FP)")
    print("  F1      = Harmonic mean of Precision and Recall")
    print("  μs/step = Mean per-step compute time (streaming overhead)")
    print("=" * 80 + "\n")
