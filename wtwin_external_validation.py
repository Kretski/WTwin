"""
wtwin_external_validation.py
=============================
Blind replay validation на W-Twin върху реални Pythia logs.

Протокол:
  1. Зарежда external CSV/JSON training log
  2. Split: първите CALIBRATION_PCT% → baseline fit
             остатъкът → blind future (скрит при fit)
  3. W-Twin прави forecast само от calibration частта
  4. Сравнява forecast срещу реалния future
  5. Измерва detection: кога W-Twin алармира vs Threshold vs CUSUM
  6. Генерира CSV, JSON, PNG, validation report

Употреба:
  python wtwin_external_validation.py pythia_loss.json
  python wtwin_external_validation.py training_log.csv

Формат на входния файл:
  JSON: [[step, loss], [step, loss], ...]
  CSV:  step,loss (с хедър)

Не пипа W-Twin алгоритъма — валидира v1.2.0 каквото е.
"""

import sys
import json
import csv
import numpy as np
from pathlib import Path

try:
    from wtwin import WTwinMonitor
except ImportError:
    print("ERROR: pip install git+https://github.com/Kretski/WTwin.git")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("matplotlib не е инсталиран — PNG няма да се генерира")

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

CALIBRATION_PCT = 0.20   # Първите 20% → baseline fit
WTWIN_CFG = dict(
    warmup_steps     = 100,
    alpha            = 2.0,
    n_consec         = 7,
    calibration_frac = 0.10,
)
CUSUM_K   = 0.5
CUSUM_H   = 5.0
THRESHOLD_SIGMA = 2.0

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_data(path):
    path = Path(path)
    if path.suffix == '.json':
        with open(path) as f:
            data = json.load(f)
        if isinstance(data[0], list):
            return [(int(s), float(l)) for s, l in data]
        elif isinstance(data[0], dict):
            return [(int(d['step']), float(d['loss'])) for d in data]
    elif path.suffix == '.csv':
        rows = []
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                s = row.get('step') or row.get('_step') or row.get('Step')
                l = row.get('loss') or row.get('Loss') or row.get('train/lm_loss')
                if s and l:
                    rows.append((int(float(s)), float(l)))
        return rows
    raise ValueError(f"Unsupported format: {path.suffix}")

# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def run_cusum(points, k=CUSUM_K, h=CUSUM_H, warmup=100):
    S = 0.0; first = None
    history = [l for _, l in points[:warmup]]
    for i, (step, loss) in enumerate(points):
        history.append(loss)
        if i < warmup: continue
        mu = np.mean(history[-50:])
        S  = max(0, S + (loss - mu) - k)
        if S > h and first is None:
            first = step
    return first

def run_threshold(points, sigma_mult=THRESHOLD_SIGMA, warmup=100):
    baseline = [l for _, l in points[:warmup]]
    mu    = np.mean(baseline)
    sigma = np.std(baseline)
    thresh = mu + sigma_mult * sigma
    for step, loss in points[warmup:]:
        if loss > thresh:
            return step
    return None

# ---------------------------------------------------------------------------
# Forecast accuracy
# ---------------------------------------------------------------------------

def forecast_accuracy(monitor, points, cal_end_idx):
    """
    Измерва колко добре baseline предсказва future trajectory.
    Използва само стъпките СЛЕД calibration периода.
    """
    future = points[cal_end_idx:]
    if not future or not monitor.baseline.is_fitted:
        return None

    steps_f  = np.array([s for s, l in future], dtype=float)
    actual_f = np.array([l for s, l in future])
    pred_f   = np.array([monitor.baseline.predict(s) for s in steps_f])

    residuals = actual_f - pred_f
    mae   = float(np.mean(np.abs(residuals)))
    rmse  = float(np.sqrt(np.mean(residuals**2)))
    bias  = float(np.mean(residuals))
    try:
        corr  = float(np.corrcoef(actual_f, pred_f)[0, 1])
    except Exception:
        corr  = float('nan')

    return {
        'mae':          round(mae, 5),
        'rmse':         round(rmse, 5),
        'bias':         round(bias, 5),
        'correlation':  round(corr, 4),
        'n_future':     len(future),
        'pred_sample':  [(int(s), round(float(p), 5))
                         for s, p in zip(steps_f[::len(steps_f)//10+1], pred_f[::len(pred_f)//10+1])],
    }

# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------

def validate(path):
    print("=" * 65)
    print("W-Twin External Validation — Blind Replay Protocol")
    print(f"File: {path}")
    print("=" * 65)

    points = load_data(path)
    points.sort(key=lambda x: x[0])
    N = len(points)

    print(f"\nData: {N} points")
    print(f"  Steps: {points[0][0]} → {points[-1][0]}")
    print(f"  Loss:  {points[0][1]:.4f} → {points[-1][1]:.4f}")

    # Split
    cal_end = int(N * CALIBRATION_PCT)
    cal_points    = points[:cal_end]
    future_points = points[cal_end:]

    print(f"\nSplit ({CALIBRATION_PCT:.0%} / {1-CALIBRATION_PCT:.0%}):")
    print(f"  Calibration: {len(cal_points)} points "
          f"(steps {cal_points[0][0]}–{cal_points[-1][0]})")
    print(f"  Future:      {len(future_points)} points "
          f"(steps {future_points[0][0]}–{future_points[-1][0]})")

    # W-Twin — fit на calibration, потом full run
    monitor = WTwinMonitor(**WTWIN_CFG)
    alert_step = None
    W_vals     = []

    for step, loss in points:
        state = monitor.update(step, loss)
        W_vals.append((step, state.W, state.alert))
        if state.alert and alert_step is None:
            alert_step = step

    fitted = monitor.baseline.is_fitted
    print(f"\nW-Twin baseline fitted: {fitted}")
    if fitted:
        print(f"  Coefficients: {monitor.baseline.coefficients}")
        print(f"  fit_mse:      {monitor.baseline.fit_mse:.6f}")

    # Forecast accuracy
    acc = forecast_accuracy(monitor, points, cal_end)

    # Baseline comparisons (на пълния run)
    cusum_step  = run_cusum(points)
    thresh_step = run_threshold(points)

    # Detection summary
    print(f"\nDetection results (на пълния run):")
    print(f"  W-Twin first alert: {alert_step if alert_step else '— (no alert)'}")
    print(f"  CUSUM first alert:  {cusum_step if cusum_step else '— (no alert)'}")
    print(f"  Threshold alert:    {thresh_step if thresh_step else '— (no alert)'}")

    if acc:
        print(f"\nForecast accuracy (future {1-CALIBRATION_PCT:.0%}):")
        print(f"  MAE:         {acc['mae']:.5f}")
        print(f"  RMSE:        {acc['rmse']:.5f}")
        print(f"  Bias:        {acc['bias']:+.5f}")
        print(f"  Correlation: {acc['correlation']:.4f}")
        print(f"  N future:    {acc['n_future']}")

    # Lead time analysis
    print(f"\nLead time analysis:")
    detectors = {
        'W-Twin':    alert_step,
        'CUSUM':     cusum_step,
        'Threshold': thresh_step,
    }
    first_any = min((s for s in detectors.values() if s), default=None)
    if first_any:
        for name, step in detectors.items():
            if step:
                lead = step - first_any
                print(f"  {name}: step {step} (lead={lead:+d} vs earliest)")
            else:
                print(f"  {name}: no alert")

    # W trajectory summary
    W_arr = np.array([w for _, w, _ in W_vals if w == w])
    if len(W_arr) > 0:
        print(f"\nW trajectory:")
        print(f"  Range: [{W_arr.min():.3f}, {W_arr.max():.3f}]")
        print(f"  Mean:  {W_arr.mean():.3f}")
        clean = (W_arr < 0).sum()
        print(f"  Steps with W<0: {clean}/{len(W_arr)} ({clean/len(W_arr):.1%})")

    # ---------------------------------------------------------------------------
    # PNG
    # ---------------------------------------------------------------------------
    if HAS_PLOT:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), facecolor='white')
        fig.suptitle(f'W-Twin External Validation — {Path(path).stem}',
                     fontsize=13, color='#0b0b0b')

        steps_arr  = np.array([s for s, l, _ in W_vals])
        losses_arr = np.array([l for s, l, _ in W_vals])
        W_arr_full = np.array([w for _, w, _ in W_vals])

        # Predicted trajectory
        if fitted:
            pred_arr = np.array([monitor.baseline.predict(s) for s in steps_arr])
            ax1.plot(steps_arr, pred_arr, '--', color='#2a78d6',
                     linewidth=1.5, alpha=0.7, label='W-Twin forecast')

        cal_step = cal_points[-1][0]
        ax1.axvline(cal_step, color='gray', linewidth=1, linestyle=':',
                    label=f'Calibration end (step {cal_step})')
        ax1.plot(steps_arr, losses_arr, color='#0b0b0b',
                 linewidth=1, alpha=0.8, label='Actual loss')

        if alert_step:
            ax1.axvline(alert_step, color='#e34948', linewidth=2,
                        label=f'W-Twin alert @{alert_step}')
        if cusum_step:
            ax1.axvline(cusum_step, color='#eb6834', linewidth=1.5,
                        linestyle='--', label=f'CUSUM @{cusum_step}')

        ax1.set_ylabel('Loss', color='#898781', fontsize=10)
        ax1.legend(fontsize=8, framealpha=0.9)
        ax1.grid(color='#e1e0d9', linewidth=0.5)
        ax1.spines[['top','right']].set_visible(False)
        ax1.set_facecolor('white')

        # W(t) plot
        ax2.plot(steps_arr, W_arr_full, color='#2a78d6', linewidth=1, alpha=0.8)
        ax2.axhline(0, color='#e34948', linewidth=1.5, linestyle='--', label='Alert threshold')
        ax2.axvline(cal_step, color='gray', linewidth=1, linestyle=':')
        ax2.fill_between(steps_arr, W_arr_full, 0,
                         where=W_arr_full > 0,
                         color='#e34948', alpha=0.2, label='W > 0 (alert zone)')
        ax2.set_xlabel('Training step', color='#898781', fontsize=10)
        ax2.set_ylabel('W(t)', color='#898781', fontsize=10)
        ax2.legend(fontsize=8, framealpha=0.9)
        ax2.grid(color='#e1e0d9', linewidth=0.5)
        ax2.spines[['top','right']].set_visible(False)
        ax2.set_facecolor('white')

        # KPI текст
        kpi = (f"MAE={acc['mae']:.4f}  RMSE={acc['rmse']:.4f}  "
               f"r={acc['correlation']:.3f}  "
               f"W-Twin alert={'step '+str(alert_step) if alert_step else 'none'}"
               if acc else "")
        fig.text(0.5, 0.01, kpi, ha='center', fontsize=9, color='#898781')

        out_png = Path(path).stem + '_validation.png'
        plt.tight_layout(rect=[0, 0.03, 1, 1])
        plt.savefig(out_png, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"\nSaved PNG → {out_png}")
        plt.close()

    # ---------------------------------------------------------------------------
    # JSON + CSV output
    # ---------------------------------------------------------------------------
    results = {
        'file':          str(path),
        'n_points':      N,
        'calibration_pct': CALIBRATION_PCT,
        'wtwin_config':  WTWIN_CFG,
        'baseline_fitted': fitted,
        'baseline_coefficients': monitor.baseline.coefficients if fitted else None,
        'baseline_mse':  monitor.baseline.fit_mse if fitted else None,
        'wtwin_alert':   alert_step,
        'cusum_alert':   cusum_step,
        'threshold_alert': thresh_step,
        'forecast_accuracy': acc,
        'W_range':       [round(float(W_arr_full.min()), 3),
                          round(float(W_arr_full.max()), 3)] if len(W_arr_full) else None,
        'W_negative_pct': round(float((W_arr_full < 0).mean()), 4) if len(W_arr_full) else None,
    }

    out_json = Path(path).stem + '_validation.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved JSON → {out_json}")

    # CSV на W trajectory
    out_csv = Path(path).stem + '_W_trajectory.csv'
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['step', 'loss', 'W', 'alert'])
        for step, loss, alert in zip(steps_arr, losses_arr, [a for _,_,a in W_vals]):
            w.writerow([int(step), round(float(loss), 5),
                        round(float(W_arr_full[list(steps_arr).index(step)]), 4),
                        int(alert)])
    print(f"Saved CSV → {out_csv}")

    print()
    print("=" * 65)
    print("VALIDATION REPORT")
    print("=" * 65)
    if not fitted:
        print("❌ Baseline not fitted — insufficient calibration data")
        print(f"   Need ≥{WTWIN_CFG['warmup_steps']} warmup + calibration points")
    elif alert_step is None and cusum_step is None:
        print("✅ Clean trajectory — no alerts from any detector")
        print("   This is expected for a normal training run")
        if acc:
            print(f"   Forecast correlation: {acc['correlation']:.3f}")
    elif alert_step and not cusum_step:
        print(f"⚠ W-Twin detected anomaly @ step {alert_step}")
        print(f"  CUSUM and Threshold: no signal")
        print(f"  W-Twin is more sensitive to progressive drift")
    elif cusum_step and not alert_step:
        print(f"⚠ CUSUM detected @ step {cusum_step}, W-Twin: no signal")
        print(f"  Likely abrupt spike — W-Twin is designed for gradual drift")
    else:
        print(f"Both W-Twin ({alert_step}) and CUSUM ({cusum_step}) detected")
        lead = cusum_step - alert_step if alert_step and cusum_step else None
        if lead and lead > 0:
            print(f"  W-Twin was {lead} steps earlier")
    print("=" * 65)

    return results

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python wtwin_external_validation.py <file.json|file.csv>")
        print("Example: python wtwin_external_validation.py pythia_loss.json")
        sys.exit(1)

    validate(sys.argv[1])
