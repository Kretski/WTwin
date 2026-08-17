"""
scale_free_wtwin.py
===================
Тестваме scale-free W-Twin срещу fixed warmup на реални Pythia logs.

Три конфигурации:
  A. Fixed (текущ): warmup=100, n_consec=7
  B. Scale-free warmup: warmup = 0.005 × T_train
  C. Adaptive warmup: warmup ends when Q(t) > Q_threshold

Два run-а:
  1. Clean (pythia_sampled.json) — очакваме 0 FA при всички конфигурации
  2. Anomalous (pythia_anomalous.json) — очакваме detection при всички

Метрики:
  - False alarms на clean run
  - Detection step на anomalous run
  - Lead time
  - Parameter sensitivity
"""

import json
import numpy as np
from wtwin import WTwinMonitor

# ---------------------------------------------------------------------------
# Зареждаме данните
# ---------------------------------------------------------------------------

def load(path):
    with open(path) as f:
        data = json.load(f)
    pts = [(int(d[0]), float(d[1])) for d in data]
    pts.sort()
    return pts

print("Зареждаме Pythia данни...")
try:
    clean    = load('pythia_sampled.json')
    anomalous = load('pythia_anomalous.json')
    print(f"  Clean:     {len(clean)} points, steps {clean[0][0]}→{clean[-1][0]}")
    print(f"  Anomalous: {len(anomalous)} points, steps {anomalous[0][0]}→{anomalous[-1][0]}")
except FileNotFoundError as e:
    print(f"ERROR: {e}")
    print("Пусни от папката с pythia_sampled.json и pythia_anomalous.json")
    exit(1)

# ---------------------------------------------------------------------------
# Конфигурации
# ---------------------------------------------------------------------------

def make_configs(points):
    T = len(points)
    return {
        'fixed_100':    dict(warmup_steps=100,          alpha=2.0, n_consec=7, calibration_frac=0.10),
        'fixed_500':    dict(warmup_steps=500,          alpha=2.0, n_consec=7, calibration_frac=0.10),
        'scale_free':   dict(warmup_steps=max(50, int(0.005 * T)), alpha=2.0, n_consec=7, calibration_frac=0.10),
        'scale_free_1': dict(warmup_steps=max(50, int(0.010 * T)), alpha=2.0, n_consec=7, calibration_frac=0.10),
        'scale_free_2': dict(warmup_steps=max(50, int(0.002 * T)), alpha=2.0, n_consec=7, calibration_frac=0.10),
        'adaptive_q':   dict(warmup_steps=max(50, int(0.005 * T)), alpha=2.0, n_consec=7,
                             calibration_frac=0.10, use_adaptive_T=True),
    }

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_wtwin(points, cfg):
    mon = WTwinMonitor(**cfg)
    for step, loss in points:
        mon.update(step, loss)
    first  = mon.first_alert_step()
    W_vals = [s.W for s in mon.history if s.W == s.W]
    return {
        'first_alert':   first,
        'W_max':         round(max(W_vals), 3) if W_vals else None,
        'W_min':         round(min(W_vals), 3) if W_vals else None,
        'W_neg_pct':     round(float((np.array(W_vals) < 0).mean()), 4) if W_vals else None,
        'baseline_fitted': mon.baseline.is_fitted,
        'warmup_used':   cfg['warmup_steps'],
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

print()
print("=" * 70)
print("Scale-Free W-Twin Experiment")
print("=" * 70)

configs_clean    = make_configs(clean)
configs_anomalous = make_configs(anomalous)

ANOMALY_STEP = 1901  # onset от find_onset.py

print()
print(f"{'Config':<15} {'Warmup':>8} | {'Clean FA':>10} {'W_neg%':>8} | {'Anom Alert':>12} {'Lead':>8}")
print("-" * 70)

results = []
for cfg_name in configs_clean:
    cfg_c = configs_clean[cfg_name]
    cfg_a = configs_anomalous[cfg_name]

    rc = run_wtwin(clean,     cfg_c)
    ra = run_wtwin(anomalous, cfg_a)

    fa_str     = str(rc['first_alert']) if rc['first_alert'] else "✅ none"
    alert_str  = str(ra['first_alert']) if ra['first_alert'] else "❌ none"
    lead       = (ra['first_alert'] - ANOMALY_STEP) if ra['first_alert'] else None
    lead_str   = f"+{lead}" if lead and lead >= 0 else (str(lead) if lead else "—")
    neg_pct    = f"{rc['W_neg_pct']:.1%}" if rc['W_neg_pct'] else "—"

    print(f"  {cfg_name:<13} {cfg_c['warmup_steps']:>8} | {fa_str:>10} {neg_pct:>8} | "
          f"{alert_str:>12} {lead_str:>8}")

    results.append({
        'config':       cfg_name,
        'warmup':       cfg_c['warmup_steps'],
        'clean_fa':     rc['first_alert'],
        'clean_W_neg':  rc['W_neg_pct'],
        'anom_alert':   ra['first_alert'],
        'lead_time':    lead,
    })

# ---------------------------------------------------------------------------
# Анализ
# ---------------------------------------------------------------------------

print()
print("=" * 70)
print("АНАЛИЗ")
print("=" * 70)

clean_ok    = [r for r in results if r['clean_fa'] is None]
anom_detect = [r for r in results if r['anom_alert'] is not None]
both_ok     = [r for r in results if r['clean_fa'] is None and r['anom_alert'] is not None]

print(f"\n  Clean FA=0:          {len(clean_ok)}/{len(results)} конфигурации")
print(f"  Anomaly detected:    {len(anom_detect)}/{len(results)} конфигурации")
print(f"  Both correct:        {len(both_ok)}/{len(results)} конфигурации")

if both_ok:
    leads = [r['lead_time'] for r in both_ok if r['lead_time'] is not None]
    if leads:
        print(f"\n  Lead time при correct configs:")
        print(f"    min={min(leads)}  max={max(leads)}  mean={np.mean(leads):.0f} steps")

# Scale-free vs Fixed
sf = next((r for r in results if r['config'] == 'scale_free'), None)
f1 = next((r for r in results if r['config'] == 'fixed_100'), None)

if sf and f1:
    print(f"\n  Scale-free (0.5% × T) vs Fixed (100):")
    print(f"    warmup: {sf['warmup']} vs {f1['warmup']}")
    sf_ok = sf['clean_fa'] is None and sf['anom_alert'] is not None
    f1_ok = f1['clean_fa'] is None and f1['anom_alert'] is not None
    print(f"    Scale-free correct: {'✅' if sf_ok else '❌'}")
    print(f"    Fixed correct:      {'✅' if f1_ok else '❌'}")

print()
print("ЗАКЛЮЧЕНИЕ:")
if len(both_ok) == len(results):
    print("  ✅ Всички конфигурации работят — W-Twin е robust към warmup избора")
    print("  Scale-free подход е валиден — не изисква ръчна настройка по модел")
elif len(both_ok) > len(results) // 2:
    print("  ⚠ Повечето конфигурации работят — scale-free е promising")
    fails = [r['config'] for r in results if r not in both_ok]
    print(f"  Проблеми при: {fails}")
else:
    print("  ❌ Само малка част работят — warmup изборът е критичен")
    print("  Scale-free хипотезата не се потвърждава с тези данни")

print()
print("ОГРАНИЧЕНИЯ:")
print("  • Само 2 Pythia runs (clean + anomalous)")
print("  • Anomaly onset е приблизителен (rolling slope, не ground truth)")
print("  • T = брой sampled точки, не реални training steps")
print("  • n_consec=7 е фиксиран — не е scale-free")

# Записваме
import json as json_mod
with open('scale_free_results.json', 'w') as f:
    json_mod.dump(results, f, indent=2)
print("\nSaved → scale_free_results.json")
