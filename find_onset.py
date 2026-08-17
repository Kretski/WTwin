"""
find_onset.py
=============
Намираме точния anomaly onset в anomalous Pythia run.

Методи:
1. Rolling slope — кога loss спира да намалява
2. Changepoint — кога variance се увеличава
3. Visual checkpoints — loss @ key steps
"""
import json
import numpy as np

with open('pythia_anomalous.json') as f:
    data = json.load(f)

steps  = np.array([d[0] for d in data], dtype=float)
losses = np.array([d[1] for d in data])

print('='*55)
print('Anomaly Onset Analysis — Pythia anomalous run')
print('='*55)
print(f'Points: {len(steps)}')
print(f'Steps:  {int(steps[0])} → {int(steps[-1])}')
print(f'Loss:   {losses[0]:.4f} → {losses[-1]:.4f}')
print()

# 1. Loss @ key checkpoints
print('Loss trajectory (every ~10K steps):')
stride = max(1, len(steps) // 15)
for i in range(0, len(steps), stride):
    print(f'  step {int(steps[i]):7d}: loss={losses[i]:.4f}')
print()

# 2. Rolling slope — кога loss спира да намалява
window = 20
slopes = []
for i in range(window, len(steps)):
    w_steps  = steps[i-window:i]
    w_losses = losses[i-window:i]
    slope = np.polyfit(w_steps, w_losses, 1)[0]
    slopes.append((int(steps[i]), slope))

# Намираме кога slope се обръща (от отрицателен към ~0 или позитивен)
print('Rolling slope analysis (window=20):')
plateau_steps = [(s, sl) for s, sl in slopes if sl > -0.000005]
if plateau_steps:
    onset = plateau_steps[0][0]
    print(f'  Slope → 0 first @ step {onset}')
    print(f'  (loss barely decreasing after this point)')
else:
    print('  Loss consistently decreasing throughout')
print()

# 3. Намираме W-Twin alert context
ALERT_STEP = 2051
print(f'W-Twin alert @ step {ALERT_STEP}')
alert_idx = np.searchsorted(steps, ALERT_STEP)
if alert_idx < len(steps):
    print(f'  Loss at alert: {losses[alert_idx]:.4f}')
    print(f'  Loss at start: {losses[0]:.4f}')
    print(f'  Loss change:   {losses[alert_idx] - losses[0]:+.4f}')
print()

# 4. Сравнение loss change rate преди и след alert
pre_alert  = losses[steps < ALERT_STEP]
post_alert = losses[steps >= ALERT_STEP]
if len(pre_alert) > 1 and len(post_alert) > 1:
    pre_steps  = steps[steps < ALERT_STEP]
    post_steps = steps[steps >= ALERT_STEP]
    pre_slope  = np.polyfit(pre_steps,  pre_alert,  1)[0]
    post_slope = np.polyfit(post_steps, post_alert, 1)[0]
    print(f'Convergence rate:')
    print(f'  Pre-alert  slope: {pre_slope:.8f} loss/step')
    print(f'  Post-alert slope: {post_slope:.8f} loss/step')
    print(f'  Ratio: {post_slope/pre_slope:.2f}x')
    print()

# 5. Заключение
print('='*55)
print('ЗАКЛЮЧЕНИЕ:')
if plateau_steps:
    lead = plateau_steps[0][0] - ALERT_STEP
    print(f'  Anomaly onset (slope→0): step {plateau_steps[0][0]}')
    print(f'  W-Twin alert:            step {ALERT_STEP}')
    if lead > 0:
        print(f'  Lead time: {lead} steps ПРЕДИ plateau')
        print(f'  ✅ W-Twin предупреждава ПРЕДИ видимата стагнация')
    else:
        print(f'  W-Twin алармира {abs(lead)} steps СЛЕД plateau onset')
        print(f'  ⚠ Detection е след onset, не преди')
else:
    print('  Loss намалява монотонно — аномалията е в')
    print('  началния trajectory (resumed checkpoint)')
    print(f'  W-Twin хваща отклонение от baseline @ step {ALERT_STEP}')
print('='*55)
