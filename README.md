W-Twin

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21842460.svg)](https://doi.org/10.5281/zenodo.21842460)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Detect progressive training degradation before it becomes visible in your loss curve.**

Most monitors react to what has already happened (spikes, NaNs, large jumps).  
W-Twin compares your live loss trajectory against a scaling-law forecast. When the two diverge, it alerts — often hundreds of steps earlier.

It is designed as a lightweight, complementary layer on top of existing tools (Weights & Biases, custom dashboards, HuggingFace Trainer, etc.).

---

## Why it matters

You are training for days. On day 2 the run starts drifting.  
The loss still looks “normal” locally. You discover the problem only on day 3.

W-Twin is built exactly for this case: **gradual, progressive degradation** that reactive monitors (threshold, CUSUM) often miss.

| Experiment              | W-Twin          | Threshold | CUSUM     |
|-------------------------|-----------------|-----------|-----------|
| Progressive drift       | **9/9 (100%)**  | 0/9       | 0/9       |
| Mean detection delay    | **257 steps**   | —         | —         |
| False alarms (clean)    | **0/30**        | 0/30      | 0/30      |
| Abrupt spike            | 2/2             | 0/2       | **2/2** (faster) |

W-Twin and CUSUM are complementary: use both.

> Results from controlled nano-GPT experiments (842K params) with injected failures.  
> External validation on larger models and real logs is ongoing.

---

## 30-second usage

```python
from wtwin import WTwinMonitor

monitor = WTwinMonitor()

for step, loss in enumerate(training_losses, 1):
    state = monitor.update(step, loss)
    if state.alert:
        print(f"⚠ Degradation detected at step {step}  (W={state.W:.2f})")
CLI (no code changes needed):
Bashpip install git+https://github.com/Kretski/WTwin.git
wtwin monitor training_log.csv
wtwin demo

How it works
W-Twin fits a power-law baseline on early steps, then at every step computes:
textD(t) = (L_obs(t) − L_pred(t)) / σ_local(t)     # deviation from forecast
Q(t) = exp(−MSE_fit / τ)                       # confidence in the baseline
W(t) = Q(t) · (D(t) − α)                       # health score
Alert fires when W(t) > 0 for several consecutive steps.
Reactive methods compare the loss to its recent history.
W-Twin compares it to a forecast of where the loss should be.

Integrations
HuggingFace Trainer
Pythonfrom transformers import TrainerCallback
from wtwin import WTwinMonitor

class WTwinCallback(TrainerCallback):
    def __init__(self):
        self.monitor = WTwinMonitor()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            st = self.monitor.update(state.global_step, logs["loss"])
            if st.alert:
                print(f"⚠ W-Twin alert at step {state.global_step}")
Weights & Biases
Log W, D and Q as custom metrics. You can then create W&B Automations that trigger Slack / email when W crosses zero.
Any training loop
Just call monitor.update(step, loss) — framework agnostic.

Installation
Bashpip install git+https://github.com/Kretski/WTwin.git
Only depends on numpy and scipy.

Limitations

Currently validated on nano-GPT scale with synthetic failures
Power-law baseline assumes generally decreasing loss
suggest() feature is experimental
Larger-scale and multi-architecture validation is in progress

Full details in the paper.

Citation
bibtex@software{kretski2026wtwin,
  author    = {Kretski, Dimitar},
  title     = {W-Twin: Forecast-Based Detection of Progressive Neural Network Training Degradation},
  year      = {2026},
  doi       = {10.5281/zenodo.21842460},
  url       = {https://zenodo.org/records/21865734}
}

Author: Dimitar Kretski
Center for Hydro- and Aerodynamics, Varna, Bulgaria
ORCID: 0000-0001-5108-2243
text
