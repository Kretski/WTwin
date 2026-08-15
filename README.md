# W-Twin

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21842460.svg)](https://doi.org/10.5281/zenodo.21842460)[![Python 310](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)[![License MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)[](https://kretski.github.io/WTwin/)

**Detect progressive training degradation before it becomes visible in your loss curve.**

Most monitors react to what has already happened — spikes, NaNs, large jumps.W-Twin compares your live loss trajectory against a scaling-law forecast. When the two diverge, it alerts — often hundreds of steps earlier.

Lightweight, framework-agnostic, drop-in alongside existing tools (W&B, HuggingFace Trainer, custom dashboards).

* * *

### Try it instantly (no install)
[![Demo Dashboard](https://img.shields.io/badge/Demo-Dashboard-orange)](https://kretski.github.io/WTwin/wtwin_demo_dashboard.html)
**Live browser demo:** [https://kretski.github.io/WTwin/](https://kretski.github.io/WTwin/)

Drop a CSV with `step` and `loss` columns. Everything runs locally — no upload, no server.

* * *

## Why it matters

You are training for days. On day 2 the run starts drifting.The loss still looks "normal" locally. You discover the problem only on day 3.

W-Twin is built exactly for this case: **gradual, progressive degradation** that reactive monitors miss.

| Experiment | W-Twin | Threshold | CUSUM |
| --- | --- | --- | --- |
| Progressive drift | **9/9 (100%)** | 0/9 | 0/9 |
| Mean detection delay | **257 steps** | —   | —   |
| False alarms (clean) | **0/30** | 0/30 | 0/30 |
| Abrupt spike | 2/2 | 0/2 | **2/2** (faster) |

W-Twin and CUSUM are complementary: use both.

> Controlled nano-GPT experiments (842K params) with injected failures.

* * *

## Real-world validation — EleutherAI Pythia

W-Twin was run on real Pythia training logs from the public EleutherAI WandB project.

| Run | Steps | W-Twin | CUSUM | Threshold |
| --- | --- | --- | --- | --- |
| Clean pretraining | 143,000 | **0 false alarms** | 0   | 0   |
| Anomalous run (resumed checkpoint, stagnated at loss ~2.8) | 143,000 | **Alert @ step 2,031** | no signal | no signal |

W-Twin was the only method that produced a signal on the anomalous run — 141,000 steps before completion.W range on clean run: [−105, −1.7] — stably negative throughout.

> W-Twin detects trajectory deviation, not high final loss.The two higher-final-loss runs that followed normal power-law convergence were correctly not flagged.

* * *

## Drop-in optimizer wrapper

The fastest integration path — three lines replace your existing optimizer:

    from wtwin_optimizer import WTwinAdamW
    
    optimizer = WTwinAdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),        # frontier standard (LLaMA / GPT-2 / Mistral)
        preset='pretraining',      # or 'adaptive_pretraining' for long cosine runs
        wtwin_on_alert=lambda step, W, state: save_checkpoint(step)
    )
    
    # Training loop — only change: pass loss to step()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step(loss=loss)

**Presets:**

| Preset | Use case | Baseline |
| --- | --- | --- |
| `pretraining` | Standard LLM pretraining | PowerLaw |
| `adaptive_pretraining` | Long runs with cosine schedule (2.4× faster detection) | PowerLaw + adaptive T |
| `finetuning` | Full fine-tuning *(experimental)* | ExpFloor |
| `custom` | Manual configuration | any |

Overhead: **0.30 ms per step** — measured against AdamW on a 10M-param model.

Also supports `WTwinSGD` and `WTwinLion`.

* * *

## 30-second usage (core API)

    from wtwin import WTwinMonitor
    
    monitor = WTwinMonitor()
    
    for step, loss in enumerate(training_losses, 1):
        state = monitor.update(step, loss)
        if state.alert:
            print(f"⚠ Degradation at step {step}  (W={state.W:.2f})")

**CLI — no code changes needed:**

    pip install git+https://github.com/Kretski/WTwin.git
    wtwin monitor training_log.csv
    wtwin demo

* * *

## How it works

W-Twin fits a power-law baseline on early steps, then computes at every step:

    D(t) = (L_obs(t) − L_pred(t)) / σ_local(t)   # deviation from forecast
    Q(t) = exp(−MSE_fit / τ)                       # confidence in baseline
    W(t) = Q(t) · (D(t) − α)                       # health score

Alert fires when `W(t) > 0` for several consecutive steps.

Reactive methods compare loss to its recent history.W-Twin compares it to a **forecast of where loss should be**.

* * *

## Integrations

**HuggingFace Trainer**

    from transformers import TrainerCallback
    from wtwin import WTwinMonitor
    
    class WTwinCallback(TrainerCallback):
        def __init__(self):
            self.monitor = WTwinMonitor()
    
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                st = self.monitor.update(state.global_step, logs["loss"])
                if st.alert:
                    print(f"⚠ W-Twin alert at step {state.global_step}")

**Weights & Biases**Log `W`, `D`, and `Q` as custom metrics, then create W&B Automations to trigger Slack or email when W crosses zero.

**Any training loop**`monitor.update(step, loss)` — framework agnostic.

* * *

## Installation

    pip install git+https://github.com/Kretski/WTwin.git

Dependencies: `numpy`, `scipy`. No framework lock-in.

* * *

## Limitations

* Validated on nano-GPT scale (synthetic failures) and Pythia-14M (real logs, clean + anomalous run)
* Power-law baseline assumes monotonically decreasing loss
* `finetuning` preset is experimental — not validated on RLHF, LoRA, or catastrophic forgetting
* External validation on independent architectures ongoing — one real training log from your pipeline is worth more than ten synthetic benchmarks

Full details in [the paper](https://doi.org/10.5281/zenodo.21842460).

* * *

## Did it work for you?

If you ran W-Twin on a real training log — whether it detected something or missed it — please open an [issue](https://github.com/Kretski/WTwin/issues). Real data from real runs is the primary path to validation.

* * *

## Open source & commercial

The core of W-Twin is MIT-licensed — free to use, modify, and integrate.

**Included:**

* `WTwinMonitor` core
* CLI tools
* Optimizer wrappers (`WTwinAdamW`, `WTwinSGD`, `WTwinLion`)
* `ExpFloorBaseline` for fine-tuning
* Browser demo
* All integrations

**If you are using W-Twin in a production training pipeline** and need help with integration, calibration for your specific architecture, or higher reliability guarantees — reach out: **kretski1@gmail.com**

* * *

## Citation

    @software{kretski2026wtwin,
      author    = {Kretski, Dimitar},
      title     = {W-Twin: Forecast-Based Detection of Progressive Neural Network Training Degradation},
      year      = {2026},
      doi       = {10.5281/zenodo.21842460},
      url       = {https://zenodo.org/records/21865734}
    }

**Author:** Dimitar KretskiCenter for Hydro- and Aerodynamics, Varna, BulgariaORCID: [0000-0001-5108-2243](https://orcid.org/0000-0001-5108-2243)
