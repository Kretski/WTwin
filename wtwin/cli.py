"""
wtwin.cli
================
Command-line interface for W-Twin monitor.

Usage:
    wtwin monitor training_log.csv
    wtwin monitor training_log.csv --loss-col train_loss --step-col step
    wtwin monitor training_log.csv --alpha 2.0 --warmup 100
    wtwin demo
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def cmd_monitor(args):
    """Run W-Twin on a CSV training log."""
    import numpy as np
    from wtwin.monitor import WTwinMonitor

    path = Path(args.file)
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    # ── Read CSV ──────────────────────────────────────────────────────────────
    steps, losses = [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if args.step_col not in reader.fieldnames:
            print(f"Error: column '{args.step_col}' not found. "
                  f"Available: {reader.fieldnames}", file=sys.stderr)
            sys.exit(1)
        if args.loss_col not in reader.fieldnames:
            print(f"Error: column '{args.loss_col}' not found. "
                  f"Available: {reader.fieldnames}", file=sys.stderr)
            sys.exit(1)

        for row in reader:
            try:
                s = float(row[args.step_col])
                l = float(row[args.loss_col])
                if not (s != s or l != l):  # skip NaN
                    steps.append(int(s))
                    losses.append(l)
            except (ValueError, KeyError):
                continue

    if len(steps) < 10:
        print(f"Error: only {len(steps)} valid rows found. Need at least 10.",
              file=sys.stderr)
        sys.exit(1)

    print(f"\n  W-Twin Monitor")
    print(f"  File:      {path.name}")
    print(f"  Steps:     {steps[0]} → {steps[-1]}  ({len(steps)} rows)")
    print(f"  Loss:      {losses[0]:.4f} → {losses[-1]:.4f}")
    print(f"  Params:    alpha={args.alpha}, warmup={args.warmup}, "
          f"n_consec={args.n_consec}")
    print()

    # ── Run monitor ───────────────────────────────────────────────────────────
    monitor = WTwinMonitor(
        warmup_steps=args.warmup,
        alpha=args.alpha,
        n_consec=args.n_consec,
        mad_window=args.mad_window,
        tau=args.tau,
    )

    alert_steps = []
    print(f"  {'Step':>8}  {'Loss':>10}  {'W':>10}  Status")
    print(f"  {'-'*48}")

    report_every = max(1, len(steps) // 20)  # ~20 lines output

    for i, (s, l) in enumerate(zip(steps, losses)):
        state = monitor.update(s, l)

        if state.alert and s not in alert_steps:
            alert_steps.append(s)

        if i % report_every == 0 or state.alert:
            status = "*** ALERT ***" if state.alert else ""
            print(f"  {s:>8d}  {l:>10.4f}  {state.W:>+10.3f}  {status}")

    print()
    first = monitor.first_alert_step()
    if first:
        print(f"  First alert: step {first}")
    else:
        print(f"  No alert fired.")

    if args.output:
        out_path = Path(args.output)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "loss", "wtwin_D", "wtwin_Q",
                             "wtwin_T", "wtwin_W", "wtwin_alert"])
            for state in monitor.history:
                writer.writerow([state.step, state.l_obs, round(state.D, 4),
                                 round(state.Q, 4), round(state.T, 4),
                                 round(state.W, 4), int(state.alert)])
        print(f"  W-Twin scores saved: {out_path}")
    print()


def cmd_suggest(args):
    """Run W-Twin + suggest() on a CSV training log."""
    import json
    import math
    import numpy as np
    from wtwin.monitor import WTwinMonitor
    from wtwin.monitor.suggest import suggest
    from pathlib import Path
    import csv

    path = Path(args.file)
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    # Read CSV
    steps, losses = [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                s = float(row[args.step_col])
                l = float(row[args.loss_col])
                if math.isfinite(s) and math.isfinite(l):
                    steps.append(int(s))
                    losses.append(l)
            except (ValueError, KeyError):
                continue

    if len(steps) < 10:
        print(f"Error: only {len(steps)} valid rows.", file=sys.stderr)
        sys.exit(1)

    print(f"\n  W-Twin Suggest")
    print(f"  File: {path.name}  ({len(steps)} steps)")
    print()

    # Run monitor
    monitor = WTwinMonitor(
        warmup_steps=args.warmup,
        alpha=args.alpha,
        n_consec=args.n_consec,
    )
    for s, l in zip(steps, losses):
        monitor.update(s, l)

    # Get suggestion
    s = suggest(monitor)
    print(s)

    if args.json:
        print()
        print(json.dumps(s.as_dict(), indent=2))


def cmd_demo(args):
    """Run W-Twin on a synthetic demo (no file needed)."""
    import numpy as np
    from wtwin.monitor import WTwinMonitor
    from wtwin.monitor.benchmark import (
        generate_failure_run, generate_clean_run
    )

    print("\n  W-Twin — Quick Demo")
    print("  =================================")
    print("  Generating synthetic training run with progressive label drift...")
    print()

    rng = np.random.default_rng(42)
    steps, losses, true_fail = generate_failure_run(
        total_steps=1000, failure_step=600,
        failure_type="drift", rng=rng
    )

    monitor = WTwinMonitor(warmup_steps=50, alpha=2.0, n_consec=5)
    for s, l in zip(steps, losses):
        monitor.update(int(s), float(l))

    alert = monitor.first_alert_step()

    print(f"  True failure injected at: step 600")
    if alert:
        print(f"  W-Twin first alert:       step {alert}  "
              f"(delay: {alert - 600:+d} steps)")
    else:
        print(f"  W-Twin alert: None")

    print()
    print("  To monitor your own training log:")
    print("    wtwin monitor your_log.csv")
    print()


def main():
    parser = argparse.ArgumentParser(
        prog="wtwin",
        description="W-Twin — forecast-based training degradation monitor",
    )
    parser.add_argument("--version", action="version", version="1.2.0")

    sub = parser.add_subparsers(dest="command", required=True)

    # ── monitor subcommand ────────────────────────────────────────────────────
    p_mon = sub.add_parser("monitor",
                           help="Run W-Twin on a CSV training log")
    p_mon.add_argument("file",
                       help="Path to CSV training log")
    p_mon.add_argument("--loss-col", default="train_loss",
                       help="Column name for training loss (default: train_loss)")
    p_mon.add_argument("--step-col", default="step",
                       help="Column name for step (default: step)")
    p_mon.add_argument("--alpha", type=float, default=2.0,
                       help="Detection threshold z-score (default: 2.0)")
    p_mon.add_argument("--warmup", type=int, default=100,
                       help="Warmup steps to skip (default: 100)")
    p_mon.add_argument("--n-consec", type=int, default=5,
                       help="Consecutive steps above threshold for alert (default: 5)")
    p_mon.add_argument("--mad-window", type=int, default=50,
                       help="MAD window size (default: 50)")
    p_mon.add_argument("--tau", type=float, default=1e-3,
                       help="Q confidence decay scale (default: 1e-3)")
    p_mon.add_argument("--output", "-o",
                       help="Save W-Twin scores to CSV file")
    p_mon.set_defaults(func=cmd_monitor)

    # ── suggest subcommand ────────────────────────────────────────────────────
    p_sug = sub.add_parser("suggest",
                           help="Run W-Twin and return failure classification + suggestion")
    p_sug.add_argument("file", help="Path to CSV training log")
    p_sug.add_argument("--loss-col", default="train_loss")
    p_sug.add_argument("--step-col", default="step")
    p_sug.add_argument("--alpha", type=float, default=2.0)
    p_sug.add_argument("--warmup", type=int, default=100)
    p_sug.add_argument("--n-consec", type=int, default=5)
    p_sug.add_argument("--json", action="store_true",
                       help="Also print full JSON output")
    p_sug.set_defaults(func=cmd_suggest)

    # ── demo subcommand ───────────────────────────────────────────────────────
    p_demo = sub.add_parser("demo",
                            help="Run a quick synthetic demo")
    p_demo.set_defaults(func=cmd_demo)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
