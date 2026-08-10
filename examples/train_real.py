"""
examples/train_real.py
======================
Nano-GPT training run with W-Twin monitoring and controlled failure injection.

Designed for CPU-only training (Ryzen 7 PRO, 16GB RAM).
Estimated runtime: 1.5–3 hours depending on config.

Logged per step (CSV) — W-Twin AND two baseline detectors (Threshold, CUSUM):
    step, train_loss, val_loss, lr, grad_norm, wall_time,
    wtwin_D, wtwin_Q, wtwin_T, wtwin_W, wtwin_alert,
    thresh_alert, cusum_alert, failure_active

Failure injection modes:
    "none"      — clean run (baseline / FAR validation)
    "lr_spike"  — sudden LR increase at failure_step (simulates bad schedule)
    "lr_drop"   — sudden LR collapse (training stall / drift)
    "dropout"   — sudden high dropout (misconfiguration)

Pre-registered detection criterion (fixed before any run):
    TRUE DETECTION = first alert in [failure_step, failure_step + 500]
    FALSE POSITIVE = alert before failure_step, or alert on clean run
    MISSED         = no alert in [failure_step, failure_step + 500]

Usage:
    # Smoke test (~5 min)
    python examples/train_real.py --mode lr_spike --failure-step 200 --steps 300 --model-size tiny

    # Clean run
    python examples/train_real.py --mode none --steps 3000 --seed 42

    # Failure run, seed 42
    python examples/train_real.py --mode lr_spike --failure-step 2000 --steps 3000 --seed 42

    # Repeat with different seed (independent replication)
    python examples/train_real.py --mode lr_spike --failure-step 2000 --steps 3000 --seed 43
"""

import argparse
import csv
import math
import sys
import time
from collections import deque
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from scalepredict.monitor import WTwinMonitor
from scalepredict.monitor.benchmark import ThresholdDetector, CUSUMDetector

# ── Pre-registered detection criterion ───────────────────────────────────────
# Fixed BEFORE any run. Do not change after seeing results.
DETECTION_WINDOW = 500   # steps after failure_step that count as true detection


# ── Model definition (nano-GPT) ───────────────────────────────────────────────

@dataclass
class GPTConfig:
    vocab_size: int = 256       # byte-level
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size))
            .view(1, 1, cfg.block_size, cfg.block_size),
        )

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class NanoGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe=nn.Embedding(cfg.block_size, cfg.n_embd),
            drop=nn.Dropout(cfg.dropout),
            h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f=nn.LayerNorm(cfg.n_embd),
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long)
        x = self.transformer.drop(
            self.transformer.wte(idx) + self.transformer.wpe(pos)
        )
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss


# ── Dataset ───────────────────────────────────────────────────────────────────

class ByteDataset(Dataset):
    """Byte-level dataset from a text file. Generates overlapping windows."""

    def __init__(self, text: str, block_size: int):
        self.data = torch.tensor(
            [ord(c) % 256 for c in text], dtype=torch.long
        )
        self.block_size = block_size

    def __len__(self):
        return max(1, len(self.data) - self.block_size)

    def __getitem__(self, i):
        x = self.data[i: i + self.block_size]
        y = self.data[i + 1: i + self.block_size + 1]
        return x, y


def get_tiny_shakespeare() -> str:
    """Download or load Tiny Shakespeare."""
    cache = Path("/tmp/tinyshakespeare.txt")
    if cache.exists():
        return cache.read_text()
    try:
        import urllib.request
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, cache)
        return cache.read_text()
    except Exception:
        # Fallback: generate synthetic text
        print("  [Warning] Could not download Tiny Shakespeare. Using synthetic text.")
        import random, string
        rng = random.Random(42)
        words = ["the", "and", "of", "to", "a", "in", "that", "he", "was", "for",
                 "on", "are", "with", "as", "his", "they", "be", "at", "one", "have"]
        return " ".join(rng.choice(words) for _ in range(50000))


# ── Failure injection ─────────────────────────────────────────────────────────

def _ramp_progress(step: int, failure_step: int, ramp_steps: int,
                   profile: str) -> float:
    """
    Compute ramp progress ∈ [0, 1] at current step.

    profile: "linear" | "quadratic" | "sigmoid"
    Returns 0.0 before failure_step, 1.0 after failure_step + ramp_steps.
    """
    if step < failure_step:
        return 0.0
    t = min((step - failure_step) / max(ramp_steps, 1), 1.0)
    if profile == "linear":
        return t
    elif profile == "quadratic":
        return t ** 2
    elif profile == "sigmoid":
        import math
        # Sigmoid centered at t=0.5
        k = 10.0
        return 1.0 / (1.0 + math.exp(-k * (t - 0.5)))
    return t  # fallback: linear


class FailureInjector:
    """
    Injects controlled failures — both abrupt (single-step) and
    progressive (ramp over many steps).

    Abrupt modes (fire once at failure_step):
        lr_spike, lr_drop, dropout, weight_corrupt, label_corrupt

    Progressive modes (ramp from failure_step over ramp_steps):
        progressive_label  — linearly growing fraction of random labels
        progressive_weight — linearly growing weight noise σ

    Parameters
    ----------
    mode         : failure mode name
    failure_step : step at which failure begins
    base_lr      : base learning rate (for lr modes)
    ramp_steps   : steps over which progressive failure ramps to max
    ramp_profile : "linear" | "quadratic" | "sigmoid"
    max_noise_prob  : max label corruption fraction (progressive_label)
    max_weight_sigma: max weight noise σ (progressive_weight)
    """

    def __init__(self, mode: str, failure_step: int, base_lr: float,
                 ramp_steps: int = 1000, ramp_profile: str = "linear",
                 max_noise_prob: float = 0.50,
                 max_weight_sigma: float = 0.02):
        self.mode = mode
        self.failure_step = failure_step
        self.base_lr = base_lr
        self.ramp_steps = ramp_steps
        self.ramp_profile = ramp_profile
        self.max_noise_prob = max_noise_prob
        self.max_weight_sigma = max_weight_sigma
        self.active = False
        self._announced = False

    def progress(self, step: int) -> float:
        """Current ramp progress [0, 1]."""
        return _ramp_progress(step, self.failure_step, self.ramp_steps,
                              self.ramp_profile)

    def apply(self, step: int, optimizer, model: NanoGPT):
        """
        Apply failure effect for current step.
        Returns announcement string on first activation, None otherwise.
        For progressive modes: applies effect every step after failure_step.
        """
        if step < self.failure_step:
            return None

        # Activate on first step at or after failure_step
        announcement = None
        if not self.active:
            self.active = True

        # ── Abrupt modes — fire once ───────────────────────────────────────
        if self.mode == "lr_spike" and not self._announced:
            for pg in optimizer.param_groups:
                pg["lr"] = self.base_lr * 10.0
            announcement = f"[FAILURE] LR spike: {self.base_lr:.2e} → {self.base_lr*10:.2e}"

        elif self.mode == "lr_drop" and not self._announced:
            for pg in optimizer.param_groups:
                pg["lr"] = self.base_lr * 0.001
            announcement = f"[FAILURE] LR drop: {self.base_lr:.2e} → {self.base_lr*0.001:.2e}"

        elif self.mode == "marginal_lr" and not self._announced:
            # 3x above optimal — realistic misconfiguration.
            # Loss degrades slowly; borderline case for detection.
            new_lr = self.base_lr * 3.0
            for pg in optimizer.param_groups:
                pg["lr"] = new_lr
            announcement = f"[FAILURE] Marginal LR: {self.base_lr:.2e} → {new_lr:.2e} (3x, slow degradation)"

        elif self.mode == "divergent_lr" and not self._announced:
            # 20x above optimal — training diverges within ~50-100 steps.
            # Tests how early W-Twin fires relative to visible divergence.
            new_lr = self.base_lr * 20.0
            for pg in optimizer.param_groups:
                pg["lr"] = new_lr
            announcement = f"[FAILURE] Divergent LR: {self.base_lr:.2e} → {new_lr:.2e} (20x, fast divergence)"

        elif self.mode == "dropout" and not self._announced:
            for block in model.transformer.h:
                block.attn.dropout.p = 0.5
                block.mlp.dropout.p = 0.5
            announcement = "[FAILURE] Dropout: 0.0 → 0.5"

        elif self.mode == "weight_corrupt" and not self._announced:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if "weight" in name:
                        mask = torch.rand_like(param) < 0.20
                        param[mask] += torch.randn_like(param[mask]) * 5.0
            announcement = "[FAILURE] Weight corruption: 20% weights + N(0,5)"

        elif self.mode == "label_corrupt" and not self._announced:
            announcement = "[FAILURE] Label corruption: 100% random labels"

        # ── Progressive modes — fire every step ───────────────────────────
        elif self.mode == "progressive_weight":
            p = self.progress(step)
            sigma = self.max_weight_sigma * p
            if sigma > 1e-6:
                with torch.no_grad():
                    for param in model.parameters():
                        param.add_(torch.randn_like(param) * sigma)
            if not self._announced:
                announcement = (
                    f"[FAILURE] Progressive weight noise — "
                    f"{self.ramp_profile} ramp over {self.ramp_steps} steps, "
                    f"max σ={self.max_weight_sigma}"
                )

        # progressive_label is handled in the training loop (needs y tensor)

        if not self._announced and announcement:
            self._announced = True

        return announcement

    def corrupt_labels(self, step: int, y: torch.Tensor,
                       vocab_size: int) -> torch.Tensor:
        """
        Return (possibly corrupted) label tensor.
        Called every step from the training loop.
        """
        if not self.active:
            return y

        if self.mode == "label_corrupt":
            # Abrupt: 100% random labels
            return torch.randint(0, vocab_size, y.shape)

        elif self.mode == "progressive_label":
            # Progressive: linearly growing fraction of random labels
            p = self.progress(step)
            noise_prob = self.max_noise_prob * p
            if noise_prob < 1e-4:
                return y
            mask = torch.rand_like(y, dtype=torch.float) < noise_prob
            random_labels = torch.randint(0, vocab_size, y.shape)
            return torch.where(mask, random_labels, y)

        return y


# ── Online baseline detectors (streaming) ────────────────────────────────────

class OnlineThreshold:
    """
    Fixed z-score threshold detector, streaming version.
    Fires when z-score > threshold for n_consec consecutive steps.
    Parameters fixed to match synthetic benchmark defaults.
    """
    def __init__(self, threshold: float = 3.0, window: int = 50, n_consec: int = 5):
        self.threshold = threshold
        self.window = window
        self.n_consec = n_consec
        self._buf: deque = deque(maxlen=window)
        self._consec = 0
        self._alert = False
        self._first_alert: int | None = None

    def update(self, step: int, loss: float) -> bool:
        self._buf.append(loss)
        if len(self._buf) < self.window:
            return False
        arr = np.array(self._buf)
        z = (loss - arr.mean()) / (arr.std() + 1e-8)
        if z > self.threshold:
            self._consec += 1
        else:
            self._consec = 0
        fired = self._consec >= self.n_consec
        if fired and self._first_alert is None:
            self._first_alert = step
        self._alert = fired
        return fired

    @property
    def first_alert_step(self) -> int | None:
        return self._first_alert


class OnlineCUSUM:
    """
    CUSUM change-point detector, streaming version.
    Parameters fixed to match synthetic benchmark defaults.
    """
    def __init__(self, k: float = 0.5, h: float = 5.0, warmup: int = 100):
        self.k = k
        self.h = h
        self.warmup = warmup
        self._buf: list[float] = []
        self._S = 0.0
        self._mu: float | None = None
        self._sigma: float | None = None
        self._first_alert: int | None = None

    def update(self, step: int, loss: float) -> bool:
        self._buf.append(loss)
        if len(self._buf) == self.warmup:
            arr = np.array(self._buf)
            self._mu = float(arr.mean())
            self._sigma = float(arr.std()) + 1e-8
        if self._mu is None:
            return False
        x = (loss - self._mu) / self._sigma
        self._S = max(0.0, self._S + x - self.k)
        fired = self._S > self.h
        if fired and self._first_alert is None:
            self._first_alert = step
        return fired

    @property
    def first_alert_step(self) -> int | None:
        return self._first_alert


# ── Gradient norm ─────────────────────────────────────────────────────────────

def compute_grad_norm(model: NanoGPT) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return math.sqrt(total)


# ── Validation loss ───────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_val_loss(model: NanoGPT, val_loader, n_batches: int = 10) -> float:
    model.eval()
    losses = []
    for i, (x, y) in enumerate(val_loader):
        if i >= n_batches:
            break
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else float("nan")


# ── Main training loop ────────────────────────────────────────────────────────

def train(args):
    # ── Reproducibility ───────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"\n{'='*60}")
    print(f"  ScalePredict Real Training Run")
    print(f"  Mode: {args.mode} | Steps: {args.steps} | Seed: {args.seed}")
    if args.mode != "none":
        print(f"  Failure at step: {args.failure_step}")
    if args.mode in ("progressive_label", "progressive_weight"):
        print(f"  Ramp: {args.ramp_profile}, {args.ramp_steps} steps")
        if args.mode == "progressive_label":
            print(f"  Max noise: {args.max_noise_prob:.0%} of labels")
        else:
            print(f"  Max σ: {args.max_weight_sigma}")
    print(f"  Detection window: [{args.failure_step}, "
          f"{args.failure_step + DETECTION_WINDOW}]  (pre-registered)")
    print(f"{'='*60}\n")

    # ── Config ────────────────────────────────────────────────────────────────
    if args.model_size == "tiny":
        cfg = GPTConfig(n_layer=2, n_head=2, n_embd=64, block_size=64)
        batch_size = 16
    elif args.model_size == "small":
        cfg = GPTConfig(n_layer=4, n_head=4, n_embd=128, block_size=128)
        batch_size = 8
    else:  # medium
        cfg = GPTConfig(n_layer=6, n_head=6, n_embd=192, block_size=128)
        batch_size = 4

    lr = 3e-4
    val_interval = 50     # evaluate val loss every N steps
    log_interval = 10     # print to console every N steps

    # ── Data ──────────────────────────────────────────────────────────────────
    print("  Loading data...")
    text = get_tiny_shakespeare()
    split = int(0.9 * len(text))
    train_ds = ByteDataset(text[:split], cfg.block_size)
    val_ds = ByteDataset(text[split:], cfg.block_size)
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            drop_last=True)
    print(f"  Train tokens: {len(train_ds):,}  |  Val tokens: {len(val_ds):,}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = NanoGPT(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}  |  Size: {args.model_size}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)

    # LR warmup scheduler
    def lr_lambda(step):
        warmup = 100
        if step < warmup:
            return step / warmup
        return max(0.1, 1.0 - (step - warmup) / (args.steps - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── W-Twin monitor ────────────────────────────────────────────────────────
    monitor = WTwinMonitor(
        warmup_steps=100,
        mad_window=50,
        threshold_window=150,
        alpha=2.0,
        tau=1e-3,
        n_consec=5,
    )

    # ── Baseline detectors (online, streaming — parameters pre-registered) ────
    thresh_det = OnlineThreshold(threshold=3.0, window=50, n_consec=5)
    cusum_det  = OnlineCUSUM(k=0.5, h=5.0, warmup=100)

    # ── Failure injector ──────────────────────────────────────────────────────
    injector = FailureInjector(
        mode=args.mode,
        failure_step=args.failure_step,
        base_lr=lr,
        ramp_steps=args.ramp_steps,
        ramp_profile=args.ramp_profile,
        max_noise_prob=args.max_noise_prob,
        max_weight_sigma=args.max_weight_sigma,
    )

    # ── Output CSV ────────────────────────────────────────────────────────────
    out_dir = Path("logs")
    out_dir.mkdir(exist_ok=True)
    run_name = f"run_{args.mode}_s{args.steps}_f{args.failure_step}_seed{args.seed}"
    csv_path = out_dir / f"{run_name}.csv"

    fieldnames = [
        "step", "train_loss", "val_loss", "lr", "grad_norm",
        "wall_time",
        "wtwin_D", "wtwin_Q", "wtwin_T", "wtwin_W", "wtwin_alert",
        "thresh_alert", "cusum_alert",
        "failure_active", "noise_prob",
    ]

    print(f"\n  Logging to: {csv_path}")
    print(f"  Starting training...\n")

    start_time = time.time()
    step = 0
    train_iter = iter(train_loader)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        model.train()

        while step < args.steps:
            # Cycle through data
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            # ── Failure injection ──────────────────────────────────────────
            msg = injector.apply(step, optimizer, model)
            if msg:
                print(f"\n  *** {msg} ***\n")

            # ── Forward / backward ─────────────────────────────────────────
            optimizer.zero_grad()
            # Label corruption (abrupt or progressive) — delegate to injector
            y = injector.corrupt_labels(step, y, cfg.vocab_size)
            _, loss = model(x, y)
            loss.backward()

            # Gradient clipping — disabled for LR-based failures so they can diverge
            clip_active = not (injector.active and
                               args.mode in ("marginal_lr", "divergent_lr"))
            if clip_active:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad_norm = compute_grad_norm(model)

            optimizer.step()
            scheduler.step()

            # Re-apply LR override after scheduler (scheduler resets lr each step)
            if injector.active and args.mode in ("marginal_lr", "divergent_lr",
                                                  "lr_spike", "lr_drop"):
                target_lr = {
                    "marginal_lr":  lr * 4.0,
                    "divergent_lr": lr * 10.0,
                    "lr_spike":     lr * 10.0,
                    "lr_drop":      lr * 0.001,
                }[args.mode]
                for pg in optimizer.param_groups:
                    pg["lr"] = target_lr

            train_loss = loss.item()

            # Guard: if loss is NaN or explodes, stop gracefully
            if not math.isfinite(train_loss) or train_loss > 1e6:
                print(f"\n  [!] Loss is {train_loss:.4f} at step {step+1} — training diverged.")
                print(f"      Stopping early.\n")
                break
            current_lr = optimizer.param_groups[0]["lr"]
            wall_time = time.time() - start_time

            # ── Validation loss (periodic) ─────────────────────────────────
            val_loss = float("nan")
            if step % val_interval == 0:
                val_loss = estimate_val_loss(model, val_loader)

            # ── Monitor updates ────────────────────────────────────────────
            state        = monitor.update(step + 1, train_loss)
            thresh_fired = thresh_det.update(step + 1, train_loss)
            cusum_fired  = cusum_det.update(step + 1, train_loss)

            # ── Log ───────────────────────────────────────────────────────
            # Compute noise_prob for logging (progressive modes only)
            noise_prob = 0.0
            if injector.active and args.mode in ("progressive_label",):
                noise_prob = injector.max_noise_prob * injector.progress(step)
            elif injector.active and args.mode in ("progressive_weight",):
                noise_prob = injector.max_weight_sigma * injector.progress(step)

            row = {
                "step": step + 1,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6) if not math.isnan(val_loss) else "",
                "lr": round(current_lr, 8),
                "grad_norm": round(grad_norm, 6),
                "wall_time": round(wall_time, 2),
                "wtwin_D": round(state.D, 4),
                "wtwin_Q": round(state.Q, 4),
                "wtwin_T": round(state.T, 4),
                "wtwin_W": round(state.W, 4),
                "wtwin_alert": int(state.alert),
                "thresh_alert": int(thresh_fired),
                "cusum_alert":  int(cusum_fired),
                "failure_active": int(injector.active),
                "noise_prob": round(noise_prob, 5),
            }
            writer.writerow(row)

            # ── Console output ─────────────────────────────────────────────
            if step % log_interval == 0:
                alert_str = " *** ALERT ***" if state.alert else ""
                val_str = f"  val={val_loss:.4f}" if not math.isnan(val_loss) else ""
                print(
                    f"  step {step+1:>5}/{args.steps}  "
                    f"loss={train_loss:.4f}{val_str}  "
                    f"lr={current_lr:.2e}  "
                    f"W={state.W:+.3f}{alert_str}"
                )

            # First alert
            if state.alert and monitor.first_alert_step() == step + 1:
                delay = (step + 1) - args.failure_step if args.mode != "none" else None
                delay_str = f"  delay from failure: {delay:+d} steps" if delay is not None else ""
                print(f"\n  *** W-TWIN ALERT at step {step+1}{delay_str} ***\n")

            step += 1

    # ── Summary with pre-registered criterion ─────────────────────────────────
    total_time = time.time() - start_time
    fs = args.failure_step
    dw = DETECTION_WINDOW
    is_failure_run = args.mode != "none"

    def classify_alert(first_alert_step: int | None) -> str:
        """Apply pre-registered criterion. Returns classification string."""
        if not is_failure_run:
            return "FALSE POSITIVE" if first_alert_step is not None else "correct (no alert)"
        if first_alert_step is None:
            return "MISSED"
        if first_alert_step < fs:
            return "FALSE POSITIVE (before failure)"
        if first_alert_step <= fs + dw:
            delay = first_alert_step - fs
            return f"TRUE DETECTION  (delay: +{delay} steps)"
        return f"MISSED (alert too late: step {first_alert_step} > {fs + dw})"

    wt_alert  = monitor.first_alert_step()
    th_alert  = thresh_det.first_alert_step
    cu_alert  = cusum_det.first_alert_step

    print(f"\n{'='*60}")
    print(f"  Training complete  |  {total_time/60:.1f} min  |  seed={args.seed}")
    print(f"  Log: {csv_path}")
    if is_failure_run:
        print(f"\n  Failure injected at step {fs}")
        print(f"  Detection window:  [{fs}, {fs + dw}]  (pre-registered)")
    print(f"\n  {'Detector':<12} {'First alert':>12}  Verdict")
    print(f"  {'-'*50}")
    for name, alert in [("W-Twin", wt_alert), ("Threshold", th_alert), ("CUSUM", cu_alert)]:
        alert_str = str(alert) if alert else "None"
        verdict = classify_alert(alert)
        print(f"  {name:<12} {alert_str:>12}  {verdict}")
    print(f"{'='*60}\n")

    return csv_path


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ScalePredict real training run")
    parser.add_argument("--mode", default="none",
                        choices=["none", "lr_spike", "lr_drop", "dropout",
                                 "weight_corrupt", "label_corrupt",
                                 "progressive_label", "progressive_weight",
                                 "marginal_lr", "divergent_lr"],
                        help="Failure injection mode")
    parser.add_argument("--failure-step", type=int, default=2000,
                        help="Step at which failure begins")
    parser.add_argument("--steps", type=int, default=3000,
                        help="Total training steps")
    parser.add_argument("--model-size", default="small",
                        choices=["tiny", "small", "medium"],
                        help="Model size")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    # Progressive failure parameters
    parser.add_argument("--ramp-steps", type=int, default=1000,
                        help="Steps over which progressive failure ramps to max")
    parser.add_argument("--ramp-profile", default="linear",
                        choices=["linear", "quadratic", "sigmoid"],
                        help="Ramp shape for progressive failures")
    parser.add_argument("--max-noise-prob", type=float, default=0.50,
                        help="Max label corruption fraction (progressive_label)")
    parser.add_argument("--max-weight-sigma", type=float, default=0.02,
                        help="Max weight noise σ (progressive_weight)")

    args = parser.parse_args()

    if args.mode != "none" and args.failure_step >= args.steps:
        parser.error("--failure-step must be < --steps")

    train(args)
