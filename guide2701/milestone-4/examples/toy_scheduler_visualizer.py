"""Diagnostic visualizer and simulator for PyTorch learning rate schedulers.

Demonstrates:
1. Formulation of linear warmup followed by cosine annealing (LambdaLR).
2. Step-decay scheduler (StepLR).
3. Validation-driven plateau decay (ReduceLROnPlateau).
4. Stepping cadence: per-batch vs per-epoch stepping.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, StepLR, ReduceLROnPlateau


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup_epochs: int = 3,
    min_lr_ratio: float = 0.01,
) -> LambdaLR:
    """Standard AdamW LambdaLR scheduler: linear warmup + cosine decay."""
    def factor(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = min(1.0, (epoch - warmup_epochs) / max(1, epochs - warmup_epochs - 1))
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, factor)


def main():
    print("=" * 75)
    print("Milestone 4: Learning Rate Scheduler Mechanics & Simulation")
    print("=" * 75)

    initial_lr = 1e-3
    epochs = 20

    # Create dummy parameters
    def make_param():
        return [nn.Parameter(torch.randn(10, 10))]

    opt_cosine = AdamW(make_param(), lr=initial_lr)
    sched_cosine = build_warmup_cosine_scheduler(opt_cosine, epochs=epochs, warmup_epochs=3, min_lr_ratio=0.01)

    opt_step = AdamW(make_param(), lr=initial_lr)
    sched_step = StepLR(opt_step, step_size=6, gamma=0.2)

    opt_plateau = AdamW(make_param(), lr=initial_lr)
    sched_plateau = ReduceLROnPlateau(opt_plateau, mode="max", factor=0.5, patience=2, min_lr=1e-5)

    # Simulated validation metric for ReduceLROnPlateau (peaks then plateaus)
    val_dice_history = [
        0.20, 0.35, 0.45, 0.52, 0.55, 0.56, 0.56, 0.56,  # plateau triggers drop at epoch 8
        0.58, 0.59, 0.59, 0.59, 0.59, 0.60, 0.60, 0.60,
        0.60, 0.60, 0.60, 0.60,
    ]

    print(f"\nSimulating a {epochs}-Epoch Training Run (Initial LR = {initial_lr:.1e}):\n")
    print(f"{'Epoch':<6} | {'Warmup+Cosine LR':<18} | {'StepLR (gamma=0.2)':<20} | {'Plateau (mode=max)':<20}")
    print("-" * 72)

    for epoch in range(epochs):
        lr_c = opt_cosine.param_groups[0]["lr"]
        lr_s = opt_step.param_groups[0]["lr"]
        lr_p = opt_plateau.param_groups[0]["lr"]

        print(f"{epoch + 1:>5}  | {lr_c:>15.2e}   | {lr_s:>17.2e}    | {lr_p:>17.2e}")

        # In PyTorch, optimizer.step() must occur before scheduler.step()
        opt_cosine.step()
        opt_step.step()
        opt_plateau.step()

        # Step schedulers
        sched_cosine.step()
        sched_step.step()
        sched_plateau.step(val_dice_history[epoch])

    print("\nKey Takeaways:")
    print("• Warmup Phase (Epochs 1-3): LR ramps up linearly (3.3e-4 -> 6.7e-4 -> 1.0e-3).")
    print("  Protects randomly initialized decoder weights from gradient shattering.")
    print("• Cosine Decay: Smooth non-linear decay down to min_lr_ratio (1.0e-5).")
    print("• StepLR: Abrupt discontinuous drops at fixed intervals (epoch 6, 12, 18).")
    print("• ReduceLROnPlateau: Reacts dynamically when the validation metric stagnates.")
    print("=" * 75)


if __name__ == "__main__":
    main()
