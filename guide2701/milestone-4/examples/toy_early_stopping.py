"""Diagnostic implementation of Early Stopping and Checkpoint Restoration.

Demonstrates:
1. Directional tracking (mode='max' for Dice vs mode='min' for Loss).
2. Distinction between stochastic validation noise and genuine improvement (min_delta).
3. Patience window management.
4. Automatic restoration of best checkpoint weights upon termination.
"""
from __future__ import annotations

import copy
import torch
import torch.nn as nn


class EarlyStopping:
    """Monitors validation metrics, tracks patience, and saves best model state."""

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.002,
        mode: str = "max",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.stale_epochs = 0
        self.best_score: float | None = None
        self.best_epoch: int = 0
        self.best_state: dict | None = None
        self.should_stop: bool = False

    def step(self, current_score: float, epoch: int, model: nn.Module) -> bool:
        """Evaluate current epoch metric and update early stopping state.

        Returns True if a new best checkpoint was recorded.
        """
        if self.best_score is None:
            self.best_score = current_score
            self.best_epoch = epoch
            self.best_state = copy.deepcopy(model.state_dict())
            self.stale_epochs = 0
            return True

        # Check if score improved beyond min_delta
        if self.mode == "max":
            improved = current_score > (self.best_score + self.min_delta)
        else:
            improved = current_score < (self.best_score - self.min_delta)

        if improved:
            self.best_score = current_score
            self.best_epoch = epoch
            self.best_state = copy.deepcopy(model.state_dict())
            self.stale_epochs = 0
            return True
        else:
            self.stale_epochs += 1
            if self.stale_epochs >= self.patience:
                self.should_stop = True
            return False

    def restore_best_checkpoint(self, model: nn.Module):
        """Restore model weights to the best recorded epoch."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
            print(f"-> Successfully restored best checkpoint from Epoch {self.best_epoch} (Score: {self.best_score:.4f})")


def main():
    print("=" * 75)
    print("Milestone 4: Early Stopping & Best Checkpoint Restoration Diagnostic")
    print("=" * 75)

    # Toy linear model representing segmentation head
    torch.manual_seed(42)
    model = nn.Linear(10, 2)

    # Instantiate early stopping (monitoring Foreground Dice, mode='max')
    patience = 4
    min_delta = 0.005
    stopper = EarlyStopping(patience=patience, min_delta=min_delta, mode="max")

    # Fictional validation Dice trajectory across 20 epochs
    # Peaks at epoch 8 (0.642), oscillates, and degrades due to overfitting
    val_dice_history = [
        0.310, 0.450, 0.520, 0.585, 0.612, 0.630, 0.635, 0.642,  # Epoch 8 is best (0.642)
        0.638, 0.640, 0.631, 0.625, 0.610, 0.605, 0.590, 0.580,  # Epoch 12 triggers stop
        0.570, 0.550, 0.540, 0.530,
    ]

    print(f"\nConfiguration: Patience = {patience} epochs, Min Delta = {min_delta:.3f}, Mode = 'max'\n")
    print(f"{'Epoch':<6} | {'Val Dice':<10} | {'Status':<30} | {'Stale Epochs':<12}")
    print("-" * 65)

    for ep, score in enumerate(val_dice_history, 1):
        # Slightly alter dummy model weight so we can verify restoration
        with torch.no_grad():
            model.weight.add_(0.01)

        is_new_best = stopper.step(score, epoch=ep, model=model)

        if is_new_best:
            status = f"★ NEW BEST CHECKPOINT ({score:.4f})"
        else:
            status = f"No improvement (best: {stopper.best_score:.4f})"

        print(f"{ep:>5}  | {score:>8.4f}   | {status:<30} | {stopper.stale_epochs:>6} / {patience}")

        if stopper.should_stop:
            print(f"\n[!] Early stopping triggered at Epoch {ep}: {stopper.stale_epochs} consecutive stale epochs.")
            break

    print("\nRestoration Protocol:")
    stopper.restore_best_checkpoint(model)

    print("\nKey Takeaways:")
    print("• Stale Counter: Increments only when improvement is LESS than min_delta.")
    print("  Epoch 7 (0.635 -> 0.642, +0.007 > 0.005) reset the counter to 0.")
    print("  Epoch 9-12 had no improvement > 0.005, exhausting patience (4 epochs).")
    print("• Best Checkpoint Preservation: Checkpoint at Epoch 8 was restored,")
    print("  preventing deployment of overfitted Epoch 12 weights.")
    print("=" * 75)


if __name__ == "__main__":
    main()
