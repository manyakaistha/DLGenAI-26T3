"""Educational Toy Example: BCE vs Soft Dice Loss on Tiny Synthetic Masks.

Demonstrates how BCEWithLogitsLoss and Soft Dice respond to:
1. Case 1: Missed small defect region.
2. Case 2: Scattered false positive noise on clean background.
3. Case 3: Completely empty mask with subtle false alarms.
"""
import torch
import torch.nn as nn


def soft_dice_loss(probs, targets, smooth=1.0):
    """Differentiable Soft Dice Loss for educational demonstration."""
    intersection = (probs * targets).sum()
    cardinality = probs.sum() + targets.sum()
    dice = (2.0 * intersection + smooth) / (cardinality + smooth)
    return 1.0 - dice


def run_toy_loss_experiments():
    print("=== Educational Loss Function Behavior on Tiny 4x4 Canvas ===\n")
    bce_fn = nn.BCEWithLogitsLoss()

    # --- Scenario 1: Missed Small Defect ---
    # Canvas 4x4 = 16 pixels. Only 1 pixel has ground truth defect.
    target_1 = torch.zeros(1, 1, 4, 4)
    target_1[0, 0, 1, 1] = 1.0  # 1 defect pixel

    # Model predicts weak probability on that pixel (prob ~ 0.1, logit ~ -2.2)
    # and background on the rest (prob ~ 0.05, logit ~ -2.94)
    logits_missed = torch.full((1, 1, 4, 4), -2.94)
    logits_missed[0, 0, 1, 1] = -2.20
    probs_missed = torch.sigmoid(logits_missed)

    bce_val_1 = bce_fn(logits_missed, target_1).item()
    dice_val_1 = soft_dice_loss(probs_missed, target_1).item()
    print("Scenario 1: Missed Small Defect (1 px defect on 16 px canvas)")
    print(f"  • BCE Loss:       {bce_val_1:.4f}  (dominated by 15 background pixels!)")
    print(f"  • Soft Dice Loss: {dice_val_1:.4f}  (heavily penalizes low overlap ratio)")
    print()

    # --- Scenario 2: False Positive Noise ---
    # Ground truth has 4 defect pixels. Model correctly predicts all 4,
    # but ALSO predicts 4 false positive pixels on clean background.
    target_2 = torch.zeros(1, 1, 4, 4)
    target_2[0, 0, :2, :2] = 1.0  # 4 defect pixels

    logits_fp = torch.full((1, 1, 4, 4), -4.0)
    logits_fp[0, 0, :2, :2] = 3.0    # 4 true positive pixels (high confidence)
    logits_fp[0, 0, 2:, 2:] = 3.0    # 4 false positive pixels (high confidence)
    probs_fp = torch.sigmoid(logits_fp)

    bce_val_2 = bce_fn(logits_fp, target_2).item()
    dice_val_2 = soft_dice_loss(probs_fp, target_2).item()
    print("Scenario 2: Scattered False Positive Noise (4 TP + 4 FP)")
    print(f"  • BCE Loss:       {bce_val_2:.4f}")
    print(f"  • Soft Dice Loss: {dice_val_2:.4f}")
    print()

    # --- Scenario 3: All-Empty Mask (Clean Strip) ---
    # Ground truth is completely clean (target = 0 everywhere).
    target_3 = torch.zeros(1, 1, 4, 4)
    # Model predicts slight background noise (e.g. logit -2.0 -> prob 0.119)
    logits_empty = torch.full((1, 1, 4, 4), -2.0)
    probs_empty = torch.sigmoid(logits_empty)

    bce_val_3 = bce_fn(logits_empty, target_3).item()
    dice_val_3 = soft_dice_loss(probs_empty, target_3, smooth=1.0).item()
    print("Scenario 3: All-Empty Mask with Subtle Noise (prob ~ 0.12 across canvas)")
    print(f"  • BCE Loss:       {bce_val_3:.4f}")
    print(f"  • Soft Dice Loss: {dice_val_3:.4f}")
    print(f"    (Soft Dice acts as L1-penalty: sum(P) / (sum(P) + eps) = {probs_empty.sum().item():.2f} / ({probs_empty.sum().item():.2f} + 1))")


if __name__ == "__main__":
    run_toy_loss_experiments()
