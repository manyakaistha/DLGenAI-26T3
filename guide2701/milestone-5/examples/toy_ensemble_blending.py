"""Milestone 5 Diagnostic: Ensemble Blending vs. Binary Mask Voting.

Demonstrates the mathematical and practical difference between averaging
predicted probability maps (soft ensembling) versus voting across thresholded
binary masks (hard voting), focusing on borderline ambiguous pixels.
"""
from __future__ import annotations

import numpy as np


def main():
    print("=" * 70)
    print("Milestone 5: Model Ensembling — Soft Blending vs. Hard Voting")
    print("=" * 70)

    # Consider 5 representative pixels along a defect boundary
    # Pixel 1: Definite Background
    # Pixel 2: Borderline Low Confidence
    # Pixel 3: Disagreement Pixel (Model A says negative, Model B says positive)
    # Pixel 4: Borderline High Confidence
    # Pixel 5: Definite Defect
    pixel_names = [
        "Pixel 1 (Clear BG)",
        "Pixel 2 (Low Border)",
        "Pixel 3 (Disagreement)",
        "Pixel 4 (High Border)",
        "Pixel 5 (Clear Defect)",
    ]

    # Model A predictions (e.g. U-Net with ResNet34 backbone)
    prob_model_a = np.array([0.05, 0.35, 0.46, 0.68, 0.94], dtype=np.float32)
    # Model B predictions (e.g. FPN with EfficientNet backbone)
    prob_model_b = np.array([0.08, 0.42, 0.62, 0.74, 0.96], dtype=np.float32)

    tau = 0.50
    print(f"Decision Threshold: tau = {tau:.2f}\n")

    # 1. Soft Blending (Probability Averaging)
    # P_blend = 0.5 * P_A + 0.5 * P_B
    prob_blend = 0.5 * prob_model_a + 0.5 * prob_model_b
    mask_blend = (prob_blend >= tau).astype(np.uint8)

    # 2. Hard Voting (Binary Thresholding then Voting)
    mask_a = (prob_model_a >= tau).astype(np.uint8)
    mask_b = (prob_model_b >= tau).astype(np.uint8)

    # Hard Voting: Unanimous (AND) vs Majority / At least one (OR)
    mask_hard_and = (mask_a & mask_b).astype(np.uint8)
    mask_hard_or = (mask_a | mask_b).astype(np.uint8)

    print(f"{'Pixel Description':<25} | {'P(Model A)':<10} | {'P(Model B)':<10} | {'P(Soft Blend)':<13} | {'Soft Mask':<9} | {'Hard AND':<8} | {'Hard OR':<8}")
    print("-" * 95)
    for i, name in enumerate(pixel_names):
        print(f"{name:<25} | {prob_model_a[i]:<10.2f} | {prob_model_b[i]:<10.2f} | {prob_blend[i]:<13.2f} | {mask_blend[i]:<9} | {mask_hard_and[i]:<8} | {mask_hard_or[i]:<8}")
    print("-" * 95)

    print("\n[Key Observation at Pixel 3: Disagreement]")
    print(f"  • Model A predicts P = {prob_model_a[2]:.2f} (below threshold 0.50 -> Mask = 0)")
    print(f"  • Model B predicts P = {prob_model_b[2]:.2f} (above threshold 0.50 -> Mask = 1)")
    print(f"  • Soft Average: P_blend = (0.46 + 0.62)/2 = {prob_blend[2]:.2f} -> Mask = {mask_blend[2]}")
    print(f"  • Hard AND Vote: 0 & 1 = {mask_hard_and[2]} (pessimistic abstention)")
    print(f"  • Hard OR  Vote: 0 | 1 = {mask_hard_or[2]} (optimistic detection)")
    print("\n[Takeaway]")
    print("Soft probability blending preserves continuous model confidence and allows")
    print("fine-grained post-processing threshold tuning across the entire ensemble distribution.")
    print("Hard voting prematurely collapses floating-point information before thresholding.")
    print("=" * 70)


if __name__ == "__main__":
    main()
