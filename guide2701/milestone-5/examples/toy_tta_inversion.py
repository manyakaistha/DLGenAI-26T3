"""Milestone 5 Diagnostic: Test-Time Augmentation (TTA) & Spatial Inversion.

Demonstrates the forward-predict-invert-aggregate cycle for Test-Time Augmentation (TTA).
Proves mathematically and empirically why omitting the inverse transformation destroys
spatial alignment and reduces segmentation IoU to zero.
"""
from __future__ import annotations

import numpy as np


def forward_hflip(tensor: np.ndarray) -> np.ndarray:
    """Horizontal flip: flip along width axis (axis -1)."""
    return np.flip(tensor, axis=-1)


def inverse_hflip(tensor: np.ndarray) -> np.ndarray:
    """Inverse horizontal flip: flip along width axis (axis -1)."""
    return np.flip(tensor, axis=-1)


def forward_vflip(tensor: np.ndarray) -> np.ndarray:
    """Vertical flip: flip along height axis (axis -2)."""
    return np.flip(tensor, axis=-2)


def inverse_vflip(tensor: np.ndarray) -> np.ndarray:
    """Inverse vertical flip: flip along height axis (axis -2)."""
    return np.flip(tensor, axis=-2)


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute Intersection-over-Union between two binary masks."""
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def mock_model_predict(image_batch: np.ndarray) -> np.ndarray:
    """Mock model forward pass returning probabilities for defect."""
    # Defect is located at pixel value > 0.8
    # Model detects pixels where image has bright anomalous spots
    probs = np.zeros_like(image_batch, dtype=np.float32)
    probs[image_batch > 0.5] = 0.92
    probs[image_batch <= 0.5] = 0.05
    return probs


def main():
    print("=" * 70)
    print("Milestone 5: Test-Time Augmentation (TTA) Spatial Inversion")
    print("=" * 70)

    # 1. Create an asymmetric industrial strip (1, 1, 128, 800)
    # Defect placed strictly on the LEFT side: rows [20..30], cols [50..120]
    H, W = 128, 800
    gt_mask = np.zeros((H, W), dtype=np.uint8)
    gt_mask[20:30, 50:120] = 1

    img = np.zeros((1, 1, H, W), dtype=np.float32)
    img[0, 0, 20:30, 50:120] = 0.95

    print(f"Ground Truth Defect location: Rows [20..30], Columns [50..120] (Left half)")
    print(f"Ground Truth Active Pixels : {gt_mask.sum()} px")

    # 2. View 1: Standard Forward Pass (No TTA)
    prob_orig = mock_model_predict(img)[0, 0]
    mask_orig = (prob_orig >= 0.5).astype(np.uint8)
    iou_orig = compute_iou(mask_orig, gt_mask)
    print(f"\n[View 1: Original Pass] IoU with Ground Truth = {iou_orig:.4f}")

    # 3. View 2: Horizontal Flip TTA with PROPER Inverse
    img_hflip = forward_hflip(img)
    # Notice: In transformed coordinates, defect has moved to cols [800 - 120 .. 800 - 50] = [680..750]
    prob_hflip_raw = mock_model_predict(img_hflip)[0, 0]
    
    # Invert the prediction back to original coordinates!
    prob_hflip_inverted = inverse_hflip(prob_hflip_raw)
    mask_hflip_inverted = (prob_hflip_inverted >= 0.5).astype(np.uint8)
    iou_hflip_inverted = compute_iou(mask_hflip_inverted, gt_mask)
    print(f"[View 2: H-Flip with INVERSE] IoU with Ground Truth = {iou_hflip_inverted:.4f}")

    # 4. View 2 BUG: What happens if you forget inverse_hflip?
    mask_hflip_buggy = (prob_hflip_raw >= 0.5).astype(np.uint8)
    iou_buggy = compute_iou(mask_hflip_buggy, gt_mask)
    print(f"[View 2 BUG: Inversion OMITTED] IoU with Ground Truth = {iou_buggy:.4f}")
    assert iou_buggy == 0.0, f"Expected 0.0 IoU due to opposite coordinates, got {iou_buggy}"

    # 5. Ensemble Blending: Average Orig + Inverted H-Flip
    prob_tta_correct = 0.5 * prob_orig + 0.5 * prob_hflip_inverted
    mask_tta_correct = (prob_tta_correct >= 0.5).astype(np.uint8)
    iou_tta_correct = compute_iou(mask_tta_correct, gt_mask)

    # Catastrophic Blend: Average Orig + Raw Un-inverted Flip
    prob_tta_broken = 0.5 * prob_orig + 0.5 * prob_hflip_raw
    mask_tta_broken = (prob_tta_broken >= 0.5).astype(np.uint8)
    iou_tta_broken = compute_iou(mask_tta_broken, gt_mask)

    print("\n--- Summary of TTA Aggregation ---")
    print(f"Baseline (No TTA)                 : IoU = {iou_orig:.4f}")
    print(f"Correct TTA (Orig + Inverted HFlip): IoU = {iou_tta_correct:.4f} (reinforced)")
    print(f"BROKEN TTA (Orig + Uninverted Flip): IoU = {iou_tta_broken:.4f} (destroyed by ghosting!)")

    assert iou_tta_correct == 1.0
    print("\n✓ Inversion cycle mathematically proven and verified!")
    print("=" * 70)


if __name__ == "__main__":
    main()
