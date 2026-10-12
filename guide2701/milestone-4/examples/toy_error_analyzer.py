"""Diagnostic framework for validation error analysis and spatial defect audits.

Demonstrates:
1. Decomposing aggregate metrics into presence detection vs. spatial boundary overlap.
2. Defect size binning (small, medium, large) to detect resolution or skip-connection failure.
3. False positive rate on clean images vs. recall on positive images.
4. Conditional Dice on detected pairs to isolate boundary quality from detection capability.
"""
from __future__ import annotations

import cv2
import numpy as np


CLASSES = ["defect_a", "defect_b", "defect_c", "defect_d"]


def compute_pair_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    """Calculate detailed detection, overlap, and area metrics for one image-class pair."""
    p = pred_mask.astype(bool)
    y = gt_mask.astype(bool)

    pred_area = int(p.sum())
    gt_area = int(y.sum())
    tp = int((p & y).sum())

    is_positive = gt_area > 0
    has_prediction = pred_area > 0
    has_overlap = tp > 0

    if pred_area + gt_area == 0:
        dice = 1.0  # true negative clean pair convention
        iou = 1.0
    elif is_positive and has_prediction:
        dice = (2.0 * tp) / (pred_area + gt_area)
        iou = float(tp) / (pred_area + gt_area - tp)
    else:
        dice = 0.0  # complete miss or false alarm
        iou = 0.0

    return {
        "pred_area": pred_area,
        "gt_area": gt_area,
        "tp": tp,
        "is_positive": is_positive,
        "has_prediction": has_prediction,
        "has_overlap": has_overlap,
        "dice": dice,
        "iou": iou,
    }


def audit_validation_errors(pairs: list[dict]) -> dict:
    """Aggregate pair-level diagnostics across validation samples."""
    total_pairs = len(pairs)
    positive_pairs = [p for p in pairs if p["is_positive"]]
    clean_pairs = [p for p in pairs if not p["is_positive"]]

    n_pos = len(positive_pairs)
    n_clean = len(clean_pairs)

    # 1. Detection vs. Overlap Breakdown
    completely_missed = sum(1 for p in positive_pairs if not p["has_prediction"])
    pred_no_overlap = sum(1 for p in positive_pairs if p["has_prediction"] and not p["has_overlap"])
    detected_with_overlap = sum(1 for p in positive_pairs if p["has_overlap"])

    presence_recall = sum(1 for p in positive_pairs if p["has_prediction"]) / max(1, n_pos)
    overlap_recall = detected_with_overlap / max(1, n_pos)

    # 2. Dice conditioned on actual defect contact
    dice_detected = [p["dice"] for p in positive_pairs if p["has_overlap"]]
    conditional_dice = float(np.mean(dice_detected)) if dice_detected else 0.0

    # 3. Clean pair false alarm rate (Specificity)
    false_alarms = sum(1 for p in clean_pairs if p["has_prediction"])
    specificity = 1.0 - (false_alarms / max(1, n_clean))

    # 4. Size binning on GT positives
    size_bins = {
        "Small (1..100 px)": [p for p in positive_pairs if 0 < p["gt_area"] <= 100],
        "Medium (101..500 px)": [p for p in positive_pairs if 100 < p["gt_area"] <= 500],
        "Large (>500 px)": [p for p in positive_pairs if p["gt_area"] > 500],
    }
    binned_stats = {}
    for label, group in size_bins.items():
        if group:
            binned_stats[label] = {
                "count": len(group),
                "overlap_recall": sum(1 for p in group if p["has_overlap"]) / len(group),
                "mean_dice": float(np.mean([p["dice"] for p in group])),
            }
        else:
            binned_stats[label] = {"count": 0, "overlap_recall": 0.0, "mean_dice": 0.0}

    return {
        "total_pairs": total_pairs,
        "positive_pairs": n_pos,
        "clean_pairs": n_clean,
        "presence_recall": presence_recall,
        "overlap_recall": overlap_recall,
        "completely_missed": completely_missed,
        "pred_no_overlap": pred_no_overlap,
        "conditional_dice": conditional_dice,
        "specificity": specificity,
        "false_alarms": false_alarms,
        "size_bins": binned_stats,
    }


def main():
    print("=" * 75)
    print("Milestone 4: Validation Error Analysis & Spatial Defect Diagnostics")
    print("=" * 75)

    # Simulate 50 validation strip pairs with varied failure modes
    rng = np.random.default_rng(42)
    pairs = []

    for i in range(50):
        # 128x800 canvas simulation
        gt = np.zeros((128, 800), dtype=np.uint8)
        pred = np.zeros((128, 800), dtype=np.uint8)

        case = i % 5
        if case == 0:
            # Clean strip, correct prediction
            pass
        elif case == 1:
            # Clean strip, false alarm (spurious speckle 30 px)
            pred[50:56, 300:305] = 1
        elif case == 2:
            # Small defect (50 px), completely missed by model
            gt[40:42, 200:225] = 1
        elif case == 3:
            # Medium defect (250 px), detected with shifted boundary
            gt[60:65, 400:450] = 1
            pred[58:67, 395:455] = 1  # dilated / shifted
        elif case == 4:
            # Large defect (800 px), good segmentation
            gt[30:70, 500:520] = 1
            pred[31:69, 501:519] = 1

        pairs.append(compute_pair_metrics(pred, gt))

    summary = audit_validation_errors(pairs)

    print(f"\n1. Global Detection & Isolation Breakdown (Total N = {summary['total_pairs']} pairs):")
    print(f"   • GT Positive Pairs: {summary['positive_pairs']}")
    print(f"   • GT Clean Pairs:    {summary['clean_pairs']}")
    print(f"   • Clean Image Specificity:     {summary['specificity']:.1%} ({summary['clean_pairs'] - summary['false_alarms']}/{summary['clean_pairs']} correct clean)")
    print(f"   • Clean Image False Alarms:   {summary['false_alarms']} instances ({summary['false_alarms'] / summary['clean_pairs']:.1%})")

    print(f"\n2. Detection vs. Overlap Breakdown on GT Positive Pairs:")
    print(f"   • Presence Recall:             {summary['presence_recall']:.1%} (Model predicted >0 px)")
    print(f"   • Overlap Recall:              {summary['overlap_recall']:.1%} (Model intersected GT by >0 px)")
    print(f"   • Completely Missed Defects:   {summary['completely_missed']} pairs")
    print(f"   • Predicted but NO Overlap:    {summary['pred_no_overlap']} pairs (Predicted wrong location)")
    print(f"   • Conditional Overlap Dice:    {summary['conditional_dice']:.4f} (Dice ONLY when defect is touched)")

    print(f"\n3. Defect Performance Stratified by Ground-Truth Area:")
    print(f"{'Size Category':<22} | {'Count':<6} | {'Overlap Recall':<15} | {'Mean Dice':<10}")
    print("-" * 62)
    for cat, stats in summary["size_bins"].items():
        print(f"{cat:<22} | {stats['count']:>5}  | {stats['overlap_recall']:>13.1%} | {stats['mean_dice']:>9.4f}")

    print("\nKey Takeaways:")
    print("• Notice how small defects (1..100 px) suffer 0.0% recall in this simulation.")
    print("  This pinpoints downsampling loss or missing skip connections, NOT a general model failure.")
    print("• Conditional Overlap Dice (0.76+) reveals that when the model actually detects")
    print("  the defect, boundary precision is solid. The primary bottleneck is presence recall.")
    print("• Separating Presence from Overlap prevents wasting time tuning loss weights")
    print("  when the real issue is encoder spatial resolution or minimum area post-processing.")
    print("=" * 75)


if __name__ == "__main__":
    main()
