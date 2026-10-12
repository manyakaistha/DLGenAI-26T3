"""Milestone 5 Diagnostic: Threshold Calibration on Synthetic Probability Maps.

Demonstrates how to sweep decision thresholds tau to optimize the competition's
weighted metric (Foreground Dice + FP penalty + Empty accuracy) vs naive mean Dice.
"""
from __future__ import annotations

import numpy as np


def compute_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict[str, float]:
    """Compute foreground Dice, empty accuracy, and false positive penalty."""
    pred_sum = pred_mask.sum()
    gt_sum = gt_mask.sum()
    intersection = (pred_mask * gt_mask).sum()

    is_gt_positive = gt_sum > 0
    is_pred_positive = pred_sum > 0

    if is_gt_positive:
        # Positive pair: evaluate Foreground Dice
        fg_dice = (2.0 * intersection) / (pred_sum + gt_sum)
        empty_acc = 0.0  # Not applicable to positive ground truth
        fp_score = 1.0   # No false positive on positive GT
    else:
        # Empty ground truth pair
        fg_dice = 0.0    # No foreground ground truth
        empty_acc = 1.0 if not is_pred_positive else 0.0
        # If model predicted pixels on a clean image, penalize
        fp_score = 0.0 if is_pred_positive else 1.0

    # Naive Dice: treats both-empty as 1.0, otherwise 2*inter/(P+G)
    if not is_gt_positive and not is_pred_positive:
        naive_dice = 1.0
    elif is_gt_positive:
        naive_dice = fg_dice
    else:
        naive_dice = 0.0

    return {
        "fg_dice": fg_dice,
        "empty_acc": empty_acc,
        "fp_score": fp_score,
        "naive_dice": naive_dice,
        "is_gt_positive": is_gt_positive,
    }


def main():
    print("=" * 70)
    print("Milestone 5: Synthetic Threshold Calibration & Metric Divergence")
    print("=" * 70)

    np.random.seed(42)
    num_samples = 40
    H, W = 128, 800

    # Create synthetic batch: 20 positive strips (with defect), 20 clean strips
    probs = []
    gts = []

    for i in range(num_samples):
        gt = np.zeros((H, W), dtype=np.uint8)
        p = np.random.beta(0.5, 9.0, size=(H, W)).astype(np.float32)  # Low background noise

        if i < 20:
            # Genuine defect: longitudinal scratch or patch
            row_start = np.random.randint(20, 100)
            col_start = np.random.randint(100, 600)
            width = np.random.randint(50, 150)
            height = np.random.randint(4, 12)
            gt[row_start:row_start+height, col_start:col_start+width] = 1

            # High model confidence with boundary fuzziness
            p[row_start:row_start+height, col_start:col_start+width] = np.random.beta(5.0, 1.5, size=(height, width))
        else:
            # Clean sheet: occasional small noise cluster
            if np.random.rand() > 0.5:
                p[50:53, 300:303] = 0.55  # Mild false alarm speckle

        probs.append(p)
        gts.append(gt)

    thresholds = [0.20, 0.35, 0.50, 0.65, 0.80]
    print(f"\nEvaluating {num_samples} synthetic strips across thresholds: {thresholds}\n")
    print(f"{'Tau':<8} | {'FG Dice (70%)':<14} | {'FP Score (20%)':<14} | {'Empty Acc (10%)':<15} | {'Comp Score':<12} | {'Naive Dice':<10}")
    print("-" * 85)

    best_comp_score = -1.0
    best_tau = None

    for tau in thresholds:
        fg_dices = []
        fp_scores = []
        empty_accs = []
        naive_dices = []

        for p, gt in zip(probs, gts):
            pred_mask = (p >= tau).astype(np.uint8)
            res = compute_metrics(pred_mask, gt)

            if res["is_gt_positive"]:
                fg_dices.append(res["fg_dice"])
            else:
                fp_scores.append(res["fp_score"])
                empty_accs.append(res["empty_acc"])

            naive_dices.append(res["naive_dice"])

        avg_fg_dice = float(np.mean(fg_dices))
        avg_fp_score = float(np.mean(fp_scores))
        avg_empty_acc = float(np.mean(empty_accs))
        avg_naive_dice = float(np.mean(naive_dices))

        comp_score = (0.70 * avg_fg_dice) + (0.20 * avg_fp_score) + (0.10 * avg_empty_acc)

        if comp_score > best_comp_score:
            best_comp_score = comp_score
            best_tau = tau

        print(f"{tau:<8.2f} | {avg_fg_dice:<14.4f} | {avg_fp_score:<14.4f} | {avg_empty_acc:<15.4f} | {comp_score:<12.4f} | {avg_naive_dice:<10.4f}")

    print("-" * 85)
    print(f"[Result] Best Competition Score = {best_comp_score:.4f} achieved at tau = {best_tau:.2f}")
    print("[Insight] Notice how Naive Mean Dice favors higher thresholds (aggressive abstention),")
    print("          whereas Competition Score rewards preserving true Foreground Dice!")
    print("=" * 70)


if __name__ == "__main__":
    main()
