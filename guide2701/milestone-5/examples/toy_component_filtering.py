"""Milestone 5 Diagnostic: 8-Connected Component Minimum-Area Filtering.

Demonstrates grouping thresholded pixels into 8-connected components,
measuring their area, removing high-frequency noise speckles, and verifying
that genuine defect structures are preserved without pixel loss.
"""
from __future__ import annotations

import cv2
import numpy as np


def filter_small_components(mask: np.ndarray, min_area: int, connectivity: int = 8) -> np.ndarray:
    """Filter out connected components smaller than min_area pixels."""
    if min_area <= 0 or mask.sum() == 0:
        return mask.copy()

    mask_uint8 = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=connectivity)

    clean_mask = np.zeros_like(mask_uint8)
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= min_area:
            clean_mask[labels == label_id] = 1

    return clean_mask


def main():
    print("=" * 70)
    print("Milestone 5: Connected Component Minimum-Area Filtering")
    print("=" * 70)

    # 1. Create a 30x80 canvas
    H, W = 30, 80
    raw_mask = np.zeros((H, W), dtype=np.uint8)

    # 2. Add a genuine 60-pixel longitudinal scratch (3 x 20 pixels)
    scratch_row, scratch_col = 10, 25
    scratch_h, scratch_w = 3, 20
    raw_mask[scratch_row:scratch_row + scratch_h, scratch_col:scratch_col + scratch_w] = 1
    true_defect_pixels = scratch_h * scratch_w

    # 3. Add 4 isolated noise speckles (areas: 1 px, 2 px, 3 px, 5 px)
    speckles = [
        ((3, 5), (1, 1)),    # 1 px
        ((5, 70), (1, 2)),   # 2 px
        ((22, 10), (1, 3)),  # 3 px
        ((25, 60), (1, 5)),  # 5 px
    ]
    total_speckle_pixels = 0
    for (r, c), (h, w) in speckles:
        raw_mask[r:r+h, c:c+w] = 1
        total_speckle_pixels += (h * w)

    print(f"Canvas size: {H}x{W} pixels")
    print(f"Initial active pixels: {raw_mask.sum()}")
    print(f"  • Genuine defect pixels: {true_defect_pixels} (1 component of size {true_defect_pixels} px)")
    print(f"  • Noise speckle pixels : {total_speckle_pixels} (4 components of sizes 1, 2, 3, 5 px)")

    # 4. Connected component analysis before filtering
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw_mask, connectivity=8)
    print(f"\n[Before Filtering] Found {num_labels - 1} connected components (8-connectivity):")
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        print(f"  Component #{lbl}: area = {area:>2} px (status: {'GENUINE DEFECT' if area >= 20 else 'NOISE SPECKLE'})")

    # 5. Apply filtering with min_area = 20 px
    min_area = 20
    clean_mask = filter_small_components(raw_mask, min_area=min_area, connectivity=8)

    print(f"\n[After Filtering with min_area = {min_area} px]")
    print(f"Active pixels remaining: {clean_mask.sum()}")

    # 6. Strict assertions
    assert clean_mask.sum() == true_defect_pixels, f"Expected {true_defect_pixels} px, got {clean_mask.sum()}"
    # Verify the genuine defect is 100% intact
    scratch_region = clean_mask[scratch_row:scratch_row + scratch_h, scratch_col:scratch_col + scratch_w]
    assert np.all(scratch_region == 1), "Genuine defect pixels were corrupted!"
    # Verify all speckles are completely removed
    for (r, c), (h, w) in speckles:
        assert np.all(clean_mask[r:r+h, c:c+w] == 0), f"Speckle at ({r}, {c}) was not zeroed out!"

    print("✓ All 4 noise speckles were successfully purged.")
    print("✓ Genuine defect structure was 100% preserved.")
    print("✓ 8-connected component filtering verified successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
