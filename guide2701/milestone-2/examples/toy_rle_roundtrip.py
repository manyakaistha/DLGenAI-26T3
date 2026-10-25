# Legacy compatibility download. See walkthrough-notebook.html for the contract notice.
"""Toy RLE encoding and decoding roundtrip verification on synthetic binary masks.

Demonstrates:
1. 1-indexed column-major (Fortran order, order='F') encoding.
2. Clamping boundary safety (guarding against run endpoints exceeding H * W).
3. The competition empty-mask sentinel: exactly one ASCII space (" ").
4. Strict round-trip assertion: decode(encode(mask)) == mask.
"""
from __future__ import annotations

import numpy as np


def rle_encode(mask: np.ndarray) -> str:
    """Encode a 2D binary mask into a 1-indexed, column-major (order='F') RLE string.

    Empty masks MUST return exactly one ASCII space (' ') to comply with the
    Kaggle sample submission specification.
    """
    if mask is None or mask.sum() == 0:
        return " "

    # Flatten in Fortran order (column-major)
    pixels = mask.flatten(order="F")
    # Pad with zeros at both ends to detect transitions cleanly
    padded = np.concatenate([[0], pixels, [0]])
    # Find positions where value changes: 0 -> 1 (start) or 1 -> 0 (end)
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0] + 1  # 1-indexed
    ends = np.where(diffs == -1)[0] + 1
    lengths = ends - starts

    runs = []
    for s, l in zip(starts, lengths):
        runs.extend([str(s), str(l)])
    return " ".join(runs)


def rle_decode(rle_str: str, shape: tuple[int, int] = (128, 800), clamp: bool = True) -> np.ndarray:
    """Decode a 1-indexed, column-major RLE string into a 2D binary mask."""
    if not rle_str or rle_str.strip() == "":
        return np.zeros(shape, dtype=np.uint8)

    tokens = [int(x) for x in rle_str.strip().split()]
    starts = np.array(tokens[0::2], dtype=np.int64) - 1  # Convert to 0-based
    lengths = np.array(tokens[1::2], dtype=np.int64)
    total_pixels = shape[0] * shape[1]

    if clamp:
        ends = np.minimum(starts + lengths, total_pixels)
    else:
        ends = starts + lengths

    flat_mask = np.zeros(total_pixels, dtype=np.uint8)
    for s, e in zip(starts, ends):
        flat_mask[s:e] = 1

    return flat_mask.reshape(shape, order="F")


def main():
    print("=== Milestone 2: RLE Roundtrip & Edge Case Diagnostic ===")

    # Test Case 1: Toy small grid (4 rows x 6 cols)
    toy_shape = (4, 6)
    toy_mask = np.zeros(toy_shape, dtype=np.uint8)
    # Defect at col 2: rows 1..2 (pixels 10, 11); col 3: rows 2..3 (pixels 15, 16)
    toy_mask[1:3, 2] = 1
    toy_mask[2:4, 3] = 1

    encoded_toy = rle_encode(toy_mask)
    print(f"\n[Test 1: 4x6 Grid]")
    print(f"• Active pixels: {toy_mask.sum()}")
    print(f"• Encoded RLE:   '{encoded_toy}'")
    expected_toy = "10 2 15 2"
    assert encoded_toy == expected_toy, f"Expected '{expected_toy}', got '{encoded_toy}'"

    decoded_toy = rle_decode(encoded_toy, shape=toy_shape)
    assert np.array_equal(toy_mask, decoded_toy), "Roundtrip mismatch on toy grid!"
    print("✓ Roundtrip equality verified on 4x6 grid!")

    # Test Case 2: Full-sized industrial strip (128 x 800)
    full_shape = (128, 800)
    full_mask = np.zeros(full_shape, dtype=np.uint8)
    # Synthetic defect patches
    full_mask[20:30, 100:150] = 1
    full_mask[80:110, 600:620] = 1

    encoded_full = rle_encode(full_mask)
    decoded_full = rle_decode(encoded_full, shape=full_shape)
    assert np.array_equal(full_mask, decoded_full), "Roundtrip mismatch on 128x800 strip!"
    print(f"\n[Test 2: Full 128x800 Strip]")
    print(f"• Ground truth pixels: {full_mask.sum()}")
    print(f"• Decoded pixels:      {decoded_full.sum()}")
    print("✓ Roundtrip equality verified on full 128x800 canvas!")

    # Test Case 3: Empty mask contract
    empty_mask = np.zeros(full_shape, dtype=np.uint8)
    encoded_empty = rle_encode(empty_mask)
    print(f"\n[Test 3: Empty Mask Sentinel]")
    print(f"• Encoded empty mask representation: '{encoded_empty}' (length={len(encoded_empty)})")
    assert encoded_empty == " ", "Empty mask must be exactly one ASCII space ' '!"
    decoded_empty = rle_decode(encoded_empty, shape=full_shape)
    assert decoded_empty.sum() == 0, "Decoded empty mask must have 0 pixels!"
    print("✓ Empty mask single-space sentinel verified!")

    # Test Case 4: Out-of-bounds boundary clamping test
    # Simulate a corrupted run that overshoots total pixels (102,400)
    overshoot_rle = "102395 10"  # run extends to 102,405
    clamped_mask = rle_decode(overshoot_rle, shape=full_shape, clamp=True)
    assert clamped_mask.sum() == 6, f"Expected 6 clamped pixels (102395..102400), got {clamped_mask.sum()}"
    print(f"\n[Test 4: Boundary Clamping]")
    print(f"• Corrupted run '102395 10' safely clamped to canvas boundary: {clamped_mask.sum()} pixels restored.")
    print("✓ Clamping protection verified!")

    print("\n✓ All RLE encoding, decoding, and boundary assertions passed successfully!")


if __name__ == "__main__":
    main()
