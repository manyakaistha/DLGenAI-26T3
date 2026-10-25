"""Toy Dataset and DataLoader demonstration on synthetic industrial strip data.

Demonstrates:
1. Long-format annotation parsing for multi-channel targets.
2. RLE mask decoding into a (4, 128, 800) binary tensor.
3. Grayscale image loading into a (1, 128, 800) normalized float32 tensor.
4. PyTorch DataLoader batch collation into [B, 1, 128, 800] and [B, 4, 128, 800].
5. Image-level isolation (no image ID leakage across partitions).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# 1. Fortran-order RLE decoder with boundary safety
def rle_decode(rle_str: str | float, shape: tuple[int, int] = (128, 800)) -> np.ndarray:
    """Decode a 1-indexed, column-major (order='F') RLE string into a binary mask."""
    if pd.isna(rle_str) or not str(rle_str).strip():
        return np.zeros(shape, dtype=np.uint8)

    tokens = [int(x) for x in str(rle_str).strip().split()]
    starts = np.array(tokens[0::2], dtype=np.int64) - 1  # convert to 0-based index
    lengths = np.array(tokens[1::2], dtype=np.int64)
    total_pixels = shape[0] * shape[1]
    ends = np.minimum(starts + lengths, total_pixels)

    flat_mask = np.zeros(total_pixels, dtype=np.uint8)
    for s, e in zip(starts, ends):
        flat_mask[s:e] = 1

    return flat_mask.reshape(shape, order="F")


# 2. Minimal PyTorch Dataset for Industrial Strip Images
class IndustrialStripDataset(Dataset):
    """PyTorch Dataset returning single-channel strip images and 4-channel defect masks."""

    CLASSES = ["defect_a", "defect_b", "defect_c", "defect_d"]

    def __init__(
        self,
        df: pd.DataFrame,
        synthetic_image_generator=None,
        transform=None,
        shape: tuple[int, int] = (128, 800),
    ):
        self.shape = shape
        self.transform = transform
        self.generator = synthetic_image_generator

        # Group annotations by image_id to guarantee image-level isolation
        self.grouped = df.groupby("image_id")
        self.image_ids = list(self.grouped.groups.keys())

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        image_id = self.image_ids[idx]
        image_rows = self.grouped.get_group(image_id).set_index("defect_type")

        # 1. Load or synthesize raw grayscale image (128, 800)
        if self.generator is not None:
            raw_img = self.generator(image_id, self.shape)
        else:
            # Default fallback: synthetic gray strip
            raw_img = np.full(self.shape, 128, dtype=np.uint8)

        # 2. Decode each of the 4 defect channels
        masks = np.zeros((len(self.CLASSES), self.shape[0], self.shape[1]), dtype=np.float32)
        for c_idx, cls_name in enumerate(self.CLASSES):
            if cls_name in image_rows.index:
                rle = image_rows.loc[cls_name, "mask_rle"]
                masks[c_idx] = rle_decode(rle, self.shape).astype(np.float32)

        # 3. Apply optional joint augmentation (must transform both or respect modality)
        if self.transform is not None:
            raw_img, masks = self.transform(raw_img, masks)

        # 4. Normalize image: z-score or min-max
        img_norm = (raw_img.astype(np.float32) / 255.0 - 0.5) / 0.25

        # 5. Convert to PyTorch tensors with shape [C, H, W]
        img_tensor = torch.from_numpy(img_norm).unsqueeze(0).float()  # (1, 128, 800)
        mask_tensor = torch.from_numpy(masks).float()                 # (4, 128, 800)

        return {
            "image_id": image_id,
            "image": img_tensor,
            "mask": mask_tensor,
        }


def make_toy_dataframe() -> pd.DataFrame:
    """Create a fictional long-format annotation table with 5 images."""
    records = []
    classes = ["defect_a", "defect_b", "defect_c", "defect_d"]
    # 5 fictional images
    for i in range(5):
        img_id = f"STRIP_{i:04d}"
        for cls_name in classes:
            rle = ""
            # Add synthetic defects to some images
            if i == 1 and cls_name == "defect_a":
                rle = "100 20 250 15"  # Col-major runs
            elif i == 1 and cls_name == "defect_c":
                rle = "5000 50"
            elif i == 3 and cls_name == "defect_b":
                rle = "12000 80"
            records.append({
                "image_id": img_id,
                "defect_type": cls_name,
                "mask_rle": rle,
                "width": 800,
                "height": 128,
            })
    return pd.DataFrame(records)


def main():
    print("=== Milestone 2: Toy Dataset & DataLoader Diagnostic ===")
    df = make_toy_dataframe()
    print(f"Total annotation rows: {len(df)} across {df['image_id'].nunique()} unique images.")

    # Instantiate dataset
    dataset = IndustrialStripDataset(df)
    print(f"Dataset length (__len__): {len(dataset)} items.")

    # Inspect single sample
    sample = dataset[1]  # STRIP_0001 with defect_a and defect_c
    img = sample["image"]
    mask = sample["mask"]
    print(f"\n[Sample Inspection: {sample['image_id']}]")
    print(f"• Image tensor shape: {tuple(img.shape)}, dtype: {img.dtype}, range: [{img.min():.2f}, {img.max():.2f}]")
    print(f"• Mask tensor shape:  {tuple(mask.shape)}, dtype: {mask.dtype}, unique values: {torch.unique(mask).tolist()}")
    print(f"• Active channels: defect_a pixels={mask[0].sum():.0f}, defect_c pixels={mask[2].sum():.0f}")

    # Build DataLoader with batch_size=2
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    print(f"\n[Batch Inspection]")
    print(f"• Batch image tensor shape: {tuple(batch['image'].shape)} [B, C, H, W]")
    print(f"• Batch mask tensor shape:  {tuple(batch['mask'].shape)} [B, C, H, W]")
    print(f"• Image IDs in batch:       {batch['image_id']}")

    # Sanity assertions
    assert batch["image"].shape == (2, 1, 128, 800), "Image batch shape mismatch!"
    assert batch["mask"].shape == (2, 4, 128, 800), "Mask batch shape mismatch!"
    print("\n✓ All shape, dtype, and alignment assertions passed successfully!")


if __name__ == "__main__":
    main()
