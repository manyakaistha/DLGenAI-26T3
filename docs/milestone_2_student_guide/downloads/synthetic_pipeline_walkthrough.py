# %% [markdown]
# # Milestone 2: Synthetic Segmentation Pipeline Walkthrough
#
# **Industrial Surface Anomaly Segmentation**
#
# **Legacy reference:** this demo uses grayscale `[1, 128, 800]` images, standardized
# pixels and dictionary samples. Current Q7 requires RGB `[3, 128, 800]` images
# scaled to `[0, 1]` and an `(image, masks)` tuple. The random textures do not
# depict the annotation shapes: this is a pipeline smoke test, not a learnable benchmark.
#
# This educational notebook demonstrates the **conceptual mechanics** of every major stage in a surface anomaly segmentation pipeline using **100% synthetic data**:
# 1. Synthetic Strip Generator & Long-Format Annotation Schema
# 2. PyTorch Dataset Contract: Image `[1, 128, 800]` and Multi-Label Mask `[4, 128, 800]`
# 3. Augmentation Alignment: Joint Geometric Transforms vs Broken Decoupled Transforms
# 4. Multi-Label Combination Stratification on Fictional Signatures
# 5. Baseline U-Net Forward/Backward Lifecycle on CPU
# 6. Post-Processing: Logits $\to$ Probabilities $\to$ Thresholding $\to$ Area Filtering
# 7. RLE Encoding: 1-Based Column-Major Traversal & Single-Space Sentinel
#
# *Note: This notebook uses zero competition data and runs on CPU in seconds.*

# %%
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2

print("PyTorch Version:", torch.__version__)
print("OpenCV Version: ", cv2.__version__)
print("NumPy Version:  ", np.__version__)

# %% [markdown]
# ## 1. Synthetic Data Generator & Schema Setup
#
# In this competition, images are line-scan steel strips ($128 \times 800$ pixels) and `train.csv` provides 4 rows per image (one for each defect type: `defect_a`, `defect_b`, `defect_c`, `defect_d`).

# %%
def generate_synthetic_strip(image_id: str, shape: tuple[int, int] = (128, 800)) -> np.ndarray:
    """Generate a synthetic line-scan steel strip with subtle rolling texture."""
    np.random.seed(abs(hash(image_id)) % (2**31))
    base_intensity = 150
    noise = np.random.normal(0, 10, shape).astype(np.int16)
    strip = np.clip(base_intensity + noise, 0, 255).astype(np.uint8)
    return strip

# Create fictional annotation table for 6 synthetic images
classes = ["defect_a", "defect_b", "defect_c", "defect_d"]
records = []
for i in range(6):
    img_id = f"STRIP_{i:04d}"
    for cls in classes:
        rle = ""
        # Inject synthetic defects into select items
        if i == 0 and cls == "defect_a":
            rle = "1281 20 1409 20"   # Col-major scratch
        elif i == 0 and cls == "defect_c":
            rle = "25601 50 25729 50" # Col-major patch
        elif i == 2 and cls == "defect_b":
            rle = "50000 80"          # Rare defect b
        elif i == 4 and cls == "defect_d":
            rle = "10000 40"
        records.append({
            "image_id": img_id,
            "defect_type": cls,
            "mask_rle": rle,
            "width": 800,
            "height": 128,
        })

df_annotations = pd.DataFrame(records)
print(f"Generated {len(df_annotations)} rows across {df_annotations['image_id'].nunique()} images.")
df_annotations.head(8)

# %% [markdown]
# ## 2. PyTorch Dataset Contract
#
# The dataset contract requires:
# - Input image: `[1, 128, 800]` float32 (single-channel grayscale).
# - Target mask: `[4, 128, 800]` float32 (four binary channels).

# %%
def rle_decode(rle_str: str | float, shape: tuple[int, int] = (128, 800)) -> np.ndarray:
    if pd.isna(rle_str) or not str(rle_str).strip():
        return np.zeros(shape, dtype=np.uint8)
    tokens = [int(x) for x in str(rle_str).strip().split()]
    starts = np.array(tokens[0::2], dtype=np.int64) - 1
    lengths = np.array(tokens[1::2], dtype=np.int64)
    ends = np.minimum(starts + lengths, shape[0] * shape[1])
    flat = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for s, e in zip(starts, ends):
        flat[s:e] = 1
    return flat.reshape(shape, order="F")

class SyntheticStripDataset(Dataset):
    def __init__(self, df: pd.DataFrame, shape: tuple[int, int] = (128, 800)):
        self.shape = shape
        self.classes = ["defect_a", "defect_b", "defect_c", "defect_d"]
        self.grouped = df.groupby("image_id")
        self.image_ids = list(self.grouped.groups.keys())

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        img_id = self.image_ids[idx]
        rows = self.grouped.get_group(img_id).set_index("defect_type")

        # 1. Image loading & normalization
        raw_img = generate_synthetic_strip(img_id, self.shape)
        img_norm = (raw_img.astype(np.float32) / 255.0 - 0.5) / 0.25

        # 2. 4-Channel Mask decoding
        masks = np.zeros((4, self.shape[0], self.shape[1]), dtype=np.float32)
        for c_i, cls in enumerate(self.classes):
            if cls in rows.index:
                masks[c_i] = rle_decode(rows.loc[cls, "mask_rle"], self.shape)

        return {
            "image_id": img_id,
            "image": torch.from_numpy(img_norm).unsqueeze(0).float(),
            "mask": torch.from_numpy(masks).float(),
        }

dataset = SyntheticStripDataset(df_annotations)
sample = dataset[0]
print(f"Sample image shape: {sample['image'].shape}, dtype: {sample['image'].dtype}")
print(f"Sample mask shape:  {sample['mask'].shape}, dtype: {sample['mask'].dtype}")
print(f"Defect channels active in sample 0: defect_a={sample['mask'][0].sum():.0f} px, defect_c={sample['mask'][2].sum():.0f} px")

# %% [markdown]
# ## 3. Augmentation Alignment: Joint vs Broken
#
# Spatial augmentations (like horizontal flips) MUST be applied identically to both image and mask!

# %%
def apply_joint_hflip(img: np.ndarray, mask: np.ndarray):
    """Flip both image and mask horizontally."""
    return np.fliplr(img), np.flip(mask, axis=-1)

raw_sample_img = generate_synthetic_strip("STRIP_0000")
sample_mask = sample["mask"].numpy()

# 1. Correct joint transform
flipped_img, flipped_mask = apply_joint_hflip(raw_sample_img, sample_mask)
print("Correct joint transform: image shape", flipped_img.shape, "mask shape", flipped_mask.shape)

# 2. The broken unsynchronized transform:
broken_img = np.fliplr(raw_sample_img)
broken_mask = sample_mask  # Left unflipped!
print("Unsynchronized transform: Mask coordinates no longer match image defect!")

# %% [markdown]
# ## 4. Multi-Label Combination Stratification
#
# Combination signatures help inspect rare groups before splitting. They cannot guarantee validation support when a group is too small; this example only counts signatures and does not perform a split.

# %%
def compute_combination_signatures(df: pd.DataFrame) -> pd.Series:
    """Create a combination signature string for each image."""
    present = df["mask_rle"].fillna("").astype(str).str.strip().ne("")
    active = df[present]
    grouped = active.groupby("image_id")["defect_type"].apply(lambda s: "+".join(sorted(s)))
    # Map back to all unique image IDs; empty images become 'clean'
    all_ids = pd.Series(index=df["image_id"].unique(), data="clean", dtype=object)
    all_ids.update(grouped)
    return all_ids

signatures = compute_combination_signatures(df_annotations)
print("Combination signature distribution across synthetic dataset:")
print(signatures.value_counts())

# %% [markdown]
# ## 5. Baseline U-Net Architecture & Optimization Lifecycle
#
# Let's verify a forward and backward step through a minimal 2-level U-Net on CPU.

# %%
class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class MinimalUNet(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 4, base_ch: int = 16):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base_ch)
        self.pool1 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base_ch, base_ch * 2)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec1 = DoubleConv(base_ch * 3, base_ch)
        self.head = nn.Conv2d(base_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        b = self.bottleneck(self.pool1(e1))
        d1 = self.dec1(torch.cat([self.up1(b), e1], dim=1))
        return self.head(d1)

model = MinimalUNet()
loader = DataLoader(dataset, batch_size=2, shuffle=True)
batch = next(iter(loader))

# Forward pass
logits = model(batch["image"])
print("Forward Pass Logits Shape:", tuple(logits.shape))

# Loss & Backward
criterion = nn.BCEWithLogitsLoss()
loss = criterion(logits, batch["mask"])
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
optimizer.zero_grad()
loss.backward()
optimizer.step()
print(f"Loss computed: {loss.item():.4f}, Gradients stepped successfully!")

# %% [markdown]
# ## 6. Post-Processing: Logits $\to$ Probabilities $\to$ Filtered Masks
#
# Thresholding probabilities at $\tau = 0.5$ and removing connected components smaller than $A_{\min} = 50$ px.

# %%
probs = torch.sigmoid(logits).detach().numpy()[0, 0]  # Channel 0
binary_raw = (probs >= 0.5).astype(np.uint8)

# Connected component area filtering
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_raw, connectivity=8)
binary_filtered = np.zeros_like(binary_raw)
min_area = 50
for lbl in range(1, num_labels):
    area = stats[lbl, cv2.CC_STAT_AREA]
    if area >= min_area:
        binary_filtered[labels == lbl] = 1

print(f"Raw binary mask positive pixels:      {binary_raw.sum()}")
print(f"Filtered binary mask positive pixels: {binary_filtered.sum()}")

# %% [markdown]
# ## 7. RLE Encoding & Submission Sentinel Check
#
# Encode 2D binary mask to 1-indexed column-major RLE string, returning `" "` for empty masks.

# %%
def rle_encode(mask: np.ndarray) -> str:
    if mask is None or mask.sum() == 0:
        return " "
    pixels = mask.flatten(order="F")
    padded = np.concatenate([[0], pixels, [0]])
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0] + 1
    ends = np.where(diffs == -1)[0] + 1
    lengths = ends - starts
    runs = []
    for s, l in zip(starts, lengths):
        runs.extend([str(s), str(l)])
    return " ".join(runs)

# Verify non-empty encoding
toy_test_mask = np.zeros((4, 6), dtype=np.uint8)
toy_test_mask[1:3, 2] = 1
encoded_test = rle_encode(toy_test_mask)
print(f"Encoded 4x6 mask: '{encoded_test}' (Expected: '10 2')")
assert encoded_test == "10 2"

# Verify empty mask returns single-space sentinel
empty_test_mask = np.zeros((128, 800), dtype=np.uint8)
encoded_empty = rle_encode(empty_test_mask)
print(f"Encoded empty mask representation: '{encoded_empty}' (len={len(encoded_empty)})")
assert encoded_empty == " "

print("\n✓ Entire synthetic pipeline walkthrough completed successfully!")
