"""Original synthetic cloud Dataset/DataLoader teaching lab. See notes.md."""
# %% 1. Imports and the teaching contract
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt

CLASSES = ("Fish", "Flower", "Gravel", "Sugar")
ROOT = Path("runs/cloud_loader_demo")
ROOT.mkdir(parents=True, exist_ok=True)

# %% 2. RLE: the CSV stores coordinates compactly
def encode_rle(mask):
    pixels = np.asarray(mask, dtype=np.uint8).reshape(-1, order="F")
    padded = np.concatenate(([0], pixels, [0]))
    changes = np.flatnonzero(np.diff(padded)) + 1
    changes[1::2] -= changes[::2]
    return " ".join(map(str, changes))


def decode_rle(text, height, width):
    flat = np.zeros(height * width, dtype=np.uint8)
    if pd.isna(text) or not str(text).strip():
        return flat.reshape(height, width, order="F")
    values = np.array(str(text).split(), dtype=np.int64)
    if len(values) % 2:
        raise ValueError("RLE needs start-length pairs")
    previous_end = 0
    for start, length in values.reshape(-1, 2):
        lo = int(start) - 1
        hi = lo + int(length)
        if lo < previous_end or length <= 0 or hi > len(flat):
            raise ValueError("Invalid or overlapping RLE run")
        flat[lo:hi] = 1
        previous_end = hi
    return flat.reshape(height, width, order="F")

small = np.zeros((4, 6), dtype=np.uint8)
small[1:3, 2] = 1
assert encode_rle(small) == "10 2"
assert np.array_equal(decode_rle("10 2", 4, 6), small)
assert decode_rle("", 4, 6).sum() == 0
print("RLE round trip passed")

# %% 3. Create a small, independent dataset
def create_demo(root, n_train=24, n_test=4):
    rng = np.random.default_rng(42)
    height, width = 60, 90
    yy, xx = np.mgrid[:height, :width]
    rows, submission = [], []
    for folder, count in (("train_images", n_train), ("test_images", n_test)):
        (root / folder).mkdir(exist_ok=True)
        for i in range(count):
            filename = f"demo_{i:03d}.jpg"
            image = np.zeros((height, width, 3), dtype=np.float32)
            image[:] = (35, 90, 135)
            image += rng.normal(0, 5, image.shape)
            for channel, label in enumerate(CLASSES):
                mask = np.zeros((height, width), dtype=np.uint8)
                if i % 6 != 0 and (i + channel) % 3 != 0:
                    cx, cy = rng.integers(18, 72), rng.integers(15, 45)
                    mask = (((xx-cx)/16)**2 + ((yy-cy)/9)**2 < 1).astype(np.uint8)
                    image[mask.astype(bool)] += 25 + channel * 8
                key = f"{filename}_{label}"
                if folder == "train_images":
                    rows.append({"Image_Label": key, "EncodedPixels": encode_rle(mask)})
                else:
                    submission.append({"Image_Label": key, "EncodedPixels": ""})
            Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).save(root / folder / filename)
    pd.DataFrame(rows).to_csv(root / "train.csv", index=False)
    pd.DataFrame(submission).to_csv(root / "sample_submission.csv", index=False)

create_demo(ROOT)
table = pd.read_csv(ROOT / "train.csv", keep_default_na=False)
print(table.head(8).to_string(index=False))
print("CSV rows:", len(table), "— training images:", len(table) // len(CLASSES))

# %% 4. Build an index: one image, four rows
def read_index(root):
    frame = pd.read_csv(root / "train.csv", keep_default_na=False)
    parts = frame["Image_Label"].str.rsplit("_", n=1, expand=True)
    frame["filename"], frame["label"] = parts[0], parts[1]
    if not frame["label"].isin(CLASSES).all():
        raise ValueError("Unknown class")
    if frame.duplicated(["filename", "label"]).any():
        raise ValueError("Duplicate image-class pair")
    counts = frame.groupby("filename")["label"].nunique()
    if not counts.eq(len(CLASSES)).all():
        raise ValueError("Every image needs four explicit annotation rows")
    return frame.pivot(index="filename", columns="label", values="EncodedPixels").reindex(columns=CLASSES)

index = read_index(ROOT)
print(index.index[:3].tolist())
print("Dataset samples:", len(index), "Mask channels:", len(index.columns))

# %% 5. Write the Dataset: constructor, length, item
class CloudDataset(Dataset):
    def __init__(self, root, size=(60, 90), transform=None):
        self.root = Path(root)
        self.index = read_index(self.root)
        self.size = size
        self.transform = transform

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        filename = self.index.index[i]
        with Image.open(self.root / "train_images" / filename) as source:
            image = source.convert("RGB")
        width, height = image.size
        masks = [decode_rle(self.index.iloc[i][label], height, width) for label in CLASSES]
        out_h, out_w = self.size
        image = image.resize((out_w, out_h), Image.Resampling.BILINEAR)
        masks = [np.asarray(Image.fromarray(m).resize(
            (out_w, out_h), Image.Resampling.NEAREST), dtype=np.float32) for m in masks]
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).contiguous()
        mask_tensor = torch.from_numpy(np.stack(masks))
        if self.transform is not None:
            image_tensor, mask_tensor = self.transform(image_tensor, mask_tensor)
        return {"image": image_tensor, "mask": mask_tensor, "filename": filename}

dataset = CloudDataset(ROOT)
sample = dataset[1]
print("Images:", len(dataset))
print(sample["filename"], sample["image"].shape, sample["mask"].shape)
assert sample["image"].dtype == torch.float32
assert sample["mask"].shape == (4, 60, 90)
assert set(sample["mask"].unique().tolist()) <= {0.0, 1.0}

# %% 6. Look before you batch
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
rgb = sample["image"].permute(1, 2, 0).numpy()
axes[0].imshow(rgb)
axes[0].set_title(sample["filename"])
for channel, label in enumerate(CLASSES):
    axes[channel + 1].imshow(rgb)
    visible = np.ma.masked_where(sample["mask"][channel].numpy() == 0,
                                 sample["mask"][channel].numpy())
    axes[channel + 1].imshow(visible, cmap="autumn", vmin=0, vmax=1, alpha=0.6)
    axes[channel + 1].set_title(label)
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()

# %% 7. Split images, then create DataLoaders
order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(42)).tolist()
cut = int(0.8 * len(order))
train_indices, val_indices = order[:cut], order[cut:]
assert set(train_indices).isdisjoint(val_indices)
train_data = Subset(dataset, train_indices)
val_data = Subset(dataset, val_indices)
train_loader = DataLoader(
    train_data, batch_size=4, shuffle=True, num_workers=0,
    drop_last=False, generator=torch.Generator().manual_seed(123),
)
val_loader = DataLoader(val_data, batch_size=4, shuffle=False, num_workers=0)
batch = next(iter(train_loader))
print(batch["image"].shape, batch["mask"].shape)
print(batch["filename"])
print("Training images:", len(train_data), "Training batches:", len(train_loader))
print("Validation images:", len(val_data), "Validation batches:", len(val_loader))

# %% 8. Trace one epoch and check edge cases
seen = []
for batch_number, batch in enumerate(train_loader, start=1):
    images, masks = batch["image"], batch["mask"]
    assert images.shape[0] == masks.shape[0]
    assert images.shape[-2:] == masks.shape[-2:]
    seen.extend(batch["filename"])
    print("Batch", batch_number, "contains", images.shape[0], "images")
assert len(seen) == len(set(seen)) == len(train_data)
assert dataset[0]["mask"].sum().item() == 0
print("One complete epoch and clean-image check passed")

# %% 9. Paired augmentations: sample once, apply together
def augment_pair(image, mask):
    if torch.rand(()).item() < 0.5:
        image = torch.flip(image, dims=(-1,))
        mask = torch.flip(mask, dims=(-1,))
    if torch.rand(()).item() < 0.5:
        image = torch.flip(image, dims=(-2,))
        mask = torch.flip(mask, dims=(-2,))
    brightness = 0.8 + 0.4 * torch.rand(()).item()
    image = (image * brightness).clamp(0, 1)
    return image, mask

# Separate Dataset objects keep validation free of random augmentation.
torch.manual_seed(42)
augmented_dataset = CloudDataset(ROOT, transform=augment_pair)
training = Subset(augmented_dataset, train_indices)
validation = Subset(CloudDataset(ROOT), val_indices)
augmented_loader = DataLoader(training, batch_size=4, shuffle=True, num_workers=0,
                             generator=torch.Generator().manual_seed(123))
augmented_batch = next(iter(augmented_loader))
assert set(augmented_batch["mask"].unique().tolist()) <= {0.0, 1.0}
assert torch.equal(validation[0]["image"], validation[0]["image"])
print("Augmented batch:", augmented_batch["image"].shape)

fig, axes = plt.subplots(1, 4, figsize=(12, 3))
for ax in axes:
    view = augmented_dataset[1]
    ax.imshow(view["image"].permute(1, 2, 0))
    foreground = view["mask"].any(dim=0).numpy()
    ax.contour(foreground, levels=[0.5], colors=["yellow"], linewidths=1)
    ax.set_title("Same filename, new transform")
    ax.axis("off")
plt.tight_layout()
plt.show()
