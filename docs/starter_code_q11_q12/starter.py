# %% [markdown]
# # Milestone 1: Questions 11 and 12 — Starter Code
# [Open in Google Colab](https://colab.research.google.com/drive/1Jf4f7jKddRSMpiZ1D8oVqmR8LDK-t86Q?usp=sharing)
# See README.md for Colab setup and output-saving instructions.
# Run cells in order. Set DATA_ROOT to the folder containing train.csv and train/.
# The supplied code prepares the holdout and training experiment. Complete the
# Question 11 calculations and Question 12 interpretation marked TODO.
# Use a GPU for the full 5–10 epoch experiment. No dataset or answers are bundled.

# %%
from pathlib import Path
import random

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp
from tqdm.auto import tqdm

DATA_ROOT = Path("data/public")  # EDIT for Kaggle or your local environment.
OUTPUT_DIR = Path("runs/milestone_1_starter")
CLASSES = ["defect_a", "defect_b", "defect_c", "defect_d"]
HEIGHT, WIDTH = 128, 800
VAL_FRACTION = 0.20
SEED = 42
EPOCHS = 5  # Question 12: use 5–10 epochs.
BATCH_SIZE = 8  # Reduce if your GPU runs out of memory.
ENCODER = "resnet34"  # Alternative: "efficientnet-b0".
ENCODER_WEIGHTS = "imagenet"  # First run needs internet to download weights.
THRESHOLDS = [0.3, 0.5, 0.7, 0.9]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("Device:", DEVICE)

# %% [markdown]
# ## Question 11: One row per image
# A CSV row describes one image–defect pair. Group all four rows before splitting.
# The class order stays fixed throughout loading, modeling, and evaluation.

# %%
annotations = pd.read_csv(DATA_ROOT / "train.csv", dtype=str, keep_default_na=False)
required = {"image_id", "defect_type", "mask_rle", "height", "width"}
if not required.issubset(annotations.columns):
    raise ValueError(f"Missing CSV columns: {required - set(annotations.columns)}")
if set(annotations["defect_type"]) != set(CLASSES):
    raise ValueError("Unexpected defect classes")
if annotations.duplicated(["image_id", "defect_type"]).any():
    raise ValueError("Duplicate image–class rows")
if not (annotations.groupby("image_id").size() == len(CLASSES)).all():
    raise ValueError("Every image must have one row for each of the four classes")
if not ((annotations["height"].astype(int) == HEIGHT)
        & (annotations["width"].astype(int) == WIDTH)).all():
    raise ValueError("Expected native height=128 and width=800")

annotations["mask_rle"] = annotations["mask_rle"].str.strip()
rle_table = annotations.pivot(
    index="image_id", columns="defect_type", values="mask_rle"
).reindex(columns=CLASSES).sort_index()
presence = rle_table.ne("")  # Boolean image × class table.


def combination_string(row):
    active_classes = [name for name in CLASSES if row[name]]
    return "+".join(active_classes) if active_classes else "clean"


image_table = presence.copy()
image_table["label_combo"] = presence.apply(combination_string, axis=1)
combination_counts = image_table["label_combo"].value_counts().sort_index()

# %% [markdown]
# ## TODO: Your Question 11 calculations
# Fill in the two expressions below using image_table, presence, or
# combination_counts. The idealized validation count allows fractional
# allocations before rounding. It is different from counting the images in the
# actual integer holdout created in the next cell.

# %%
n_combos = None  # TODO: number of distinct combination strings, including clean.
b_in_val_ideal = None  # TODO: ideal 20% defect_b-positive count, rounded once.
# Record your calculation and final short answer in your own working notes.

# %% [markdown]
# ## Create and save the actual 80/20 holdout
# Split image IDs, stratifying on the complete combination string. Actual sample
# counts are integers, so exact 80/20 proportions need not hold in every stratum.
# Do not repeatedly change the seed to find a more favorable validation score.
# If a combination occurs only once, inspect it and document a split policy;
# this starter deliberately reports that issue instead of silently splitting at random.

# %%
if combination_counts.min() < 2:
    rare = combination_counts[combination_counts < 2]
    raise ValueError(f"Singleton combinations need an explicit split policy:\n{rare}")
train_ids, val_ids = train_test_split(
    image_table.index.to_numpy(),
    test_size=VAL_FRACTION,
    random_state=SEED,
    stratify=image_table["label_combo"].to_numpy(),
)
assert not set(train_ids) & set(val_ids)
assert set(train_ids) | set(val_ids) == set(image_table.index)

split_table = image_table[["label_combo"]].copy()
split_table["partition"] = "train"
split_table.loc[val_ids, "partition"] = "validation"
split_table.to_csv(OUTPUT_DIR / "holdout.csv", index_label="image_id")
print("Train images:", len(train_ids), "Validation images:", len(val_ids))
# Optional: compare class prevalence in each partition. Do not infer the ideal
# Question 11 count solely from the rounded counts in this actual split.

# %% [markdown]
# ## Question 12: Decode targets and load image–mask pairs
# Masks use 1-based, column-major RLE. Clip run ends to the native canvas after
# checking the run structure. Audit the original endpoints separately for Q4.
# Images and masks stay at 128 × 800. This simple baseline uses no augmentation.

# %%
def decode_rle(rle):
    flat = np.zeros(HEIGHT * WIDTH, dtype=np.float32)
    if not str(rle).strip():
        return flat.reshape(HEIGHT, WIDTH, order="F")
    tokens = np.asarray(str(rle).split(), dtype=np.int64)
    if len(tokens) % 2:
        raise ValueError("RLE needs start/length pairs")
    starts, lengths = tokens[0::2] - 1, tokens[1::2]
    ends = starts + lengths
    if np.any(starts < 0) or np.any(starts >= flat.size) or np.any(lengths <= 0):
        raise ValueError("Invalid RLE start or length")
    if np.any(starts[1:] < ends[:-1]):
        raise ValueError("Unsorted or overlapping runs in one mask")
    for start, end in zip(starts, np.minimum(ends, flat.size)):
        flat[start:end] = 1.0
    return flat.reshape(HEIGHT, WIDTH, order="F")


class SurfaceDataset(Dataset):
    def __init__(self, image_ids, rle_table, data_root):
        self.image_ids = list(image_ids)
        self.rle_table = rle_table
        self.data_root = Path(data_root)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        filename = image_id if image_id.endswith(".jpg") else image_id + ".jpg"
        with Image.open(self.data_root / "train" / filename) as handle:
            image = np.asarray(handle.convert("L"), dtype=np.float32) / 255.0
        if image.shape != (HEIGHT, WIDTH):
            raise ValueError(f"Wrong image shape for {image_id}: {image.shape}")
        # Normalize independently per image; train and validation use the same rule.
        image = (image - image.mean()) / max(float(image.std()), 0.05)
        masks = np.stack([decode_rle(self.rle_table.loc[image_id, c]) for c in CLASSES])
        return torch.from_numpy(image[None].copy()), torch.from_numpy(masks.copy())


train_dataset = SurfaceDataset(train_ids, rle_table, DATA_ROOT)
val_dataset = SurfaceDataset(val_ids, rle_table, DATA_ROOT)
# workers=0 works in notebooks and on macOS without multiprocessing setup.
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
    pin_memory=DEVICE.type == "cuda", generator=torch.Generator().manual_seed(SEED),
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    pin_memory=DEVICE.type == "cuda",
)
image, masks = train_dataset[0]
assert image.shape == (1, HEIGHT, WIDTH)
assert masks.shape == (len(CLASSES), HEIGHT, WIDTH)
fig, axes = plt.subplots(1, 5, figsize=(18, 3))
axes[0].imshow(image[0], cmap="gray")
axes[0].set_title("Training image")
for c, axis in enumerate(axes[1:]):
    axis.imshow(masks[c], cmap="gray", vmin=0, vmax=1)
    axis.set_title(CLASSES[c])
for axis in axes:
    axis.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Baseline model and loss
# Q12 asks for sigmoid probabilities. Use four independent output channels;
# the model returns logits, and the loss applies sigmoid where appropriate.
# BCE supplies supervision on both positive and empty targets. The Dice term
# averages over nonempty image–class targets. No area filtering is used.

# %%
model = smp.Unet(
    encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS,
    in_channels=1, classes=len(CLASSES), activation=None,
).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
bce_loss = nn.BCEWithLogitsLoss()


def segmentation_loss(logits, targets):
    bce = bce_loss(logits, targets)
    probs = logits.sigmoid()
    intersection = (probs * targets).sum(dim=(-2, -1))
    denominator = probs.sum(dim=(-2, -1)) + targets.sum(dim=(-2, -1))
    soft_dice = (2 * intersection + 1e-6) / (denominator + 1e-6)
    positive = (targets.sum(dim=(-2, -1)) > 0).to(logits.dtype)
    dice_loss = ((1 - soft_dice) * positive).sum() / positive.sum().clamp_min(1)
    return 0.5 * bce + 0.5 * dice_loss


@torch.no_grad()
def validation_loss(model, loader):
    model.eval()
    total, samples = 0.0, 0
    for images, targets in loader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        loss = segmentation_loss(model(images), targets)
        total += loss.item() * len(images)
        samples += len(images)
    return total / samples

# %% [markdown]
# ## Train for 5–10 epochs
# Execute this cell when ready to start the full experiment. It saves the model
# with the lowest validation loss. The threshold sweep uses that same checkpoint
# for every threshold. Validation loss is a checkpoint criterion, not the Q8 score.

# %%
history = []
best_val_loss = float("inf")
checkpoint_path = OUTPUT_DIR / "best.pt"
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss, samples = 0.0, 0
    progress = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
    for images, targets in progress:
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        loss = segmentation_loss(model(images), targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(images)
        samples += len(images)
        progress.set_postfix(loss=running_loss / samples)
    val_loss = validation_loss(model, val_loader)
    history.append({"epoch": epoch, "train_loss": running_loss / samples, "val_loss": val_loss})
    print(history[-1])
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), checkpoint_path)
pd.DataFrame(history).to_csv(OUTPUT_DIR / "training_history.csv", index=False)
model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE, weights_only=True))

# %% [markdown]
# ## Threshold sweep on the complete validation holdout
# Accumulate sufficient statistics instead of storing all probability maps.
# Each image–class pair contributes equally to its relevant component average.
# The assessment_score below implements the explicit definitions in Question 8;
# it does not claim to reproduce an unpublished leaderboard implementation.
# Two false-positive measures are recorded: false pixels / canvas pixels on
# empty pairs, and the proportion of empty pairs with any predicted foreground.

# %%
@torch.no_grad()
def threshold_sweep(model, loader, thresholds):
    model.eval()
    totals = {tau: dict(pairs=0, positive=0, empty=0, dice_sum=0.0,
                       fg_dice_sum=0.0, fp_fraction_sum=0.0, correct_empty=0)
              for tau in thresholds}
    for images, targets in tqdm(loader, desc="Validation threshold sweep"):
        probs = model(images.to(DEVICE)).sigmoid()
        truth = targets.to(DEVICE) > 0.5
        gt_pixels = truth.sum(dim=(-2, -1))
        positive, empty = gt_pixels > 0, gt_pixels == 0
        for tau in thresholds:
            prediction = probs >= tau  # Raw probabilities; no component/area filtering.
            pred_pixels = prediction.sum(dim=(-2, -1))
            intersection = (prediction & truth).sum(dim=(-2, -1))
            denominator = pred_pixels + gt_pixels
            dice = torch.where(
                denominator == 0, torch.ones_like(denominator, dtype=torch.float32),
                2.0 * intersection / denominator.clamp_min(1),
            )
            total = totals[tau]
            total["pairs"] += truth.shape[0] * truth.shape[1]
            total["positive"] += int(positive.sum().item())
            total["empty"] += int(empty.sum().item())
            total["dice_sum"] += dice.sum().item()
            total["fg_dice_sum"] += dice[positive].sum().item()
            total["fp_fraction_sum"] += (pred_pixels[empty].float() / (HEIGHT * WIDTH)).sum().item()
            total["correct_empty"] += int((empty & (pred_pixels == 0)).sum().item())
    rows = []
    for tau, total in totals.items():
        if not total["positive"] or not total["empty"]:
            raise ValueError("The holdout needs both positive and empty image–class pairs")
        fg = total["fg_dice_sum"] / total["positive"]
        fp_fraction = total["fp_fraction_sum"] / total["empty"]
        empty_accuracy = total["correct_empty"] / total["empty"]
        rows.append({
            "threshold": tau,
            "naive_dice": total["dice_sum"] / total["pairs"],
            "foreground_dice": fg,
            "fp_pixel_fraction": fp_fraction,
            "fp_pair_rate": 1 - empty_accuracy,
            "false_positive_score": 1 - fp_fraction,
            "empty_accuracy": empty_accuracy,
            "assessment_score": 0.70 * fg + 0.20 * (1 - fp_fraction) + 0.10 * empty_accuracy,
            "holdout_empty_pair_fraction": total["empty"] / total["pairs"],
        })
    return pd.DataFrame(rows).sort_values("threshold")


results = threshold_sweep(model, val_loader, THRESHOLDS)
results.to_csv(OUTPUT_DIR / "threshold_sweep.csv", index=False)
print(results.to_string(index=False))
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for metric in ["naive_dice", "foreground_dice", "assessment_score"]:
    axes[0].plot(results["threshold"], results[metric], marker="o", label=metric)
axes[0].set_ylabel("Score")
axes[1].plot(results["threshold"], results["fp_pair_rate"], marker="o")
axes[1].set_ylabel("Empty pairs with any false positive (fraction)")
axes[2].plot(results["threshold"], results["fp_pixel_fraction"], marker="o")
axes[2].set_ylabel("Mean false-positive pixel fraction on empty pairs")
for axis in axes:
    axis.set_xlabel("Decision threshold")
    axis.set_xticks(THRESHOLDS)
    axis.grid(alpha=0.3)
axes[0].legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "threshold_sweep.png", dpi=160)
plt.show()

# %% [markdown]
# ## TODO: Write your Question 12 answer in bullet points
# - Record the model, seed, epochs, checkpoint selection and actual holdout size.
# - Describe the observed false-positive and Foreground Dice trends. Distinguish
#   a monotonic threshold property from the measured behavior of Dice.
# - Identify the peak threshold for each of the two aggregate scores. If several
#   thresholds tie, report the tie. Your experiment may produce coincident peaks.
# - Connect the difference (or lack of one) to the measured empty-pair fraction.
# - Explain any weak foreground performance using examples or per-class inspection.
#
# Resources: https://smp.readthedocs.io/en/latest/models.html
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
# https://docs.pytorch.org/docs/stable/data.html
