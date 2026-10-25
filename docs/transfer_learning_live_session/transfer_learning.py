# /// script
# requires-python = ">=3.10"
# dependencies = ["torch==2.8.0", "torchvision==0.23.0", "numpy==2.2.6", "Pillow==12.3.0", "matplotlib==3.10.9"]
# ///
# %% [markdown]
# # From a baseline U-Net to a pretrained encoder
# Live session · 26 October 2026. Use the real LGG TIFF dataset, split patients,
# then compare identical ResNet18 U-Nets with random versus ImageNet encoders.
# Dataset: https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation
# Attribution: Mateusz Buda / TCIA / TCGA-LGG; Kaggle lists CC BY-NC-SA 4.0.
# Target: binary FLAIR abnormality, not tumor grade. Educational use, not diagnosis.
# Download/extract or attach the dataset separately. No patient data is bundled.
# A short classroom subset is not a benchmark. Read notebook.md before running.

# %%
import copy
import json
import os
import platform
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import PIL
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models import resnet18, ResNet18_Weights

# %% [markdown]
# ## 1. Configure the experiment
# Edit these values directly in a notebook, or set environment variables for the
# script. Use the path shown in Kaggle's input panel; mount spellings can differ.
# Caps apply AFTER patient splitting; 0 means unlimited. The classroom defaults
# deliberately use a small deterministic subset, with no mask-based slice selection.
# Both arms fully fine-tune with the same loss, normalization and training policy.

# %%
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/kaggle/input/lgg-mri-segmentation"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "runs/mri_transfer_session"))
IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", 128))
EPOCHS = int(os.environ.get("EPOCHS", 2))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 4))
MAX_TRAIN_PATIENTS = int(os.environ.get("MAX_TRAIN_PATIENTS", 12))
MAX_VAL_PATIENTS = int(os.environ.get("MAX_VAL_PATIENTS", 4))
MAX_SLICES_PER_PATIENT = int(os.environ.get("MAX_SLICES_PER_PATIENT", 12))
INPUT_MODE = os.environ.get("INPUT_MODE", "sequences")
ARMS = os.environ.get("ARMS", "random,imagenet").split(",")
SEED = 42
LR = 1e-3
THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
# Bound CPU threads for portable, small classroom batches; this is not GPU tuning.
torch.set_num_threads(min(4, os.cpu_count() or 1))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


# %% [markdown]
# ## 2. Index files and split PATIENTS
# Matching the mask suffix is safer than independently sorting image/mask lists.
# Patient identity comes from the complete case directory, not the slice number.
# Fail on duplicate extraction trees, inconsistent names and unpaired images.

# %%
def read_index(root):
    if not root.is_dir():
        raise FileNotFoundError(f"Set DATA_ROOT to extracted LGG TIFF folders: {root}")
    masks = sorted(root.rglob("*_mask.tif"))
    if not masks:
        raise ValueError(f"No *_mask.tif files below {root}")
    records, keys, paired = [], set(), set()
    for mask in masks:
        image = mask.with_name(mask.stem[:-5] + ".tif")
        if not image.is_file():
            raise FileNotFoundError(f"Missing image for {mask}")
        patient = mask.parent.name
        prefix = patient + "_"
        if not image.stem.startswith(prefix):
            raise ValueError(f"Case directory/name mismatch: {image}")
        slice_text = image.stem[len(prefix):]
        if not slice_text.isdigit():
            raise ValueError(f"Expected numeric slice suffix: {image}")
        key = (patient, int(slice_text))
        if key in keys:
            raise ValueError(f"Duplicate patient/slice; check duplicate extractions: {key}")
        keys.add(key)
        paired.add(image.resolve())
        records.append(dict(image=str(image), mask=str(mask), patient=patient,
                            slice=int(slice_text)))
    all_images = {p.resolve() for p in root.rglob("*.tif") if not p.stem.endswith("_mask")}
    if paired != all_images:
        raise ValueError(f"Found {len(all_images - paired)} unpaired TIFF images")
    return sorted(records, key=lambda r: (r["patient"], r["slice"]))


def split_patients(records, seed=SEED):
    patients = sorted({r["patient"] for r in records})
    if len(patients) < 2:
        raise ValueError("At least two distinct patients are required")
    np.random.default_rng(seed).shuffle(patients)
    n_val = min(len(patients)-1, max(1, int(np.ceil(0.2 * len(patients)))))
    train_ids, val_ids = sorted(patients[n_val:]), sorted(patients[:n_val])
    assert set(train_ids).isdisjoint(val_ids)
    assert set(train_ids) | set(val_ids) == set(patients)
    return train_ids, val_ids


def select_records(records, ids, patient_cap, slice_cap):
    ids = np.random.default_rng(SEED + 1).permutation(sorted(ids)).tolist()
    ids = sorted(ids[:patient_cap or None])
    selected = []
    for patient in ids:
        group = [r for r in records if r["patient"] == patient]
        if slice_cap and len(group) > slice_cap:
            indices = np.linspace(0, len(group)-1, slice_cap, dtype=int)
            group = [group[i] for i in indices]
        selected.extend(group)
    return selected


# %% [markdown]
# ## 3. Return the right tensor contract
# Sequence order: pre-contrast, FLAIR, post-contrast; missing sequences can be
# replaced with FLAIR in the supplied dataset. Do not call this RGB anatomy.
# This loader explicitly supports the inspected uint8 format and hard binary masks.
# Resize the WHOLE field, bilinear image / nearest mask. No target-guided cropping.
# Fixed ImageNet normalization is a starting convention, not MRI-derived statistics.

# %%
class MRIDataset(Dataset):
    def __init__(self, records, size=IMAGE_SIZE, input_mode=INPUT_MODE):
        if input_mode not in ("sequences", "flair_repeat"):
            raise ValueError("INPUT_MODE must be sequences or flair_repeat")
        self.records, self.size, self.input_mode = records, size, input_mode

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(record["image"]) as source:
            image = np.array(source)
        with Image.open(record["mask"]) as source:
            mask = np.array(source)
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError(f"Expected uint8 H,W,3 TIFF: {record['image']}")
        if mask.ndim != 2 or mask.shape != image.shape[:2]:
            raise ValueError(f"Image/mask shape mismatch: {record['mask']}")
        if not set(np.unique(mask)).issubset({0, 1, 255}):
            raise ValueError(f"Expected a binary mask: {record['mask']}")
        if self.input_mode == "flair_repeat":
            image = np.repeat(image[..., 1:2], 3, axis=2)
        x = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.0
        y = torch.from_numpy((mask > 0).astype(np.float32))[None]
        x = F.interpolate(x[None], size=(self.size, self.size), mode="bilinear",
                          align_corners=False)[0]
        y = F.interpolate(y[None], size=(self.size, self.size), mode="nearest")[0]
        x = (x - MEAN) / STD
        return dict(image=x, mask=y, patient=record["patient"], path=record["image"])


# %% [markdown]
# ## 4. A small, explicit ResNet18 U-Net
# The encoder exposes five spatial features. Its classification pool/head are not
# retained. The new decoder upsamples to each skip's exact size and concatenates
# channels. The final head returns LOGITS, with the same H,W as the input.

# %%
class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))


class Encoder(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        backbone = resnet18(weights=weights)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.layer1, self.layer2 = backbone.layer1, backbone.layer2
        self.layer3, self.layer4 = backbone.layer3, backbone.layer4

    def forward(self, x):
        s1 = self.stem(x)
        s2 = self.layer1(self.pool(s1))
        s3 = self.layer2(s2)
        s4 = self.layer3(s3)
        s5 = self.layer4(s4)
        return [s1, s2, s3, s4, s5]


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([
            ConvBlock(512+256, 256), ConvBlock(256+128, 128),
            ConvBlock(128+64, 64), ConvBlock(64+64, 32)])
        self.refine = ConvBlock(32, 16)
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, features, size):
        x = features[-1]
        for block, skip in zip(self.blocks, reversed(features[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skip], dim=1))
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return self.head(self.refine(x))


class ResNetUNet(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.encoder = Encoder(weights)
        self.decoder = Decoder()

    def forward(self, x):
        return self.decoder(self.encoder(x), x.shape[-2:])


def configure_training(model, frozen=False, lr=LR):
    # Rebuilding the optimizer intentionally resets its state for this policy.
    model.zero_grad(set_to_none=True)
    for p in model.encoder.parameters():
        p.requires_grad_(not frozen)
    return torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                             lr=lr, weight_decay=1e-4)


# %% [markdown]
# ## 5. Sanity checks, then limited grayscale equivalence
# The equality below applies only to an unnormalized repeated grayscale input at
# one convolution. Channel-specific normalization changes the premise.

# %%
def check_grayscale_identity():
    seed_all(SEED)
    conv3 = nn.Conv2d(3, 4, 3, padding=1, bias=True)
    conv1 = nn.Conv2d(1, 4, 3, padding=1, bias=True)
    with torch.no_grad():
        conv1.weight.copy_(conv3.weight.sum(dim=1, keepdim=True))
        conv1.bias.copy_(conv3.bias)
        gray = torch.rand(2, 1, 16, 16)
        error = (conv3(gray.repeat(1, 3, 1, 1)) - conv1(gray)).abs().max().item()
    assert error < 1e-5
    print(f"Unnormalized first-convolution sum identity: max error {error:.2e}")


def inspect_data(dataset, out):
    sample = dataset[0]
    x, y = sample["image"], sample["mask"]
    assert x.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert y.shape == (1, IMAGE_SIZE, IMAGE_SIZE)
    assert set(y.unique().tolist()).issubset({0.0, 1.0})
    assert torch.isfinite(x).all()
    image = (x * STD + MEAN).clamp(0, 1).numpy()
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    names = ["Pre-contrast", "FLAIR", "Post-contrast"] if INPUT_MODE == "sequences" else ["FLAIR copy 1", "FLAIR copy 2", "FLAIR copy 3"]
    for c, name in enumerate(names):
        axes[c].imshow(image[c], cmap="gray", vmin=0, vmax=1)
        axes[c].set_title(name)
    axes[3].imshow(y[0], cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("Binary target")
    for axis in axes: axis.axis("off")
    fig.suptitle(f"Data inspection — {Path(sample['path']).name}")
    fig.tight_layout(); fig.savefig(out / "data_inspection.png", dpi=120); plt.close(fig)
    print("Sample:", x.shape, y.shape, "patient:", sample["patient"])


# %% [markdown]
# ## 6. Loss, hard metrics and one training epoch
# Dice loss averages per-sample spatial soft Dice. Validation reports equal-weight
# patient-volume Dice over the SELECTED slices, positive-slice Dice and an explicit
# any-pixel false-positive rate on empty slices. Missing subgroups are None/null.

# %%
def bce_dice(logits, target):
    probability = logits.sigmoid()
    dims = (1, 2, 3)
    intersection = (probability * target).sum(dims)
    denominator = probability.sum(dims) + target.sum(dims)
    soft_dice = (2*intersection + 1e-6) / (denominator + 1e-6)
    return F.binary_cross_entropy_with_logits(logits, target) + (1-soft_dice).mean()


def hard_dice(tp, fp, fn):
    denominator = 2*tp + fp + fn
    return 2*tp/denominator if denominator else 1.0


def train_epoch(model, loader, optimizer, frozen=False):
    model.train()
    if frozen:
        model.encoder.eval()  # after model.train(), every epoch
    total, count = 0.0, 0
    for batch in loader:
        x, y = batch["image"].to(DEVICE), batch["mask"].to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        loss = bce_dice(model(x), y)
        if not torch.isfinite(loss): raise RuntimeError("Non-finite training loss")
        loss.backward(); optimizer.step()
        total += loss.item() * len(x); count += len(x)
    return total / count


@torch.inference_mode()
def evaluate(model, loader):
    model.eval()
    total, count, empty_count, empty_fp = 0.0, 0, 0, 0
    patients, positive_scores = {}, []
    for batch in loader:
        x, y = batch["image"].to(DEVICE), batch["mask"].to(DEVICE)
        logits = model(x)
        total += bce_dice(logits, y).item() * len(x); count += len(x)
        predicted, target = (logits.sigmoid() >= THRESHOLD).cpu(), (y > 0.5).cpu()
        for i, patient in enumerate(batch["patient"]):
            p, t = predicted[i], target[i]
            tp = int((p & t).sum()); fp = int((p & ~t).sum()); fn = int((~p & t).sum())
            counts = patients.setdefault(patient, [0, 0, 0])
            for j, value in enumerate((tp, fp, fn)): counts[j] += value
            if bool(t.any()): positive_scores.append(hard_dice(tp, fp, fn))
            else:
                empty_count += 1; empty_fp += int(bool(p.any()))
    per_patient = {pid: hard_dice(*values) for pid, values in patients.items()}
    return dict(loss=total/count, patient_dice=float(np.mean(list(per_patient.values()))),
                foreground_slice_dice=float(np.mean(positive_scores)) if positive_scores else None,
                empty_slice_fp_rate=empty_fp/empty_count if empty_count else None,
                foreground_slices=len(positive_scores), empty_slices=empty_count,
                per_patient_dice=per_patient)


def save_curves(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    epochs = [h["epoch"] for h in history]
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="Train")
    axes[0].plot(epochs, [h["validation"]["loss"] for h in history], label="Validation")
    axes[0].set_title("BCE + soft Dice loss"); axes[0].legend()
    axes[1].plot(epochs, [h["validation"]["patient_dice"] for h in history], marker="o")
    axes[1].set_title("Mean selected-patient volume Dice"); axes[1].set_ylim(0, 1)
    for axis in axes: axis.set_xlabel("Epoch"); axis.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


@torch.inference_mode()
def save_predictions(model, dataset, path):
    # Select by target and sorted validation order, never by a model's score.
    selected, seen = [], set()
    for i in range(len(dataset)):
        sample = dataset[i]; kind = bool(sample["mask"].any())
        if kind not in seen: selected.append(sample); seen.add(kind)
        if len(selected) == 2: break
    model.eval()
    fig, axes = plt.subplots(len(selected), 4, figsize=(11, 3*len(selected)), squeeze=False)
    for row, sample in enumerate(selected):
        x, target = sample["image"], sample["mask"][0].numpy()
        probability = model(x[None].to(DEVICE)).sigmoid()[0, 0].cpu().numpy()
        flair = (x * STD + MEAN)[1].numpy()
        views = [flair, target, probability >= THRESHOLD, probability]
        titles = [Path(sample["path"]).name, "Reference", f"Prediction ≥ {THRESHOLD}", "Probability overlay"]
        for col, (view, title) in enumerate(zip(views, titles)):
            axis = axes[row, col]
            if col == 3:
                axis.imshow(flair, cmap="gray", vmin=0, vmax=1)
                axis.imshow(view, cmap="magma", vmin=0, vmax=1, alpha=.55)
            else: axis.imshow(view, cmap="gray", vmin=0, vmax=1)
            axis.set_title(title, fontsize=8); axis.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# %% [markdown]
# ## 7. Run both initialization arms
# Decoder/head initialization is explicitly shared. Both arms train ALL parameters
# with the same policy. The ImageNet download must succeed; no silent fallback.
# Checkpoints and output artifacts stay local. Use a new OUTPUT_DIR for each run.

# %%
def run_experiment():
    if IMAGE_SIZE < 64 or IMAGE_SIZE % 32 or EPOCHS < 1 or BATCH_SIZE < 1:
        raise ValueError("Use IMAGE_SIZE >=64 divisible by 32; positive epochs/batch size")
    if min(MAX_TRAIN_PATIENTS, MAX_VAL_PATIENTS, MAX_SLICES_PER_PATIENT) < 0:
        raise ValueError("Caps must be nonnegative; 0 means unlimited")
    if not ARMS or len(set(ARMS)) != len(ARMS) or any(a not in ("random", "imagenet") for a in ARMS):
        raise ValueError("ARMS must contain unique random and/or imagenet entries")
    records = read_index(DATA_ROOT)
    train_ids, val_ids = split_patients(records)
    train_records = select_records(records, train_ids, MAX_TRAIN_PATIENTS, MAX_SLICES_PER_PATIENT)
    val_records = select_records(records, val_ids, MAX_VAL_PATIENTS, MAX_SLICES_PER_PATIENT)
    assert set(r["patient"] for r in train_records).isdisjoint(r["patient"] for r in val_records)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = dict(all_train_patients=train_ids, all_validation_patients=val_ids,
                    selected_train=train_records, selected_validation=val_records)
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    config = {name: globals()[name] for name in ["IMAGE_SIZE", "EPOCHS", "BATCH_SIZE", "MAX_TRAIN_PATIENTS", "MAX_VAL_PATIENTS", "MAX_SLICES_PER_PATIENT", "INPUT_MODE", "ARMS", "SEED", "LR", "THRESHOLD"]}
    config.update(data_root=str(DATA_ROOT), device=str(DEVICE), torch=torch.__version__,
                  torchvision=torchvision.__version__, numpy=np.__version__, pillow=PIL.__version__,
                  python=platform.python_version(), encoder_weights="ResNet18_Weights.IMAGENET1K_V1",
                  normalization_mean=MEAN.flatten().tolist(), normalization_std=STD.flatten().tolist(),
                  training_policy="full fine-tuning; no random augmentation; AdamW wd=1e-4",
                  loss="BCEWithLogits + mean per-sample soft Dice, epsilon=1e-6",
                  metric="mean equal-patient volume Dice over selected validation slices")
    (OUTPUT_DIR / "configuration.json").write_text(json.dumps(config, indent=2))
    print(f"{len(records)} indexed slices; {len(train_ids)}/{len(val_ids)} train/val patients before caps")
    print(f"Selected {len(train_records)} train and {len(val_records)} validation slices; device={DEVICE}")
    train_data, val_data = MRIDataset(train_records), MRIDataset(val_records)
    inspect_data(train_data, OUTPUT_DIR); check_grayscale_identity()
    seed_all(SEED)
    reference = ResNetUNet()
    decoder_start = copy.deepcopy(reference.decoder.state_dict())
    reference.eval()
    with torch.no_grad():
        x = train_data[0]["image"][None]
        print("Encoder shapes:", [list(f.shape) for f in reference.encoder(x)])
        assert reference(x).shape == (1, 1, IMAGE_SIZE, IMAGE_SIZE)
    del reference
    comparison = {}
    for arm in ARMS:
        seed_all(SEED)
        weights = ResNet18_Weights.IMAGENET1K_V1 if arm == "imagenet" else None
        model = ResNetUNet(weights)
        model.decoder.load_state_dict(decoder_start)
        assert all(torch.equal(v, decoder_start[k]) for k, v in model.decoder.state_dict().items())
        model.to(DEVICE)
        optimizer = configure_training(model)
        # Fresh generator and identical sample set/order policy for each arm.
        seed_all(SEED)
        loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                            generator=torch.Generator().manual_seed(SEED))
        valid_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        arm_dir = OUTPUT_DIR / arm; arm_dir.mkdir(exist_ok=True)
        history = []; start = time.perf_counter()
        for epoch in range(1, EPOCHS+1):
            train_loss = train_epoch(model, loader, optimizer)
            metrics = evaluate(model, valid_loader)
            history.append(dict(epoch=epoch, train_loss=train_loss, validation=metrics))
            print(arm, "epoch", epoch, "train loss", round(train_loss, 4), "patient Dice", round(metrics["patient_dice"], 4), flush=True)
        if DEVICE.type == "cuda": torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        (arm_dir / "history.json").write_text(json.dumps(history, indent=2))
        final = dict(arm=arm, seconds_train_and_validation=elapsed, final_epoch=metrics)
        (arm_dir / "metrics.json").write_text(json.dumps(final, indent=2))
        # Save the FINAL epoch equally, not an arm-specific best-epoch selection.
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save(dict(model_state=state, configuration=config, arm=arm), arm_dir / "final.pt")
        save_curves(history, arm_dir / "curves.png")
        save_predictions(model, val_data, arm_dir / "predictions.png")
        comparison[arm] = final
        del state, model, optimizer, loader, valid_loader
        if DEVICE.type == "cuda": torch.cuda.empty_cache()
    (OUTPUT_DIR / "comparison.json").write_text(json.dumps(comparison, indent=2))
    print("Saved teaching artifacts to", OUTPUT_DIR.resolve())
    print("Interpret your measured comparison; a short capped run is not a clinical benchmark.")
    return comparison


if __name__ == "__main__":
    results = run_experiment()
