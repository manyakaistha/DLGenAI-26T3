# %% [markdown]
# # Milestone 4: Synthetic Optimization & Training Improvements Walkthrough
#
# Welcome to the **Milestone 4 interactive walkthrough notebook**!
#
# In this standalone notebook, you will explore the mechanisms behind:
# 1. **Multi-Level Class Imbalance**: Managing clean strips and rare defect classes with sampling and loss weighting.
# 2. **Learning-Rate Schedules**: Implementing linear warmup and cosine decay using `torch.optim.lr_scheduler.LambdaLR`.
# 3. **Early Stopping & Checkpoint Governance**: Monitoring validation metrics, handling jitter, and restoring best weights.
# 4. **PyTorch 2.x AMP & Gradient Accumulation**: Simulating mixed precision (`torch.amp.autocast`, `torch.amp.GradScaler`) and microbatch accumulation.
# 5. **Metric Governance**: Resolving the "Naive Dice Trap" by separating foreground Dice from clean-image true negatives.
# 6. **Validation Error Auditing**: Decomposing errors into presence recall, overlap recall, boundary quality, and size bins.
#
# *Note: This entire notebook runs deterministically on CPU in seconds using 100% synthetic data.*

# %%
from __future__ import annotations

import copy
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# Set deterministic seed
torch.manual_seed(42)
np.random.seed(42)

CLASSES = ["defect_a", "defect_b", "defect_c", "defect_d"]
print(f"PyTorch Version: {torch.__version__}")
print("Initialized synthetic optimization workspace.")

# %% [markdown]
# ## 1. Multi-Level Imbalance: Combination-Group Sampling
#
# Groups are exact label combinations. Each image receives its group's per-image
# weight, and total group mass is image count times that weight. These toy counts
# are invented and separate from the assessment subset.

# %%
from collections import Counter

fictional_groups = ["clean"] * 16 + ["defect_a"] * 20 + ["defect_b"] * 4
counts = Counter(fictional_groups)
n_clean = counts["clean"]
group_weight = {g: min(math.sqrt(n_clean / n), 5.0) for g, n in counts.items()}
weight_sum = sum(counts[g] * group_weight[g] for g in counts)

print(f"{'group':<12} {'images':>8} {'weight':>9} {'draw share':>12}")
for group, n in counts.items():
    print(f"{group:<12} {n:>8} {group_weight[group]:>9.3f} {n * group_weight[group] / weight_sum:>11.1%}")

per_image = [group_weight[g] for g in fictional_groups]
sampler = WeightedRandomSampler(
    torch.tensor(per_image, dtype=torch.double), num_samples=400,
    replacement=True, generator=torch.Generator().manual_seed(42),
)
sample_count = Counter(fictional_groups[i] for i in sampler)
print(f"\nOne seeded 400-draw epoch: {dict(sample_count)}")
print("Finite replacement draws fluctuate around the expected group shares above.")

# %% [markdown]
# ## 2. Learning Rate Schedule: Linear Warmup + Cosine Annealing
#
# An industrial segmentation model often pairs a pretrained encoder with a randomly initialized decoder and head.
# A high initial learning rate may destabilize early optimization. Linear warmup reduces the scheduled rate early,
# followed by smooth cosine decay down to a minimum ratio (`min_lr_ratio = 0.01`).

# %%
def build_warmup_cosine_scheduler(optimizer, epochs, warmup_epochs=3, min_lr_ratio=0.01):
    def factor(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = min(1.0, (epoch - warmup_epochs) / max(1, epochs - warmup_epochs - 1))
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)

dummy_param = [nn.Parameter(torch.randn(5, 5))]
opt = torch.optim.AdamW(dummy_param, lr=1e-3)
scheduler = build_warmup_cosine_scheduler(opt, epochs=15, warmup_epochs=3, min_lr_ratio=0.01)

print(f"{'Epoch':<6} | {'Learning Rate':<15} | {'Phase':<20}")
print("-" * 45)
for ep in range(15):
    lr = opt.param_groups[0]["lr"]
    phase = "Linear Warmup" if ep < 3 else "Cosine Decay"
    print(f"{ep + 1:>5}  | {lr:>13.2e}   | {phase:<20}")
    opt.step()
    scheduler.step()

# %% [markdown]
# ## 3. Early Stopping & Checkpoint Rollback
#
# Early stopping terminates training when validation metrics stop improving.
# Critical components:
# - `mode='max'`: Higher is better (e.g. Foreground Dice).
# - `min_delta`: Prevents stopping on random metric noise.
# - `patience`: Window of allowable non-improving epochs.
# - `restore_best_weights`: Automatically reloads the peak checkpoint weights upon exit.

# %%
class EarlyStoppingTracker:
    def __init__(self, patience=4, min_delta=0.005, mode="max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.stale_epochs = 0
        self.best_score = None
        self.best_epoch = 0
        self.best_state = None
        self.should_stop = False

    def step(self, score: float, epoch: int, model: nn.Module) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.best_state = copy.deepcopy(model.state_dict())
            return True

        improved = (score > self.best_score + self.min_delta) if self.mode == "max" else (score < self.best_score - self.min_delta)
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.best_state = copy.deepcopy(model.state_dict())
            self.stale_epochs = 0
            return True
        else:
            self.stale_epochs += 1
            if self.stale_epochs >= self.patience:
                self.should_stop = True
            return False

# Simulate 12 epochs of validation Dice scores
toy_model = nn.Linear(4, 2)
stopper = EarlyStoppingTracker(patience=3, min_delta=0.005)
simulated_val_dice = [0.32, 0.45, 0.53, 0.58, 0.62, 0.64, 0.642, 0.638, 0.635, 0.620]

print(f"{'Epoch':<6} | {'Val Dice':<10} | {'Status':<30} | {'Stale':<8}")
print("-" * 60)
for ep, score in enumerate(simulated_val_dice, 1):
    new_best = stopper.step(score, ep, toy_model)
    msg = f"★ New Best ({score:.3f})" if new_best else "No significant gain"
    print(f"{ep:>5}  | {score:>8.3f}   | {msg:<30} | {stopper.stale_epochs:>3}/{stopper.patience}")
    if stopper.should_stop:
        print(f"\n-> Early stopping triggered at Epoch {ep}!")
        break

print(f"-> Restoring weights from Best Epoch: {stopper.best_epoch} (Dice = {stopper.best_score:.3f})")

# %% [markdown]
# ## 4. PyTorch 2.x Automatic Mixed Precision (AMP) Lifecycle
#
# Mixed precision computes tensor operations in FP16 (or BF16 on CPU/Ampere) to accelerate throughput and halve activation memory.
# In PyTorch 2.x, modern syntax uses `torch.amp.autocast` and `torch.amp.GradScaler`.

# %%
device = torch.device("cpu")  # CPU execution for educational portability
amp_dtype = torch.bfloat16

model = nn.Sequential(
    nn.Conv2d(1, 16, 3, padding=1),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    nn.Conv2d(16, 4, 1),
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scaler = torch.amp.GradScaler(device.type, enabled=False)  # enabled on CUDA

# Simulate microbatch accumulation (microbatch = 2, accumulation = 2 -> effective = 4)
microbatch = 2
accum_steps = 2
criterion = nn.BCEWithLogitsLoss()

optimizer.zero_grad(set_to_none=True)
for mb in range(accum_steps):
    x = torch.randn(microbatch, 1, 64, 200)
    y = (torch.rand(microbatch, 4, 64, 200) > 0.95).float()

    with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
        logits = model(x)
        loss = criterion(logits, y) / accum_steps

    scaler.scale(loss).backward()
    print(f"Microbatch {mb + 1}/{accum_steps} backward completed. Logits dtype: {logits.dtype}")

# Unscale, clip, step, update
scaler.unscale_(optimizer)
norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()
optimizer.zero_grad(set_to_none=True)
print(f"Optimizer step executed! Clipped gradient norm: {norm.item():.4f}")

# %% [markdown]
# ## 5. Metric Governance: Resolving the "Naive Dice Trap"
#
# Why does naive mean Dice mislead students?
# Because many segmentation datasets contain empty image-class pairs ($Y = 0$).
# When both ground truth and prediction are empty, the standard metric convention assigns $\text{Dice} = 1.0$.
# If most pairs are empty and the model predicts all zeros, a mean that counts
# empty-empty pairs as Dice 1 can look high while foreground Dice is zero.

# %%
def evaluate_dice_types(pred_masks: np.ndarray, gt_masks: np.ndarray):
    """pred_masks, gt_masks: shape [N, C, H, W] boolean arrays"""
    tp = (pred_masks & gt_masks).sum(axis=(-2, -1))
    pa = pred_masks.sum(axis=(-2, -1))
    ya = gt_masks.sum(axis=(-2, -1))

    total = pa + ya
    dice = np.divide(2.0 * tp, total, out=np.ones_like(total, dtype=float), where=total > 0)

    is_positive = ya > 0
    is_empty = ya == 0

    naive_mean_dice = float(dice.mean())
    fg_dice = float(dice[is_positive].mean()) if is_positive.any() else 0.0
    empty_accuracy = float((pa[is_empty] == 0).mean()) if is_empty.any() else 1.0
    fp_pixel_score = float((1.0 - pa[is_empty] / (32 * 100)).mean()) if is_empty.any() else 1.0

    # Exact 70/20/10 component decomposition, with this toy canvas size.
    weighted_proxy = 0.70 * fg_dice + 0.20 * fp_pixel_score + 0.10 * empty_accuracy

    return {
        "naive_mean_dice": naive_mean_dice,
        "foreground_dice": fg_dice,
        "empty_accuracy": empty_accuracy,
        "false_positive_pixel_score": fp_pixel_score,
        "weighted_proxy": weighted_proxy,
    }

# Synthetic scenario: predictions have partial overlap and a few false blobs.
n_pairs = 100
gt_sim = np.zeros((n_pairs, 4, 32, 100), dtype=bool)
# Ten pairs have toy targets. Predictions overlap most, but not all, target pixels.
gt_sim[:10, 0, 10:20, 10:20] = True
toy_pred = np.zeros_like(gt_sim)
toy_pred[:10, 0, 10:18, 10:20] = True
# A few empty pairs contain small false blobs.
toy_pred[10:30, 0, 2:4, 2:12] = True

metrics = evaluate_dice_types(toy_pred, gt_sim)
print("Evaluation of a synthetic partial-overlap prediction:")
print(f"  • Naive Mean Dice:        {metrics['naive_mean_dice']:.4f}")
print(f"  • Foreground-Pair Dice:   {metrics['foreground_dice']:.4f}")
print(f"  • Empty-Pair Accuracy:    {metrics['empty_accuracy']:.4f}")
print(f"  • False-Positive Pixel:   {metrics['false_positive_pixel_score']:.4f}")
print(f"  • Official 70/20/10 score:{metrics['weighted_proxy']:.4f}")

# %% [markdown]
# ## 6. Validation Error Auditing & Spatial Diagnostics
#
# Moving beyond aggregate scores:
# - Separate **presence detection** ($pa > 0$) from **overlap hit** ($tp > 0$).
# - Size-stratify recall (Small 1..100 px, Medium 101..500 px, Large > 500 px).

# %%
def audit_spatial_errors():
    # Simulate 30 pairs with known error categories
    results = []
    for i in range(30):
        gt_area = (i + 1) * 25  # areas from 25 px to 750 px
        if i < 8:
            # Small defects: completely missed
            pred_area, tp = 0, 0
        elif i < 18:
            # Medium defects: detected with boundary shift
            pred_area = gt_area + 30
            tp = int(gt_area * 0.70)
        else:
            # Large defects: strong segmentation
            pred_area = gt_area
            tp = int(gt_area * 0.92)

        dice = (2.0 * tp) / (pred_area + gt_area) if pred_area + gt_area > 0 else 0.0
        results.append({
            "gt_area": gt_area,
            "pred_area": pred_area,
            "tp": tp,
            "dice": dice,
            "hit": tp > 0,
        })

    # Bin by size
    bins = {
        "Small (1..100 px)": [r for r in results if r["gt_area"] <= 100],
        "Medium (101..500 px)": [r for r in results if 100 < r["gt_area"] <= 500],
        "Large (>500 px)": [r for r in results if r["gt_area"] > 500],
    }

    print(f"{'Defect Size Category':<22} | {'Count':<6} | {'Overlap Recall':<15} | {'Mean Dice':<10}")
    print("-" * 62)
    for name, group in bins.items():
        rec = sum(r["hit"] for r in group) / len(group)
        mean_d = float(np.mean([r["dice"] for r in group]))
        print(f"{name:<22} | {len(group):>5}  | {rec:>13.1%} | {mean_d:>9.4f}")

audit_spatial_errors()

# %% [markdown]
# ## 7. Summary & The Controlled Experimentation Checklist
#
# When optimizing your industrial surface segmentation models:
# 1. **Change One Factor at a Time**: Never alter learning rate, sampling, and loss simultaneously.
# 2. **Freeze Baseline Invariants**: Keep fold splits, data seeds, evaluation thresholds, and image dimensions ($128 \times 800$) fixed.
# 3. **Monitor Both Sides of Imbalance**: Check that gains in rare-defect recall do not cause clean-image false alarms to surge.
# 4. **Govern Checkpoints with Foreground Dice**: Use foreground-pair Dice or the competition proxy for early stopping and model selection.
# 5. **Audit Errors Spatially**: Inspect whether errors stem from missed detection, speckle noise, or boundary misalignment.
