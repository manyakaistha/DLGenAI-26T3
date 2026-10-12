# %% [markdown]
# # Pixels, Masks & Experiments
# ## A tiny, honest W&B segmentation lab
# Train a three-layer CNN on 48 synthetic strips; validate on 16 different strips.
# Synthetic images are 200 × 32 (the same 6.25:1 aspect ratio as 800 × 128).
# Bright rectangles are deliberately easy to learn. This is a logging lesson, not
# a competition baseline. No data download, pretrained weights or GPU required.
# Run cells in order. Online mode needs a free W&B account and internet.
# Offline mode writes logs locally; it does not create a live dashboard.

# %% [markdown]
# ### 1. Setup (Colab / Jupyter)
# The next notebook cell installs the tested versions. Reconnect/restart the
# runtime if the notebook asks, then continue with the imports cell.
# Locally use the repository's uv environment, or:
# `uv run --with torch==2.8.0 --with numpy==2.2.6 --with matplotlib==3.10.9 --with wandb==0.23.1 wandb_tutorial.py`
# Never paste API keys into notebook source. `wandb.login()` prompts securely.

# %%
from pathlib import Path
import json
import time
import sys
if "ipykernel" not in sys.modules:
    import matplotlib
    matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import wandb

MODE = "offline"  # change to "online" for the live dashboard
if MODE == "online":
    wandb.login()
CONFIG = dict(seed=42, lr=0.01, batch_size=8, epochs=20, height=32, width=200,
              train_samples=48, val_samples=16, pos_weight=12.0, architecture="3-layer CNN",
              dataset="synthetic-bright-rectangles-v1", threshold=0.5)
assert (CONFIG["height"], CONFIG["width"]) == (32, 200), "This lesson generator uses fixed 200 × 32 geometry"
OUTPUT = Path("runs/live_session_demo")
OUTPUT.mkdir(parents=True, exist_ok=True)
torch.manual_seed(CONFIG["seed"])
torch.set_num_threads(2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
print("Versions:", {"torch": torch.__version__, "wandb": wandb.__version__, "numpy": np.__version__})

# %% [markdown]
# ### 2. A small dataset with a real signal
# The notebook uses fixed 200 × 32 strips (change the generator explicitly for
# other geometries). Separate seeds generate training and validation images. Every fourth strip is
# clean. We preserve background as label 0 and defect as label 1 in overlays.
# Students: why would unrelated random images and masks fail to teach learning?

# %%
def make_strips(n, seed):
    rng = np.random.default_rng(seed)
    images = rng.normal(0.25, 0.035, (n, 1, CONFIG["height"], CONFIG["width"])).astype("float32")
    masks = np.zeros_like(images)
    for i in range(n):
        if i % 4 == 0:
            continue
        y, x = int(rng.integers(2, 20)), int(rng.integers(5, 160))
        h, w = int(rng.integers(5, 11)), int(rng.integers(12, 35))
        masks[i, 0, y:y+h, x:x+w] = 1
        images[i, 0, y:y+h, x:x+w] += 0.5
    return torch.from_numpy(np.clip(images, 0, 1)), torch.from_numpy(masks)

train_x, train_y = make_strips(CONFIG["train_samples"], CONFIG["seed"])
val_x, val_y = make_strips(CONFIG["val_samples"], CONFIG["seed"] + 1000)
generator = torch.Generator().manual_seed(CONFIG["seed"])
loader = DataLoader(TensorDataset(train_x, train_y), batch_size=CONFIG["batch_size"],
                    shuffle=True, generator=generator)
model = nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.ReLU(),
                      nn.Conv2d(4, 4, 3, padding=1), nn.ReLU(),
                      nn.Conv2d(4, 1, 1)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([CONFIG["pos_weight"]], device=device))
fig, axes = plt.subplots(2, 1, figsize=(10, 3))
axes[0].imshow(train_x[1, 0], cmap="gray", vmin=0, vmax=1)
axes[1].imshow(train_y[1, 0], cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Synthetic input"); axes[1].set_title("Binary ground truth")
for ax in axes: ax.axis("off")
fig.tight_layout(); fig.savefig(OUTPUT / "dataset.png", dpi=140); plt.show(); plt.close(fig)

# %% [markdown]
# ### 3. Start a run: configuration, history, summary
# Config describes the recipe. History is the time series. Summary records the
# final/best outcomes. Separate custom axes avoid mixing batch steps and epochs.

# %%
run = wandb.init(project="pixels-masks-experiments", name=f"tiny-cnn-lr-{CONFIG['lr']}",
                 mode=MODE, config=CONFIG, dir=str(OUTPUT),
                 settings=wandb.Settings(x_disable_stats=True))
run.define_metric("batch_step")
run.define_metric("train/*", step_metric="batch_step")
run.define_metric("epoch")
run.define_metric("val/*", step_metric="epoch")
print("Dashboard:", run.url if MODE == "online" else "offline: sync after the session")
run.log({"dataset/example": wandb.Image(str(OUTPUT / "dataset.png"))})
run.summary["model_parameters"] = sum(p.numel() for p in model.parameters())
run.summary["all_background_pixel_accuracy"] = float((val_y == 0).float().mean())
run.summary["all_background_foreground_dice"] = 0.0

# %% [markdown]
# ### 4. Train and log measured validation metrics
# Hard Dice uses a 0.5 threshold. Foreground Dice averages only nonempty targets.
# Empty accuracy measures how often clean validation strips remain clean.
# Pixel accuracy can look high while foreground Dice is zero. These diagnostics
# are NOT the official competition score. Validation runs in eval/no_grad mode.

# %%
@torch.no_grad()
def evaluate():
    model.eval()
    logits = model(val_x.to(device))
    target = val_y.to(device)
    probs = logits.sigmoid()
    pred = probs >= CONFIG["threshold"]
    intersection = (pred * target).sum((1, 2, 3))
    denominator = pred.sum((1, 2, 3)) + target.sum((1, 2, 3))
    dice = torch.where(denominator > 0, 2 * intersection / denominator.clamp_min(1), 1.)
    positive = target.sum((1, 2, 3)) > 0
    empty_accuracy = (pred[~positive].sum((1, 2, 3)) == 0).float().mean()
    return dict(loss=criterion(logits, target).item(),
                foreground_dice=dice[positive].mean().item(),
                empty_accuracy=empty_accuracy.item(),
                foreground_recall=(intersection.sum() / target.sum().clamp_min(1)).item(),
                false_positive_pixels=int((pred & ~target.bool()).sum().item()),
                pixel_accuracy=(pred == target.bool()).float().mean().item()), probs.cpu(), dice.cpu()

best = -1.0
batch_step = 0
history = []
started = time.perf_counter()
for epoch in range(1, CONFIG["epochs"] + 1):
    model.train()
    for images, masks in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images.to(device)), masks.to(device))
        loss.backward()
        optimizer.step()
        batch_step += 1
        run.log({"batch_step": batch_step, "train/loss": loss.item(),
                 "train/lr": optimizer.param_groups[0]["lr"]})
    metrics, probs, dice = evaluate()
    history.append({"epoch": epoch, **metrics})
    run.log({"epoch": epoch, **{f"val/{k}": v for k, v in metrics.items()}})
    if metrics["foreground_dice"] > best:
        best = metrics["foreground_dice"]
        torch.save({"model_state": model.state_dict(), "config": CONFIG,
                    "epoch": epoch, "metrics": metrics,
                    "optimizer_state": optimizer.state_dict()}, OUTPUT / "best.pt")
    print(f"Epoch {epoch}: loss={metrics['loss']:.3f}, foreground Dice={metrics['foreground_dice']:.3f}")
print(f"Training + logging: {time.perf_counter() - started:.1f}s on {device}")

# %% [markdown]
# ### 5. Rich media: masks, tables, histograms and custom plots
# These predictions use the final epoch; the artifact stores the best epoch.
# Open the image overlay layers, sort the table by Dice, and inspect clean rows.
# Histograms show distributions; averages alone can hide failures.

# %%
labels = {0: "background", 1: "toy defect"}
table = wandb.Table(columns=["sample", "clean", "dice", "overlay"])
for i in range(len(val_x)):
    image = (val_x[i, 0].numpy() * 255).astype("uint8")
    overlay = wandb.Image(image, masks={
        "ground_truth": {"mask_data": val_y[i, 0].numpy().astype("uint8"), "class_labels": labels},
        "prediction": {"mask_data": (probs[i, 0].numpy() >= 0.5).astype("uint8"), "class_labels": labels}},
        caption=f"Held-out strip {i}; final epoch")
    table.add_data(i, bool(val_y[i].sum() == 0), float(dice[i]), overlay)
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot([h["epoch"] for h in history], [h["foreground_dice"] for h in history], marker="o")
ax.set(xlabel="Epoch", ylabel="Foreground Dice", ylim=(0, 1), title="Measured held-out learning")
fig.tight_layout()
fig.savefig(OUTPUT / "learning_curve.png", dpi=140)
run.log({"examples/final_validation": table,
         "diagnostics/probability_histogram": wandb.Histogram(probs.numpy()),
         "plots/learning_curve": wandb.Image(fig)})
plt.show(); plt.close(fig)
sweep = wandb.Table(columns=["threshold", "foreground_dice"])
for threshold in (0.2, 0.35, 0.5, 0.65, 0.8):
    pred = probs >= threshold
    positive = val_y.sum((1, 2, 3)) > 0
    numerator = 2 * (pred * val_y).sum((1, 2, 3))
    denominator = pred.sum((1, 2, 3)) + val_y.sum((1, 2, 3))
    sweep.add_data(threshold, float((numerator[positive] / denominator[positive]).mean()))
truth=val_y.numpy().astype(bool).ravel()
predicted=(probs.numpy() >= CONFIG["threshold"]).ravel()
confusion=np.array([[np.sum(~truth & ~predicted), np.sum(~truth & predicted)],
                    [np.sum(truth & ~predicted), np.sum(truth & predicted)]])
fig, ax = plt.subplots(figsize=(5, 4))
ax.imshow(confusion, cmap="Blues")
for y in range(2):
    for x in range(2):
        ax.text(x, y, f"{confusion[y,x]:,}", ha="center", va="center",
                color="white" if confusion[y,x] > confusion.max()/2 else "black")
ax.set(xticks=[0,1], yticks=[0,1], xticklabels=["BG", "defect"],
       yticklabels=["BG", "defect"], xlabel="Predicted", ylabel="True",
       title="Pixel confusion counts — final epoch")
fig.tight_layout(); fig.savefig(OUTPUT / "confusion.png", dpi=140)
run.log({"plots/confusion": wandb.Image(fig)})
plt.show(); plt.close(fig)
run.log({"plots/threshold_sweep": wandb.plot.line(sweep, "threshold", "foreground_dice", title="Validation threshold sensitivity")})

# %% [markdown]
# ### 6. Save, version and restore the best model
# A local checkpoint survives internet failure. An artifact connects that file to
# a run and recipe. Aliases such as `best` can move; version IDs such as `v0` are
# fixed. Upload completion is meaningful only in online mode.

# %%
(OUTPUT / "config.json").write_text(json.dumps(CONFIG, indent=2))
(OUTPUT / "metrics.json").write_text(json.dumps(history, indent=2))
artifact = wandb.Artifact("tiny-strip-segmenter", type="model",
                          metadata={"best_foreground_dice": best, "synthetic": True})
artifact.add_file(str(OUTPUT / "best.pt"))
artifact.add_file(str(OUTPUT / "config.json"))
artifact.add_file(str(OUTPUT / "metrics.json"))
logged = run.log_artifact(artifact, aliases=["best", "session-demo"])
run.summary["best_foreground_dice"] = best
run.summary["training_seconds"] = time.perf_counter() - started
if MODE == "online":
    logged.wait()
    print("Committed artifact:", logged.qualified_name)
else:
    print("Artifact recorded offline; upload after sync. Local checkpoint:", OUTPUT / "best.pt")
run.finish()
checkpoint = torch.load(OUTPUT / "best.pt", map_location=device, weights_only=True)
model.load_state_dict(checkpoint["model_state"])
restored_metrics, _, _ = evaluate()
assert abs(restored_metrics["foreground_dice"] - checkpoint["metrics"]["foreground_dice"]) < 1e-6
print("Restored best checkpoint verified:", restored_metrics)
(OUTPUT / "verification.json").write_text(json.dumps({"mode": MODE, "device": str(device),
    "best_foreground_dice": best, "restored_foreground_dice": restored_metrics["foreground_dice"],
    "restore_passed": True, "measured_history": history}, indent=2))

# %% [markdown]
# ### 7. Dashboard exercise and online artifact download
# Online: open the run URL → inspect loss/Dice → open the validation table →
# open Artifacts → inspect files, version, metadata and producing run.
# Repeat with lr=0.001 and a new run name; compare runs with the same split/seed.
# To download a committed artifact later (replace the path with your own):
# ```python
# api = wandb.Api()
# saved = api.artifact("YOUR_ENTITY/pixels-masks-experiments/tiny-strip-segmenter:v0")
# folder = saved.download(root="runs/restored_demo")
# checkpoint = torch.load(Path(folder) / "best.pt", map_location="cpu", weights_only=True)
# restored = nn.Sequential(nn.Conv2d(1,4,3,padding=1), nn.ReLU(),
#                          nn.Conv2d(4,4,3,padding=1), nn.ReLU(), nn.Conv2d(4,1,1))
# restored.load_state_dict(checkpoint["model_state"])
# restored.eval()
# with torch.no_grad():
#     prediction = restored(val_x[:1]).sigmoid() >= checkpoint["config"]["threshold"]
# ```
# Offline upload after the run finishes: `wandb sync runs/live_session_demo/wandb/offline-run-*`.
# System telemetry is disabled for portability; training/media logs still work.
# A saved optimizer state supports resuming, while inference only needs weights.
# Optional lineage: in a new run, call `run.use_artifact(full_versioned_path)`
# before downloading; this records which saved model the new run consumed.
# Logs may contain images: only upload data you are allowed to share.
# If online initialization fails, change MODE to offline and rerun from cell 1.
# Keep the split fixed; changing data and learning rate together prevents a fair comparison.
