"""Generate high-resolution conceptual figures for Milestone 4 student guides using synthetic data."""
from __future__ import annotations

import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import numpy as np

OUTPUT_DIR = Path("docs/milestone_4_student_guide/assets")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Aesthetic matching the Type Garden / student guide editorial styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "figure.titleweight": "bold",
})


def make_fig01_imbalance_three_levels():
    """Fig 1: Deconstruct class imbalance into image-level, class-level, and pixel-level."""
    fig = plt.figure(figsize=(14, 8), dpi=150)
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1.2], wspace=0.3, hspace=0.35)

    # 1. Image-level imbalance
    ax1 = fig.add_subplot(gs[0, 0])
    labels = ["Clean Strips\n(No Defects)", "Defective Strips\n(1+ Defects)"]
    sizes = [52, 48]
    colors = ["#2b6cb0", "#c57924"]
    wedges, texts, autotexts = ax1.pie(
        sizes, labels=labels, autopct="%1.0f%%", startangle=140, colors=colors,
        textprops=dict(color="#172d3a", fontsize=10, weight="bold")
    )
    for at in autotexts:
        at.set_color("white")
    ax1.set_title("Level 1: Image-Level Imbalance\n(Clean vs. Defective Strips)", fontsize=11)

    # 2. Class-level imbalance
    ax2 = fig.add_subplot(gs[0, 1])
    classes = ["Defect A\n(Scratch)", "Defect B\n(Micro-crack)", "Defect C\n(Inclusion)", "Defect D\n(Patch)"]
    counts = [950, 280, 3600, 1500]
    bars = ax2.bar(classes, counts, color=["#319795", "#e53e3e", "#dd6b20", "#805ad5"], width=0.55)
    ax2.set_ylabel("Fictional Positive Count")
    ax2.set_title("Level 2: Class-Level Imbalance\n(Frequent vs. Rare Defects)", fontsize=11)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 80, f"{h}", ha="center", fontsize=9, weight="bold")
    ax2.set_ylim(0, 4200)

    # 3. Pixel-level imbalance
    ax3 = fig.add_subplot(gs[0, 2])
    px_labels = ["Background\n(98.2%)", "Defect Pixels\n(1.8%)"]
    px_sizes = [98.2, 1.8]
    px_colors = ["#cbd5e0", "#e53e3e"]
    ax3.pie(
        px_sizes, labels=px_labels, autopct="%1.1f%%", startangle=15, colors=px_colors,
        explode=(0, 0.2), textprops=dict(color="#172d3a", fontsize=10, weight="bold")
    )
    ax3.set_title("Level 3: Pixel-Level Imbalance\n(Defect vs. Canvas Area)", fontsize=11)

    # Bottom: Synthetic Strip breakdown
    ax4 = fig.add_subplot(gs[1, :])
    np.random.seed(101)
    strip = np.full((128, 800), 160, dtype=np.int16)
    strip += np.random.normal(0, 8, (128, 800)).astype(np.int16)
    # Rare defect: tiny micro-crack
    strip[60:64, 380:420] = 30
    strip = np.clip(strip, 0, 255).astype(np.uint8)

    ax4.imshow(strip, cmap="gray", aspect="auto", vmin=0, vmax=255)
    ax4.set_title("Synthetic Strip: Rare Defect B on an $800 \\times 128$ Canvas (102,400 Pixels)", fontsize=12)
    ax4.set_xlabel("Width: 800 Pixels (Rolling Direction)")
    ax4.set_ylabel("Height: 128 Pixels")

    # Bounding box around tiny defect
    rect = patches.Rectangle((370, 50), 60, 28, linewidth=2, edgecolor="#e53e3e", facecolor="none")
    ax4.add_patch(rect)
    ax4.text(370, 42, "Rare Defect B: 160 px (~0.15% of strip)", color="#e53e3e", fontsize=10, weight="bold")
    ax4.text(10, 20, "Background Canvas: 102,240 pixels (99.85%)", color="#2b6cb0", fontsize=10, weight="bold")

    fig.suptitle("Synthetic Example: Three Levels of Class Imbalance", fontsize=14, y=0.98)
    fig.savefig(OUTPUT_DIR / "fig01_imbalance_three_levels.png", bbox_inches="tight")
    plt.close(fig)


def make_fig02_sampling_vs_weighting():
    """Fig 2: Compare sampling vs loss weighting mechanics and their impact on training prior."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=160, gridspec_kw={"width_ratios": [1.05, 1.35]})

    # Sampling mechanics
    cats = ["Clean", "Defect A", "Defect B (Rare)", "Defect C"]
    natural_freq = [0.50, 0.10, 0.03, 0.37]
    oversampled_freq = [0.25, 0.20, 0.25, 0.30]

    x = np.arange(len(cats))
    w = 0.35
    ax1.bar(x - w/2, natural_freq, w, label="Natural Distribution (Evaluation Prior)", color="#4a5568")
    ax1.bar(x + w/2, oversampled_freq, w, label="Heavily Oversampled (Training Prior)", color="#dd6b20")
    ax1.set_xticks(x)
    ax1.set_xticklabels(cats)
    ax1.set_ylabel("Encounter Probability per Batch")
    ax1.set_title("Strategy A: Data-Level Oversampling\n(Shifts Apparent Class Prior)")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.text(
        0.5, -0.20,
        "Illustrative consequence: more rare-defect exposure can shift the learned prior.\nCheck for false alarms on naturally distributed, clean validation images.",
        transform=ax1.transAxes, ha="center", fontsize=9, color="#9b2c2c", style="italic",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff5f5", edgecolor="#feb2b2")
    )

    # Loss weighting mechanics: compact wording keeps every cell legible at guide width.
    ax2.axis("off")
    table_data = [
        ["Dimension", "Image sampling", "Loss weighting"],
        ["Mechanism", "Changes which images\nare drawn", "Changes loss per\npositive pixel"],
        ["Rare signal", "More rare-image\ndraws", "Larger rare-pixel\ngradients"],
        ["Clean examples", "Fewer clean draws\nwhen oversampling", "Natural image mix\nretained"],
        ["Main risk", "Prior shift and\nrepeat exposure", "Excessive weights can\ndestabilize training"],
        ["Validation", "Natural, unweighted\ndistribution", "Natural, unweighted\ndistribution"],
    ]
    tbl = ax2.table(cellText=table_data, loc="center", cellLoc="left", colWidths=[0.23, 0.385, 0.385], bbox=[0.01, 0.08, 0.98, 0.79])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1.0, 1.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.PAD = 0.07
        if r == 0:
            cell.set_facecolor("#2b6cb0")
            cell.set_text_props(color="white", weight="bold")
        elif r % 2 == 1:
            cell.set_facecolor("#f7fafc")
        else:
            cell.set_facecolor("#edf2f7")
    ax2.set_title("Strategy B: Conceptual Trade-Off Matrix\n(Sampling vs. Loss Formulation)", y=0.95, pad=8)

    fig.suptitle("Interventions for Rare Defects: Data Sampling vs. Loss Weighting", fontsize=15, y=0.99)
    fig.savefig(OUTPUT_DIR / "fig02_sampling_vs_weighting.png", bbox_inches="tight")
    plt.close(fig)


def make_fig03_lr_schedules_comparison():
    """Fig 3: Compare Constant, Step, Cosine with Warmup, and ReduceLROnPlateau curves."""
    fig, ax = plt.subplots(figsize=(13, 6), dpi=150)
    epochs = np.linspace(0, 30, 300)
    initial_lr = 1e-3
    min_lr = 1e-5

    # 1. Constant
    lr_constant = np.full_like(epochs, initial_lr)

    # 2. Step Decay (drop at epoch 12 and 22 by 0.2)
    lr_step = np.piecewise(
        epochs,
        [epochs < 12, (epochs >= 12) & (epochs < 22), epochs >= 22],
        [initial_lr, initial_lr * 0.2, initial_lr * 0.04]
    )

    # 3. Cosine with 3-epoch linear warmup
    warmup_epochs = 3
    lr_cosine = np.zeros_like(epochs)
    for i, ep in enumerate(epochs):
        if ep < warmup_epochs:
            lr_cosine[i] = initial_lr * (ep / warmup_epochs)
        else:
            progress = (ep - warmup_epochs) / (30 - warmup_epochs)
            lr_cosine[i] = min_lr + 0.5 * (initial_lr - min_lr) * (1 + np.cos(np.pi * progress))

    # 4. Reduce on Plateau (synthetic reactive step drops)
    lr_plateau = np.piecewise(
        epochs,
        [epochs < 10, (epochs >= 10) & (epochs < 18), (epochs >= 18) & (epochs < 25), epochs >= 25],
        [initial_lr, initial_lr * 0.5, initial_lr * 0.25, initial_lr * 0.125]
    )

    ax.plot(epochs, lr_constant, label="Constant LR (No Schedule)", color="#a0aec0", linestyle="--", linewidth=2)
    ax.plot(epochs, lr_step, label="StepLR (Drops at 12 & 22 epochs)", color="#dd6b20", linewidth=2.5)
    ax.plot(epochs, lr_plateau, label="ReduceLROnPlateau (Plateau-triggered)", color="#805ad5", linewidth=2)
    ax.plot(epochs, lr_cosine, label="Cosine Annealing with Warmup (3-epoch linear warmup)", color="#006e6b", linewidth=3)

    # Annotate Warmup
    ax.axvspan(0, 3, color="#006e6b", alpha=0.1)
    ax.text(1.5, 5e-4, "Warmup Phase\n(Protects random weights)", ha="center", fontsize=9, color="#006e6b", weight="bold")

    # Annotate Cosine tail
    ax.annotate(
        "Smooth decay to min_lr_ratio (1%)\nAllows fine boundary refinement",
        xy=(26, lr_cosine[260]), xytext=(18, 7e-4),
        arrowprops=dict(facecolor="#006e6b", shrink=0.08, width=1.5, headwidth=6),
        fontsize=9, weight="bold", color="#006e6b"
    )

    ax.set_yscale("log")
    ax.set_ylim(5e-6, 2e-3)
    ax.set_xlim(0, 30)
    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("Learning Rate (Log Scale)")
    ax.set_title("Comparison of Learning-Rate Schedules Across a 30-Epoch Training Budget", fontsize=13)
    ax.grid(True, linestyle=":", alpha=0.6, which="both")
    ax.legend(loc="lower left", fontsize=10, framealpha=0.9)

    fig.savefig(OUTPUT_DIR / "fig03_lr_schedules_comparison.png", bbox_inches="tight")
    plt.close(fig)


def make_fig04_early_stopping_dynamics():
    """Fig 4: Early stopping dynamics, validation jitter, patience window, and checkpoint restoration."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), dpi=150, sharex=True)
    epochs = np.arange(1, 26)

    np.random.seed(42)
    # Synthetic Loss curves
    train_loss = 0.8 * np.exp(-epochs / 5) + 0.15 + np.random.normal(0, 0.008, len(epochs))
    val_loss = 0.82 * np.exp(-epochs / 6) + 0.22 + 0.008 * (epochs - 12)**2 * (epochs > 12) + np.random.normal(0, 0.015, len(epochs))

    # Synthetic Foreground Dice
    val_dice = 0.15 + 0.45 * (1 - np.exp(-epochs / 5.5)) - 0.003 * (epochs - 14)**2 * (epochs > 14) + np.random.normal(0, 0.012, len(epochs))
    val_dice = np.clip(val_dice, 0.1, 0.62)

    # Plot Loss
    ax1.plot(epochs, train_loss, label="Training Loss (Compound BCE + Soft Dice)", color="#3182ce", linewidth=2.5)
    ax1.plot(epochs, val_loss, label="Validation Loss", color="#e53e3e", linewidth=2.5)
    ax1.set_ylabel("Loss")
    ax1.set_title("Tracking Generalization: Loss vs. Foreground Dice Across Epochs", fontsize=13)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=9.5)

    # Mark validation loss minimum
    best_loss_ep = int(np.argmin(val_loss)) + 1
    ax1.axvline(best_loss_ep, color="#e53e3e", linestyle=":", alpha=0.7)
    ax1.text(best_loss_ep + 0.3, 0.42, f"Min Val Loss (Epoch {best_loss_ep})", color="#e53e3e", fontsize=9, weight="bold")

    # Plot Dice
    ax2.plot(epochs, val_dice, label="Validation Foreground Dice (Positive Pairs Only)", color="#006e6b", linewidth=2.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Foreground Dice")
    ax2.grid(True, linestyle=":", alpha=0.6)

    # Mark Best Dice Checkpoint
    best_dice_ep = int(np.argmax(val_dice)) + 1
    best_dice_val = val_dice[best_dice_ep - 1]
    ax2.scatter([best_dice_ep], [best_dice_val], color="#d69e2e", s=140, zorder=5, edgecolor="black")
    ax2.axvline(best_dice_ep, color="#006e6b", linestyle="--", linewidth=2)
    ax2.text(best_dice_ep + 0.3, best_dice_val - 0.03, f"BEST CHECKPOINT (Epoch {best_dice_ep})\nDice = {best_dice_val:.3f}", color="#006e6b", fontsize=9.5, weight="bold")

    # Patience window (Patience = 5)
    patience = 5
    stop_ep = best_dice_ep + patience
    ax2.axvspan(best_dice_ep, stop_ep, color="#dd6b20", alpha=0.15)
    ax2.text((best_dice_ep + stop_ep)/2, 0.25, f"Patience Window ({patience} epochs)\nNo improvement > min_delta", ha="center", fontsize=9, color="#c05621", weight="bold")

    # Early stop trigger
    ax2.axvline(stop_ep, color="#c53030", linestyle="-", linewidth=2.5)
    ax2.text(stop_ep + 0.3, 0.40, f"Early Stop Triggered (Epoch {stop_ep})\nRestore Weights from Epoch {best_dice_ep}", color="#c53030", fontsize=9.5, weight="bold")

    ax2.legend(loc="upper left", fontsize=9.5)
    fig.savefig(OUTPUT_DIR / "fig04_early_stopping_dynamics.png", bbox_inches="tight")
    plt.close(fig)


def make_fig05_amp_memory_breakdown():
    """Fig 5: GPU Memory consumption breakdown and FP32 vs AMP FP16/BF16 tensor precisions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    # Memory breakdown bar chart for a fictional 15M parameter model
    components = ["Model Weights", "Gradients", "AdamW States\n(1st + 2nd moments)", "Activations\n(Batch 8, 128x800)", "Workspace & Cache"]
    # In FP32 (Megabytes)
    # 15M * 4 bytes = 60 MB weights, 60 MB grads, 120 MB adamw, ~1200 MB activations, ~200 MB workspace
    mem_fp32 = [60, 60, 120, 1250, 250]
    # In AMP (FP16/FP32 mixed)
    # weights: 60 MB master FP32 + 30 MB FP16 = 90 MB, grads: 30 MB FP16, adamw: 120 MB FP32, activations: ~625 MB FP16, workspace: 180 MB
    mem_amp = [90, 30, 120, 625, 180]

    y = np.arange(len(components))
    h = 0.35
    ax1.barh(y - h/2, mem_fp32, h, label="Pure FP32 Precision", color="#4a5568")
    ax1.barh(y + h/2, mem_amp, h, label="Mixed Precision (PyTorch AMP)", color="#006e6b")
    ax1.set_yticks(y)
    ax1.set_yticklabels(components)
    ax1.set_xlabel("VRAM Footprint (MB)")
    ax1.set_title("VRAM Consumption: Pure FP32 vs. AMP\n(Fictional 15M-Parameter Segmentation Network)", fontsize=11)
    ax1.legend(loc="lower right", fontsize=9.5)
    ax1.grid(axis="x", linestyle="--", alpha=0.6)

    # Precision bits diagram
    ax2.axis("off")
    # Draw IEEE-754 bit layout
    ax2.text(0.05, 0.90, "Floating-Point Precision Formats", fontsize=12, weight="bold", color="#172d3a")

    formats = [
        ("FP32 (Single Precision)", "1 sign", "8 exponent bits", "23 mantissa (fraction) bits", "#2b6cb0", "Range: ~10^±38, Standard dynamic range"),
        ("FP16 (Half Precision)", "1 sign", "5 exponent bits", "10 mantissa bits", "#dd6b20", "Range: ~10^±5, Underflow risk! Requires GradScaler"),
        ("BF16 (Brain Floating)", "1 sign", "8 exponent bits", "7 mantissa bits", "#805ad5", "Range: ~10^±38, Same dynamic range as FP32, lower precision"),
    ]

    top = 0.75
    for name, s, exp, man, col, note in formats:
        ax2.text(0.05, top, name, fontsize=10, weight="bold", color=col)
        # Draw bit rectangles
        rect_s = patches.Rectangle((0.05, top - 0.08), 0.08, 0.05, facecolor="#e2e8f0", edgecolor="black")
        rect_e = patches.Rectangle((0.14, top - 0.08), 0.28, 0.05, facecolor=col, alpha=0.3, edgecolor="black")
        rect_m = patches.Rectangle((0.43, top - 0.08), 0.48, 0.05, facecolor=col, alpha=0.7, edgecolor="black")
        ax2.add_patch(rect_s)
        ax2.add_patch(rect_e)
        ax2.add_patch(rect_m)

        ax2.text(0.09, top - 0.055, s, fontsize=7.5, ha="center")
        ax2.text(0.28, top - 0.055, exp, fontsize=7.5, ha="center")
        ax2.text(0.67, top - 0.055, man, fontsize=7.5, ha="center")
        ax2.text(0.05, top - 0.12, note, fontsize=8.5, style="italic", color="#4a5568")
        top -= 0.24

    fig.suptitle("GPU Memory Distribution & Numerical Precision in Deep Segmentation", fontsize=14)
    fig.savefig(OUTPUT_DIR / "fig05_amp_memory_breakdown.png", bbox_inches="tight")
    plt.close(fig)


def make_fig06_gradient_accumulation_steps():
    """Fig 6: Microbatch vs effective batch size and gradient accumulation step sequence."""
    fig, ax = plt.subplots(figsize=(14, 5.5), dpi=150)
    ax.axis("off")

    ax.text(0.5, 0.95, "Gradient Accumulation Lifecycle: Microbatch Size = 4, Accumulation = 4 -> Effective Batch = 16",
            ha="center", fontsize=13, weight="bold", color="#172d3a")

    steps = [
        ("Microbatch 1\n(4 Strips)", "Forward pass (AMP)\nCompute loss / 4\nscaler.scale(loss).backward()", "Gradients accumulate\ninto param.grad", "#ebf8ff", "#3182ce"),
        ("Microbatch 2\n(4 Strips)", "Forward pass (AMP)\nCompute loss / 4\nscaler.scale(loss).backward()", "Gradients add to\nexisting param.grad", "#ebf8ff", "#3182ce"),
        ("Microbatch 3\n(4 Strips)", "Forward pass (AMP)\nCompute loss / 4\nscaler.scale(loss).backward()", "Gradients add to\nexisting param.grad", "#ebf8ff", "#3182ce"),
        ("Microbatch 4\n(4 Strips)", "Forward pass (AMP)\nCompute loss / 4\nscaler.scale(loss).backward()", "Final accumulation step\nReaches Effective N=16", "#fefcbf", "#d69e2e"),
    ]

    for i, (title, fwd, grad, bg, bdr) in enumerate(steps):
        x = 0.04 + i * 0.24
        box = patches.FancyBboxPatch((x, 0.38), 0.20, 0.45, boxstyle="round,pad=0.03", facecolor=bg, edgecolor=bdr, linewidth=2)
        ax.add_patch(box)
        ax.text(x + 0.10, 0.77, title, ha="center", fontsize=10, weight="bold", color="#172d3a")
        ax.text(x + 0.10, 0.62, fwd, ha="center", fontsize=8.5, color="#2d3748")
        ax.text(x + 0.10, 0.44, grad, ha="center", fontsize=8, style="italic", color="#4a5568")

        if i < 3:
            ax.annotate("", xy=(x + 0.23, 0.60), xytext=(x + 0.20, 0.60),
                        arrowprops=dict(facecolor="#718096", shrink=0.05, width=1.5, headwidth=5))

    # Optimizer Update Box at the bottom
    update_box = patches.FancyBboxPatch((0.20, 0.08), 0.60, 0.22, boxstyle="round,pad=0.03", facecolor="#f0fff4", edgecolor="#38a169", linewidth=2.5)
    ax.add_patch(update_box)
    ax.text(0.50, 0.24, "Optimizer Update Step (Every 4 Microbatches)", ha="center", fontsize=11, weight="bold", color="#22543d")
    ax.text(0.50, 0.13, "Unscale → clip (optional) → scaler.step → scaler.update\nAdvance scheduler only after a successful update; then zero gradients",
            ha="center", fontsize=9, color="#1c4532")

    # Connect Microbatch 4 to Optimizer update
    ax.annotate("", xy=(0.50, 0.31), xytext=(0.86, 0.38),
                arrowprops=dict(facecolor="#38a169", shrink=0.05, width=2, headwidth=6))

    fig.savefig(OUTPUT_DIR / "fig06_gradient_accumulation_steps.png", bbox_inches="tight")
    plt.close(fig)


def make_fig07_learning_curves_archetypes():
    """Fig 7: 4-panel diagnostic grid of learning curve archetypes."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    epochs = np.arange(1, 26)

    # 1. Healthy Generalization
    np.random.seed(1)
    ax = axes[0, 0]
    tr_loss = 0.8 * np.exp(-epochs / 6) + 0.15 + np.random.normal(0, 0.008, 25)
    val_loss = 0.85 * np.exp(-epochs / 6.5) + 0.18 + np.random.normal(0, 0.012, 25)
    ax.plot(epochs, tr_loss, label="Train Loss", color="#3182ce", linewidth=2)
    ax.plot(epochs, val_loss, label="Val Loss", color="#38a169", linewidth=2)
    ax.set_title("Archetype 1: Healthy Generalization", color="#22543d", fontsize=11)
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=8.5)
    ax.text(0.5, 0.12, "Train and Val loss descend in tandem.\nIntervention: Continue training or test scheduler decay.",
            transform=ax.transAxes, ha="center", fontsize=8.5, bbox=dict(boxstyle="round", facecolor="#f0fff4", edgecolor="#9ae6b4"))

    # 2. Classic Overfitting
    ax = axes[0, 1]
    tr_loss = 0.8 * np.exp(-epochs / 3.5) + 0.05 + np.random.normal(0, 0.005, 25)
    val_loss = 0.8 * np.exp(-epochs / 5) + 0.22 + 0.004 * (epochs - 9)**2 * (epochs > 9) + np.random.normal(0, 0.01, 25)
    ax.plot(epochs, tr_loss, label="Train Loss", color="#3182ce", linewidth=2)
    ax.plot(epochs, val_loss, label="Val Loss", color="#e53e3e", linewidth=2)
    ax.set_title("Archetype 2: Severe Overfitting", color="#9b2c2c", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=8.5)
    ax.text(0.5, 0.12, "Train loss approaches zero while Val loss surges.\nIntervention: Early stopping, dropout, weight decay, stronger augmentations.",
            transform=ax.transAxes, ha="center", fontsize=8.5, bbox=dict(boxstyle="round", facecolor="#fff5f5", edgecolor="#feb2b2"))

    # 3. Underfitting / High Bias
    ax = axes[1, 0]
    tr_loss = 0.3 * np.exp(-epochs / 2) + 0.52 + np.random.normal(0, 0.01, 25)
    val_loss = 0.3 * np.exp(-epochs / 2) + 0.55 + np.random.normal(0, 0.015, 25)
    ax.plot(epochs, tr_loss, label="Train Loss", color="#3182ce", linewidth=2)
    ax.plot(epochs, val_loss, label="Val Loss", color="#dd6b20", linewidth=2)
    ax.set_title("Archetype 3: Severe Underfitting (High Bias)", color="#9c4221", fontsize=11)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=8.5)
    ax.text(0.5, 0.12, "Both losses flatten high; model fails to fit training strips.\nIntervention: Increase capacity, lower regularization, check learning rate.",
            transform=ax.transAxes, ha="center", fontsize=8.5, bbox=dict(boxstyle="round", facecolor="#fffaf0", edgecolor="#fbd38d"))

    # 4. Metric-Loss Divergence
    ax = axes[1, 1]
    val_loss = 0.5 * np.exp(-epochs / 6) + 0.25 + 0.002 * (epochs - 12)**2 * (epochs > 12) + np.random.normal(0, 0.01, 25)
    val_fg_dice = 0.15 + 0.42 * (1 - np.exp(-epochs / 7)) + np.random.normal(0, 0.01, 25)
    ax.plot(epochs, val_loss, label="Val Loss (BCE+Dice)", color="#e53e3e", linewidth=2)
    ax.plot(epochs, val_fg_dice, label="Val Foreground Dice", color="#006e6b", linewidth=2)
    ax.set_title("Archetype 4: Metric-Loss Divergence (Probability Drift)", color="#234e52", fontsize=11)
    ax.set_xlabel("Epoch")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=8.5)
    ax.text(0.5, 0.12, "Val loss rises due to confident BCE errors, yet Dice continues rising.\nIntervention: Calibrate thresholds $\\tau$; monitor Foreground Dice for checkpointing.",
            transform=ax.transAxes, ha="center", fontsize=8.5, bbox=dict(boxstyle="round", facecolor="#e6fffa", edgecolor="#b2f5ea"))

    fig.suptitle("Diagnostic Taxonomy of Deep Segmentation Learning Curves", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig07_learning_curves_archetypes.png", bbox_inches="tight")
    plt.close(fig)


def make_fig08_metric_dice_decomposition():
    """Fig 8: 2x2 contingency grid explaining naive Dice vs foreground Dice vs empty accuracy."""
    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
    ax.axis("off")

    ax.text(0.5, 0.96, "The Evaluation Contingency Matrix: Resolving the 'Naive Dice Trap'",
            ha="center", fontsize=14, weight="bold", color="#172d3a")

    # 2x2 quadrants
    quads = [
        # (x, y, title, formula, behavior, bg, bdr)
        (0.08, 0.52, "Quadrant 1: Both Masks Nonempty\n(GT Present, Pred Present)",
         "Dice = 2|P ∩ Y| / (|P| + |Y|)\nRange: [0.0, 1.0]",
         "Measures spatial boundary alignment.\nEntering metric: Included in Foreground Dice.",
         "#f0fff4", "#38a169"),

        (0.52, 0.52, "Quadrant 2: False Negative Pair\n(GT Present, Pred Empty)",
         "Dice = 2(0) / (0 + |Y|) = 0.0\nPair Dice: ZERO",
         "Model completely missed a real defect.\nEntering metric: Included as zero in Foreground Dice.",
         "#fff5f5", "#e53e3e"),

        (0.08, 0.08, "Quadrant 3: False Positive Pair\n(GT Empty, Pred Present)",
         "Dice = 2(0) / (|P| + 0) = 0.0\nPair Dice: ZERO",
         "Model hallucinated defect on a clean strip.\nEntering metric: Penalizes Specificity & FP Score.",
         "#fffaf0", "#dd6b20"),

        (0.52, 0.08, "Quadrant 4: True Negative Pair\n(GT Empty, Pred Empty)",
         "Dice Convention: 1.0\nPair Dice: 1.0 (not the official score)",
         "Model correctly stayed silent on clean strip.\nTRAP: Dilutes Naive Mean Dice to artificially high values!",
         "#ebf8ff", "#3182ce"),
    ]

    for (x, y, title, form, desc, bg, bdr) in quads:
        box = patches.FancyBboxPatch((x, y), 0.40, 0.38, boxstyle="round,pad=0.03", facecolor=bg, edgecolor=bdr, linewidth=2.5)
        ax.add_patch(box)
        ax.text(x + 0.20, y + 0.32, title, ha="center", fontsize=11, weight="bold", color="#172d3a")
        ax.text(x + 0.20, y + 0.20, form, ha="center", fontsize=9.5, family="monospace", color="#2d3748")
        ax.text(x + 0.20, y + 0.08, desc, ha="center", fontsize=8.5, color="#4a5568")

    # Column and row headers
    ax.text(0.28, 0.91, "Model Predicted Defect (|P| > 0)", ha="center", fontsize=10, weight="bold", color="#4a5568")
    ax.text(0.72, 0.91, "Model Predicted Clean (|P| = 0)", ha="center", fontsize=10, weight="bold", color="#4a5568")
    ax.text(0.02, 0.71, "GT Positive\n(|Y| > 0)", va="center", ha="center", fontsize=10, weight="bold", color="#4a5568", rotation=90)
    ax.text(0.02, 0.27, "GT Clean\n(|Y| = 0)", va="center", ha="center", fontsize=10, weight="bold", color="#4a5568", rotation=90)

    fig.savefig(OUTPUT_DIR / "fig08_metric_dice_decomposition.png", bbox_inches="tight")
    plt.close(fig)


def make_fig09_validation_error_gallery():
    """Fig 9: Visual gallery of the six spatial error archetypes on synthetic industrial strips."""
    fig = plt.figure(figsize=(17, 6.5), dpi=180)
    grid = fig.add_gridspec(6, 5, width_ratios=[1.65, 4, 4, 4, 4], wspace=0.10, hspace=0.48)
    axes = np.empty((6, 4), dtype=object)
    label_axes = []
    for i in range(6):
        label_axes.append(fig.add_subplot(grid[i, 0]))
        label_axes[-1].set_axis_off()
        for j in range(4):
            axes[i, j] = fig.add_subplot(grid[i, j + 1])

    col_titles = ["1. Raw Strip (128×800)", "2. Ground Truth Mask", "3. Model Probability Heatmap", "4. Post-Processed Binary Mask"]
    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=10.5, weight="bold", pad=8)

    archetypes = [
        ("A:\nMissed thin defect\n(false negative)", "hairline", "miss"),
        ("B:\nFalse alarm on clean\nstrip (artifact)", "clean", "false_alarm"),
        ("C:\nClass confusion\n(A vs. C)", "scratch", "confusion"),
        ("D:\nSalt-and-pepper\nspeckle noise", "inclusion", "speckle"),
        ("E:\nBoundary dilation /\nmisplacement", "patch", "boundary_shift"),
        ("F:\nFragmented longitudinal\ncrack", "crack", "fragmented"),
    ]

    np.random.seed(42)

    for i, (name, base_type, err_type) in enumerate(archetypes):
        label_axes[i].text(
            0.98, 0.5, name, ha="right", va="center",
            fontsize=8.0, weight="bold", color="#243b53", linespacing=1.15,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#f0f4f8", edgecolor="#bcccdc"),
        )
        # Generate base synthetic strip
        strip = np.full((128, 800), 160, dtype=np.int16)
        strip += np.random.normal(0, 10, (128, 800)).astype(np.int16)

        gt = np.zeros((128, 800), dtype=np.uint8)
        prob = np.zeros((128, 800), dtype=np.float32)

        if base_type == "hairline":
            strip[60:63, 200:400] = 50
            gt[60:63, 200:400] = 1
            prob[60:63, 200:400] = 0.15 # Under confidence
        elif base_type == "clean":
            strip[40:90, 500:550] = 185 # Subtle bright roller mark
            # GT is all zeros
            prob[40:90, 500:550] = 0.78 # False alarm
        elif base_type == "scratch":
            strip[50:56, 100:300] = 40
            gt[50:56, 100:300] = 1 # GT is Defect A
            prob[50:56, 100:300] = 0.85 # But predicted on wrong channel
        elif base_type == "inclusion":
            strip[70:85, 350:390] = 45
            gt[70:85, 350:390] = 1
            prob[70:85, 350:390] = 0.90
            # Add speckle noise across strip
            speckles_y = np.random.randint(10, 118, 15)
            speckles_x = np.random.randint(50, 750, 15)
            for sy, sx in zip(speckles_y, speckles_x):
                prob[sy:sy+3, sx:sx+3] = 0.65
        elif base_type == "patch":
            strip[30:80, 450:550] = 60
            gt[30:80, 450:550] = 1
            # Bloated / dilated prediction shifted by 15 pixels
            prob[20:95, 435:570] = 0.85
        elif base_type == "crack":
            strip[64:68, 150:650] = 35
            gt[64:68, 150:650] = 1
            # Broken into 4 disjoint islands
            prob[64:68, 150:230] = 0.85
            prob[64:68, 290:380] = 0.88
            prob[64:68, 440:510] = 0.82
            prob[64:68, 580:650] = 0.90

        strip = np.clip(strip, 0, 255).astype(np.uint8)
        pred_bin = (prob >= 0.5).astype(np.uint8)

        # Plot row
        # Equal data scaling preserves the actual 800:128 strip geometry.
        axes[i, 0].imshow(strip, cmap="gray", aspect="equal", vmin=0, vmax=255)
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])

        if base_type == "scratch":
            # A class-confusion example must distinguish channel identity, even
            # though the two binary spatial masks have identical geometry.
            gt_rgb = np.zeros((*gt.shape, 3), dtype=np.float32)
            gt_rgb[gt > 0] = (0.10, 0.62, 0.88)  # defect_a: blue
            pred_rgb = np.zeros((*pred_bin.shape, 3), dtype=np.float32)
            pred_rgb[pred_bin > 0] = (0.96, 0.42, 0.12)  # defect_c: orange
            axes[i, 1].imshow(gt_rgb, aspect="equal")
            axes[i, 3].imshow(pred_rgb, aspect="equal")
            axes[i, 1].text(0.02, 0.92, "GT: defect_a", transform=axes[i, 1].transAxes,
                            color="white", fontsize=7, weight="bold", va="top",
                            bbox=dict(facecolor="#243b53", alpha=0.9, edgecolor="none", pad=2))
            axes[i, 3].text(0.02, 0.92, "Pred: defect_c", transform=axes[i, 3].transAxes,
                            color="white", fontsize=7, weight="bold", va="top",
                            bbox=dict(facecolor="#243b53", alpha=0.9, edgecolor="none", pad=2))
        else:
            axes[i, 1].imshow(gt, cmap="gray", aspect="equal", vmin=0, vmax=1)
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])

        im_p = axes[i, 2].imshow(prob, cmap="magma", aspect="equal", vmin=0, vmax=1)
        axes[i, 2].set_xticks([])
        axes[i, 2].set_yticks([])

        if base_type != "scratch":
            axes[i, 3].imshow(pred_bin, cmap="gray", aspect="equal", vmin=0, vmax=1)
        axes[i, 3].set_xticks([])
        axes[i, 3].set_yticks([])

    fig.suptitle("Visual Diagnostic Gallery: The Six Spatial Error Archetypes in Surface Segmentation", fontsize=14, y=0.99)
    fig.savefig(OUTPUT_DIR / "fig09_validation_error_gallery.png", bbox_inches="tight")
    plt.close(fig)


def make_fig10_controlled_experimentation_matrix():
    """Fig 10: Scientific iteration workflow and controlled experimentation framework."""
    fig, ax = plt.subplots(figsize=(14, 7), dpi=150)
    ax.axis("off")

    ax.text(0.5, 0.96, "The Controlled Experimentation Protocol: Scientific Attribution in Deep Learning",
            ha="center", fontsize=14, weight="bold", color="#172d3a")

    stages = [
        ("1. Invariant Baseline",
         "Hold Constant:\n• Split (Canonical Fold 0)\n• Seed (e.g., 42)\n• Metric & Threshold Grid\n• Preprocessing & Resolution",
         "#ebf8ff", "#3182ce"),

        ("2. Falsifiable Hypothesis",
         "Formulate Reason:\n'Because rare Defect B has\n0.00 recall, bounded sampling\nweights will increase positive\nexposure and raise B recall.'",
         "#fefcbf", "#d69e2e"),

        ("3. One-Factor Intervention",
         "Isolate Variable:\nChange ONLY the sampling rule.\nDO NOT change loss, LR,\nepochs, or batch size in the\nsame experiment run!",
         "#fff5f5", "#e53e3e"),

        ("4. Comprehensive Audit",
         "Evaluate Metrics:\n• Rare Class Recall (B)\n• Clean Image Specificity\n• Foreground-Pair Dice\n• Memory & Throughput",
         "#f0fff4", "#38a169"),
    ]

    for i, (title, text, bg, bdr) in enumerate(stages):
        x = 0.05 + i * 0.24
        box = patches.FancyBboxPatch((x, 0.48), 0.20, 0.38, boxstyle="round,pad=0.03", facecolor=bg, edgecolor=bdr, linewidth=2)
        ax.add_patch(box)
        ax.text(x + 0.10, 0.79, title, ha="center", fontsize=10.5, weight="bold", color="#172d3a")
        ax.text(x + 0.10, 0.62, text, ha="center", fontsize=8.5, color="#2d3748")

        if i < 3:
            ax.annotate("", xy=(x + 0.23, 0.67), xytext=(x + 0.20, 0.67),
                        arrowprops=dict(facecolor="#4a5568", shrink=0.05, width=1.5, headwidth=6))

    # Decision branches at bottom
    decisions = [
        ("Evidence Supports Hypothesis", "Metrics confirm target improvement without unmonitored collapse (e.g. clean specificity preserved). Promote to new baseline anchor.", "#f0fff4", "#276749"),
        ("Hypothesis Refuted / Inconclusive", "Target metric unchanged, or gain wiped out by clean-image false alarms. Revert change immediately; log failure mechanism in experiment journal.", "#fff5f5", "#9b2c2c"),
    ]

    for j, (d_title, d_desc, bg, bdr) in enumerate(decisions):
        x = 0.10 + j * 0.43
        box = patches.FancyBboxPatch((x, 0.12), 0.37, 0.26, boxstyle="round,pad=0.03", facecolor=bg, edgecolor=bdr, linewidth=2)
        ax.add_patch(box)
        ax.text(x + 0.185, 0.31, d_title, ha="center", fontsize=10, weight="bold", color=bdr)
        ax.text(x + 0.185, 0.20, d_desc, ha="center", fontsize=8.5, color="#2d3748")

    # Connect stage 4 to decisions
    ax.annotate("", xy=(0.28, 0.39), xytext=(0.87, 0.48),
                arrowprops=dict(facecolor="#276749", shrink=0.05, width=1.5, headwidth=5))
    ax.annotate("", xy=(0.71, 0.39), xytext=(0.87, 0.48),
                arrowprops=dict(facecolor="#9b2c2c", shrink=0.05, width=1.5, headwidth=5))

    fig.savefig(OUTPUT_DIR / "fig10_controlled_experimentation_matrix.png", bbox_inches="tight")
    plt.close(fig)


def main():
    print("Generating Milestone 4 conceptual figures...")
    make_fig01_imbalance_three_levels()
    print("  ✓ Figure 1: Three levels of class imbalance")
    make_fig02_sampling_vs_weighting()
    print("  ✓ Figure 2: Sampling vs loss weighting mechanics")
    make_fig03_lr_schedules_comparison()
    print("  ✓ Figure 3: Learning rate schedules comparison")
    make_fig04_early_stopping_dynamics()
    print("  ✓ Figure 4: Early stopping dynamics & checkpoint selection")
    make_fig05_amp_memory_breakdown()
    print("  ✓ Figure 5: GPU memory breakdown and numerical precision")
    make_fig06_gradient_accumulation_steps()
    print("  ✓ Figure 6: Gradient accumulation lifecycle")
    make_fig07_learning_curves_archetypes()
    print("  ✓ Figure 7: Learning curve archetypes")
    make_fig08_metric_dice_decomposition()
    print("  ✓ Figure 8: Evaluation contingency matrix (Naive Dice trap)")
    make_fig09_validation_error_gallery()
    print("  ✓ Figure 9: Six spatial error archetypes gallery")
    make_fig10_controlled_experimentation_matrix()
    print("  ✓ Figure 10: Controlled experimentation protocol")
    print(f"All 10 figures successfully written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
