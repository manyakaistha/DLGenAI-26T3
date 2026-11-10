"""Generate conceptual figures for Milestone 5 student guides using synthetic data."""
from __future__ import annotations

import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import numpy as np
import cv2

OUTPUT_DIR = Path("docs/milestone_5_student_guide/assets")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set clean aesthetic matching project guides
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "figure.titleweight": "bold",
})


def make_fig01_decision_pipeline_overview():
    """Fig 1: Ordered inference-to-submission stages, without implied tuning values."""
    fig, ax = plt.subplots(figsize=(15, 6), dpi=150)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6)
    ax.axis("off")
    stages = [
        ("1 · Input", ["Read sample IDs", "Native canvas: 128 × 800", "Apply model-specific", "preprocessing"], "#e9f3f1", "#006e6b"),
        ("2 · Model", ["model.eval()", "Inference mode", "Native logits", "[B, C, 128, 800]"], "#f2f6f5", "#172d3a"),
        ("3 · TTA", ["Transform each view", "Predict, then invert", "spatial flips on maps", "before aggregation"], "#e8eff8", "#255a9b"),
        ("4 · Aggregate", ["Choose logits OR", "probabilities; document", "the choice and weights", "P: [B, C, H, W]"], "#fff0cf", "#b36a18"),
        ("5 · Decide", ["Per-class rule: p > τ", "Binary masks", "4- or 8-connectivity", "discard components < N"], "#fadfe2", "#bd2348"),
        ("6 · Submit", ["Fortran order, 1-based RLE", "Empty mask: one space", "Write template row order", "12,000 rows + header"], "#e9f3f1", "#006e6b"),
    ]
    for i, (title, desc, fill, border) in enumerate(stages):
        x = 0.22 + i * 2.48
        rect = patches.FancyBboxPatch((x, 1.43), 2.15, 3.75,
            boxstyle="round,pad=0.08,rounding_size=0.12", facecolor=fill,
            edgecolor=border, linewidth=2)
        ax.add_patch(rect)
        ax.text(x + 1.075, 4.72, title, ha="center", va="center", fontsize=11,
                fontweight="bold", color="#172d3a")
        ax.text(x + 1.075, 3.17, "\n".join(desc), ha="center", va="center",
                fontsize=9.1, color="#435762", linespacing=1.55)
        if i < len(stages) - 1:
            ax.annotate("", xy=(x + 2.44, 3.28), xytext=(x + 2.18, 3.28),
                        arrowprops=dict(arrowstyle="->", lw=2.2, color="#087e83"))
    ax.text(7.5, 0.73,
            "Per image × class: restore native coordinates → decide → encode → write the matching template row",
            ha="center", va="center", fontsize=10.5, color="#087e83", fontweight="bold")
    ax.set_title("Inference to submission: keep each operation in order", pad=18,
                 fontsize=15, color="#172d3a")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig01_decision_pipeline_overview.png", dpi=150)
    plt.close(fig)
    print("Created fig01_decision_pipeline_overview.png")


def make_fig02_threshold_precision_recall_tradeoff():
    """Fig 2: Synthetic holdout using strict p>tau and the stated score components."""
    rng = np.random.default_rng(42)
    h, w, n_pairs = 128, 800, 40
    n_positive = n_pairs // 2
    gt = np.zeros((n_pairs, h, w), dtype=np.uint8)
    probs = rng.beta(0.5, 10.0, size=(n_pairs, h, w)).astype(np.float32)
    for i in range(n_positive):
        y0, x0 = 18 + (i * 17) % 82, 30 + (i * 137) % 590
        hh, ww = 4 + i % 7, 35 + (i * 11) % 95
        gt[i, y0:y0 + hh, x0:x0 + ww] = 1
        core = gt[i].astype(bool)
        probs[i, core] = rng.beta(5.0, 1.7, size=int(core.sum()))
        probs[i] = np.maximum(probs[i], cv2.dilate(gt[i], np.ones((3, 3), np.uint8)) * 0.25)
    # Empty pairs with small false alarms separate pixel FP score from empty accuracy.
    for i in range(n_positive, n_pairs):
        if i % 2 == 0:
            y0, x0 = 52 + i % 13, 170 + i * 19 % 570
            probs[i, y0:y0 + 2, x0:x0 + 4] = 0.72

    taus = np.linspace(0.05, 0.90, 35)
    fg_dice, fp_score, empty_acc, total_score = [], [], [], []
    for tau in taus:
        pred = probs > tau  # Strict: probability equal to tau is negative.
        pair_dice = []
        for i in range(n_positive):
            p_count = int(pred[i].sum())
            y_count = int(gt[i].sum())
            overlap = int(np.logical_and(pred[i], gt[i]).sum())
            pair_dice.append(2 * overlap / (p_count + y_count) if p_count + y_count else 1.0)
        empty_pred = pred[n_positive:].reshape(n_pairs - n_positive, -1)
        fp_fraction = empty_pred.sum(axis=1) / (h * w)
        fg = float(np.mean(pair_dice))
        fp = float(1.0 - np.mean(fp_fraction))
        ea = float(np.mean(empty_pred.sum(axis=1) == 0))
        fg_dice.append(fg)
        fp_score.append(fp)
        empty_acc.append(ea)
        total_score.append(0.70 * fg + 0.20 * fp + 0.10 * ea)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.7), dpi=150)
    ax1.hist(probs[~gt.astype(bool)], bins=45, density=True, alpha=0.65,
             color="#536773", label="Synthetic background pixels")
    ax1.hist(probs[gt.astype(bool)], bins=45, density=True, alpha=0.70,
             color="#087e83", label="Synthetic GT defect pixels")
    ax1.axvline(0.50, color="#bd2348", linestyle="--", linewidth=2,
                label="Example τ = 0.50; positive iff p > τ")
    ax1.set_title("Synthetic pixel scores (illustration)")
    ax1.set_xlabel("Predicted probability p")
    ax1.set_ylabel("Density")
    ax1.legend(loc="upper center", fontsize=8.4)
    ax1.grid(True, alpha=0.22)

    ax2.plot(taus, fg_dice, label="Foreground Dice · 70%", color="#087e83", linewidth=2.2)
    ax2.plot(taus, fp_score, label="FP score · 20% (1 − mean FP pixel fraction)", color="#d88836", linewidth=2.0)
    ax2.plot(taus, empty_acc, label="Empty accuracy · 10%", color="#255a9b", linewidth=2.0)
    ax2.plot(taus, total_score, label="Weighted score on synthetic set", color="#bd2348", linewidth=2.5)
    ax2.set_title("Threshold sweep: 20 positive + 20 empty pairs")
    ax2.set_xlabel("Threshold τ (mask = p > τ)")
    ax2.set_ylabel("Score component")
    ax2.set_ylim(-0.02, 1.02)
    ax2.legend(loc="lower left", fontsize=8.2)
    ax2.grid(True, alpha=0.22)
    fig.suptitle("Illustrative synthetic example — not dataset or leaderboard results", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUTPUT_DIR / "fig02_threshold_precision_recall_tradeoff.png", dpi=150)
    plt.close(fig)
    print("Created fig02_threshold_precision_recall_tradeoff.png")


def make_fig03_connected_components_connectivity():
    """Fig 3: 4-connectivity vs 8-connectivity breakdown on diagonal defect."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), dpi=150)

    # 8x8 toy grid with a diagonal defect scratch
    grid = np.zeros((8, 8), dtype=np.uint8)
    for i in range(1, 6):
        grid[i, i] = 1
    # Add a thickness pixel
    grid[3, 4] = 1

    # 1. Binary Mask
    ax1.imshow(grid, cmap="Blues", origin="upper")
    ax1.set_title("1. Thresholded Binary Mask\n(Diagonal Scratch, 6 px)")
    for y in range(8):
        for x in range(8):
            val = grid[y, x]
            ax1.text(x, y, str(val), ha="center", va="center",
                     color="white" if val == 1 else "#536773", fontweight="bold" if val == 1 else "normal")
    ax1.set_xticks(range(8))
    ax1.set_yticks(range(8))
    ax1.grid(color="gray", linestyle="-", linewidth=0.5, alpha=0.5)

    # 2. 4-Connectivity CC
    num4, labels4, stats4, _ = cv2.connectedComponentsWithStats(grid, connectivity=4)
    cmap4 = plt.colormaps["tab10"].resampled(max(num4, 1))
    ax2.imshow(labels4, cmap=cmap4, origin="upper")
    ax2.set_title(f"2. 4-Connectivity: {num4-1} Fragmented Blobs\n(Diagonal neighbors NOT connected!)")
    for y in range(8):
        for x in range(8):
            val = labels4[y, x]
            if val > 0:
                ax2.text(x, y, f"C{val}", ha="center", va="center", color="black", fontweight="bold", fontsize=9)
    ax2.set_xticks(range(8))
    ax2.set_yticks(range(8))
    ax2.grid(color="gray", linestyle="-", linewidth=0.5, alpha=0.5)

    # 3. 8-Connectivity CC
    num8, labels8, stats8, _ = cv2.connectedComponentsWithStats(grid, connectivity=8)
    cmap8 = plt.colormaps["tab10"].resampled(max(num8, 2))
    ax3.imshow(labels8, cmap=cmap8, origin="upper")
    ax3.set_title(f"3. 8-Connectivity: {num8-1} Unified Blob\n(Preserves diagonal continuity!)")
    for y in range(8):
        for x in range(8):
            val = labels8[y, x]
            if val > 0:
                ax3.text(x, y, f"C{val}\n({stats8[val, cv2.CC_STAT_AREA]}px)", ha="center", va="center", color="black", fontweight="bold", fontsize=9)
    ax3.set_xticks(range(8))
    ax3.set_yticks(range(8))
    ax3.grid(color="gray", linestyle="-", linewidth=0.5, alpha=0.5)

    fig.suptitle("Neighborhood Topology: 4-Connected vs. 8-Connected Components", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig03_connected_components_connectivity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Created fig03_connected_components_connectivity.png")


def make_fig04_min_area_filtering_tradeoff():
    """Fig 4: Invented component areas show what a minimum-area rule removes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)
    rng = np.random.default_rng(42)
    noise_areas = rng.exponential(scale=6, size=1500).astype(int) + 1
    gt_areas = rng.lognormal(mean=4.8, sigma=0.8, size=600).astype(int) + 1
    bins = np.logspace(0, 3.5, 40)
    ax1.hist(noise_areas, bins=bins, color="#d95a40", alpha=0.62,
             label="Synthetic false-positive components")
    ax1.hist(gt_areas, bins=bins, color="#087e83", alpha=0.68,
             label="Synthetic GT components")
    example_n = 40
    ax1.set_xscale("log")
    ax1.axvline(example_n, color="#172d3a", linestyle="--", linewidth=2,
                 label="Example N = 40: discard area < N")
    ax1.set_title("Invented component areas (log scale)")
    ax1.set_xlabel("Component area after thresholding (pixels)")
    ax1.set_ylabel("Component count")
    ax1.legend(loc="upper right", fontsize=8.8)
    ax1.grid(True, alpha=0.22)

    cutoffs = np.arange(0, 151)
    noise_removed = np.array([(noise_areas < n).mean() for n in cutoffs]) * 100
    gt_retained = np.array([
        gt_areas[gt_areas >= n].sum() / gt_areas.sum() if n > 0 else 1.0
        for n in cutoffs
    ]) * 100
    ax2.plot(cutoffs, noise_removed, color="#d95a40", linewidth=2.2,
             label="Synthetic FP components removed")
    ax2.plot(cutoffs, gt_retained, color="#087e83", linewidth=2.2,
             label="Synthetic GT component pixels retained")
    ax2.axvline(example_n, color="#172d3a", linestyle="--", linewidth=1.7,
                 label="Example N = 40 (not a recommendation)")
    ax2.set_title("Toy trade-off computed from left-panel samples")
    ax2.set_xlabel("Minimum component area N (pixels)")
    ax2.set_ylabel("Share of toy components / GT pixels (%)")
    ax2.set_ylim(-2, 102)
    ax2.legend(loc="center right", fontsize=8.5)
    ax2.grid(True, alpha=0.22)
    fig.suptitle("Illustrative only: invented areas, not measured dataset statistics", fontsize=13)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUTPUT_DIR / "fig04_min_area_filtering_tradeoff.png", dpi=150)
    plt.close(fig)
    print("Created fig04_min_area_filtering_tradeoff.png")


def make_fig05_tta_flip_inversion_cycle():
    """Fig 5: The Test-Time Augmentation (TTA) forward-predict-invert-average cycle."""
    fig, axs = plt.subplots(2, 3, figsize=(15, 7.5), dpi=150)

    # Row 1: The correct TTA pipeline
    # 1. Original image with asymmetric defect on left
    img = np.zeros((6, 12), dtype=np.float32)
    img[2:4, 2:4] = 0.95  # Defect on left

    axs[0, 0].imshow(img, cmap="magma", vmin=0, vmax=1)
    axs[0, 0].set_title("1. Original Image x\n(Defect at x=[2..3])")
    axs[0, 0].text(2.5, 3, "★ Defect", color="white", fontweight="bold", ha="center")

    # 2. H-Flipped image
    img_hflip = np.fliplr(img)
    axs[0, 1].imshow(img_hflip, cmap="magma", vmin=0, vmax=1)
    axs[0, 1].set_title("2. Transformed: H-Flip T(x)\n(Defect moves to right x=[8..9])")
    axs[0, 1].text(8.5, 3, "★ Defect", color="white", fontweight="bold", ha="center")

    # 3. Inverted prediction
    pred_flipped = img_hflip  # ideal model prediction
    pred_inv = np.fliplr(pred_flipped)
    axs[0, 2].imshow(pred_inv, cmap="magma", vmin=0, vmax=1)
    axs[0, 2].set_title("3. Inverted: T^-1(Pred)\n(Restored to original x=[2..3]!)")
    axs[0, 2].text(2.5, 3, "★ Restored", color="white", fontweight="bold", ha="center")

    # Row 2: Inversion bug comparison vs Final Blend
    # 4. Inversion OMITTED (BUG)
    axs[1, 0].imshow(pred_flipped, cmap="Reds", vmin=0, vmax=1)
    axs[1, 0].set_title("INVERSION BUG: Raw TTA Output\n(Un-inverted: Defect at WRONG side!)")
    axs[1, 0].text(8.5, 3, "[X] Misaligned", color="black", fontweight="bold", ha="center")

    # 5. Averaged with Bug (Catastrophic ghosting!)
    bad_blend = 0.5 * img + 0.5 * pred_flipped
    axs[1, 1].imshow(bad_blend, cmap="magma", vmin=0, vmax=1)
    axs[1, 1].set_title("Average WITH Bug (0.5 Orig + 0.5 Bad)\n(Double ghost defects, 0.5 prob!)")
    axs[1, 1].text(2.5, 3, "0.48", color="white", ha="center")
    axs[1, 1].text(8.5, 3, "0.48", color="white", ha="center")

    # 6. Correct Averaged Blend
    good_blend = 0.5 * img + 0.5 * pred_inv
    axs[1, 2].imshow(good_blend, cmap="magma", vmin=0, vmax=1)
    axs[1, 2].set_title("CORRECT Blend (0.5 Orig + 0.5 Inverted)\n(Reinforced prediction, IoU = 1.0!)")
    axs[1, 2].text(2.5, 3, "0.95 ★", color="white", fontweight="bold", ha="center")

    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Test-Time Augmentation (TTA) Spatial Inversion & Failure Mode", fontsize=14, y=0.98)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig05_tta_flip_inversion_cycle.png", dpi=150)
    plt.close(fig)
    print("Created fig05_tta_flip_inversion_cycle.png")


def make_fig06_ensemble_blending_vs_voting():
    """Fig 6: Soft probability averaging vs hard binary mask voting near decision boundary."""
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 4.5), dpi=150)

    # 1D slice across a defect boundary
    x = np.linspace(0, 10, 50)
    # Model A: slightly higher confidence, shifted left
    p_a = 1.0 / (1.0 + np.exp(-1.5 * (x - 4.5)))
    # Model B: slightly lower confidence, shifted right
    p_b = 1.0 / (1.0 + np.exp(-1.2 * (x - 5.5)))

    # 1. Individual models
    ax1.plot(x, p_a, label="Model A (U-Net)", color="#006e6b", linewidth=2.0)
    ax1.plot(x, p_b, label="Model B (FPN)", color="#c57924", linewidth=2.0)
    ax1.axhline(0.5, color="gray", linestyle="--", label="Threshold τ = 0.5")
    ax1.set_title("1. Individual Probabilities")
    ax1.set_xlabel("Pixel Coordinate x")
    ax1.set_ylabel("Predicted Probability")
    ax1.legend(loc="upper left", fontsize=8.5)
    ax1.grid(True, alpha=0.25)

    # 2. Soft Probability Blend
    p_blend = 0.5 * p_a + 0.5 * p_b
    ax2.plot(x, p_blend, label="Soft Average (0.5 A + 0.5 B)", color="#172d3a", linewidth=2.5)
    ax2.plot(x, p_a, color="#006e6b", alpha=0.3, linestyle=":")
    ax2.plot(x, p_b, color="#c57924", alpha=0.3, linestyle=":")
    ax2.axhline(0.5, color="gray", linestyle="--")
    ax2.set_title("2. Soft Probability Blending\n(Continuous Uncertainty)")
    ax2.set_xlabel("Pixel Coordinate x")
    ax2.legend(loc="upper left", fontsize=8.5)
    ax2.grid(True, alpha=0.25)

    # 3. Hard Voting (Majority)
    mask_a = (p_a >= 0.5).astype(int)
    mask_b = (p_b >= 0.5).astype(int)
    hard_vote_any = ((mask_a + mask_b) >= 1).astype(int)
    hard_vote_all = ((mask_a + mask_b) >= 2).astype(int)

    ax3.step(x, mask_a, label="Mask A (τ=0.5)", color="#006e6b", alpha=0.6, where="mid")
    ax3.step(x, mask_b, label="Mask B (τ=0.5)", color="#c57924", alpha=0.6, where="mid")
    ax3.step(x, hard_vote_all, label="Unanimous Vote (AND)", color="#bd2348", linewidth=2.0, where="mid")
    ax3.set_title("3. Hard Binary Voting\n(Information Loss at τ)")
    ax3.set_xlabel("Pixel Coordinate x")
    ax3.legend(loc="upper left", fontsize=8.5)
    ax3.grid(True, alpha=0.25)

    # 4. Final Binary Masks Comparison
    mask_soft = (p_blend >= 0.5).astype(int)
    ax4.step(x, mask_soft, label="Thresholded Soft Blend", color="#172d3a", linewidth=2.2, where="mid")
    ax4.step(x, hard_vote_all, label="Hard Unanimous (AND)", color="#bd2348", linestyle="--", where="mid")
    ax4.step(x, hard_vote_any, label="Hard Any (OR)", color="#006e6b", linestyle=":", where="mid")
    ax4.set_title("4. Resulting Decision Boundary")
    ax4.set_xlabel("Pixel Coordinate x")
    ax4.legend(loc="upper left", fontsize=8.5)
    ax4.grid(True, alpha=0.25)

    fig.suptitle("Ensemble Aggregation: Soft Blending vs. Hard Voting", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig06_ensemble_blending_vs_voting.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Created fig06_ensemble_blending_vs_voting.png")


def make_fig07_streaming_inference_memory_profile():
    """Fig 7: Decimal cache arithmetic; the streaming number is one buffer only."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.7), dpi=150)
    images = np.arange(0, 3001, 100)
    # Decimal bytes: N images × 4 channels × 128 × 800 × 4-byte float32.
    one_view_gb = images * 4 * 128 * 800 * 4 / 1e9
    four_views_gb = one_view_gb * 4
    ax1.plot(images, one_view_gb, color="#d88836", linewidth=2.2,
             label="One cached probability map / image")
    ax1.plot(images, four_views_gb, color="#d95a40", linewidth=2.5,
             label="Four cached TTA views / image")
    ax1.axhline(20, color="#255a9b", linestyle="--", linewidth=1.8,
                label="20 GB working-disk quota")
    ax1.axhline(30, color="#172d3a", linestyle=":", linewidth=1.8,
                label="30 GB host-RAM limit (separate resource)")
    ax1.scatter([3000, 3000], [one_view_gb[-1], four_views_gb[-1]],
                color=["#d88836", "#d95a40"], zorder=4)
    ax1.annotate("4.915 GB", (3000, one_view_gb[-1]), xytext=(-64, 8),
                 textcoords="offset points", fontsize=9, color="#8b571c")
    ax1.annotate("19.661 GB", (3000, four_views_gb[-1]), xytext=(-76, 8),
                 textcoords="offset points", fontsize=9, color="#a63c2d")
    ax1.set_title("Full-test float32 cache grows with image count")
    ax1.set_xlabel("Processed test images")
    ax1.set_ylabel("Cache size (decimal GB)")
    ax1.set_ylim(0, 33)
    ax1.legend(loc="upper left", fontsize=8.2)
    ax1.grid(True, alpha=0.22)

    batch_bytes = 16 * 4 * 128 * 800 * 4
    cache_4view_bytes = 3000 * 4 * 128 * 800 * 4 * 4
    values_mb = [batch_bytes / 1e6, cache_4view_bytes / 1e6]
    bars = ax2.barh([1, 0], values_mb, color=["#087e83", "#d95a40"], height=0.56)
    ax2.set_yticks([1, 0], ["One B=16 probability batch", "Four views × all 3,000 images"])
    ax2.set_xscale("log")
    ax2.set_xlim(10, 35000)
    ax2.set_xlabel("Float32 probability storage (decimal MB)")
    ax2.set_title("Streaming bounds this probability buffer only")
    ax2.grid(axis="x", alpha=0.22)
    for bar, value in zip(bars, values_mb):
        label = f"{value:,.1f} MB" if value < 1000 else f"{value / 1000:.3f} GB"
        ax2.text(value + 250, bar.get_y() + bar.get_height() / 2, label,
                 va="center", fontsize=9, color="#172d3a")
    ax2.text(0.0, -0.23,
             "Excludes model weights, images, logits and temporary buffers; measure total peak RAM/VRAM.",
             transform=ax2.transAxes, fontsize=8.7, color="#536773")

    fig.suptitle("Streaming keeps prediction buffers bounded; full caches scale with the test set", fontsize=13)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(OUTPUT_DIR / "fig07_streaming_inference_memory_profile.png", dpi=150)
    plt.close(fig)
    print("Created fig07_streaming_inference_memory_profile.png")


def make_fig08_fortran_rle_traversal():
    """Fig 8: Column-major Fortran RLE traversal, 1-indexing, and boundary clamping."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)

    # Left: 4x5 toy grid showing 1-indexed Fortran traversal order
    H, W = 4, 5
    canvas = np.zeros((H, W), dtype=int)
    idx = 1
    for col in range(W):
        for row in range(H):
            canvas[row, col] = idx
            idx += 1

    ax1.imshow(canvas, cmap="Blues", alpha=0.3)
    for row in range(H):
        for col in range(W):
            ax1.text(col, row, f"Pixel #{canvas[row, col]}\n(r={row}, c={col})",
                     ha="center", va="center", fontsize=9, color="#172d3a", fontweight="bold")
            # Draw column traversal arrows
            if row < H - 1:
                ax1.annotate("", xy=(col, row + 0.35), xytext=(col, row + 0.15),
                             arrowprops=dict(arrowstyle="->", lw=1.2, color="#006e6b"))

    for col in range(W - 1):
        ax1.annotate("", xy=(col + 1, 0), xytext=(col, H - 1),
                     arrowprops=dict(arrowstyle="->", lw=1.0, color="#e7462b", linestyle=":"))

    ax1.set_title("Fortran Traversal (Top-to-Bottom, Left-to-Right)")
    ax1.set_xticks(range(W))
    ax1.set_xticklabels([f"Col {c}" for c in range(W)])
    ax1.set_yticks(range(H))
    ax1.set_yticklabels([f"Row {r}" for r in range(H)])
    ax1.grid(color="gray", linestyle="-", linewidth=0.5, alpha=0.4)

    # Right: Example defect runs and RLE string
    mask = np.zeros((H, W), dtype=int)
    # Run 1: Pixels 2, 3 (col 0, row 1, 2) -> start=2, length=2
    mask[1, 0] = 1; mask[2, 0] = 1
    # Run 2: Pixels 10, 11, 12 (col 2, row 1, 2, 3) -> start=10, length=3
    mask[1, 2] = 1; mask[2, 2] = 1; mask[3, 2] = 1

    ax2.imshow(mask, cmap="Reds", alpha=0.4)
    for row in range(H):
        for col in range(W):
            val = mask[row, col]
            p_idx = canvas[row, col]
            ax2.text(col, row, f"#{p_idx}\n{'DEFECT' if val else '0'}",
                     ha="center", va="center", fontsize=9,
                     color="#bd2348" if val else "#536773",
                     fontweight="bold" if val else "normal")

    ax2.set_title("Encoded RLE Runs: '2 2 10 3'")
    ax2.set_xticks(range(W))
    ax2.set_xticklabels([f"Col {c}" for c in range(W)])
    ax2.set_yticks(range(H))
    ax2.set_yticklabels([f"Row {r}" for r in range(H)])
    ax2.grid(color="gray", linestyle="-", linewidth=0.5, alpha=0.4)

    fig.suptitle("Run-Length Encoding (RLE): 1-Indexed Column-Major Contract", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig08_fortran_rle_traversal.png", dpi=150)
    plt.close(fig)
    print("Created fig08_fortran_rle_traversal.png")


def make_fig09_inference_ablation_ladder():
    """Fig 9: Multi-metric ablation ladder radar/bar chart demonstrating causal attribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)

    stages = [
        "1. Raw Baseline (τ=0.5)",
        "2. + Threshold Search",
        "3. + Min-Area Filter",
        "4. + Test-Time Aug (H-Flip)",
        "5. + 2-Model Ensemble"
    ]

    # Fictional validation results illustrating the progressive hypothesis testing
    fg_dice = [0.510, 0.545, 0.548, 0.562, 0.575]
    fp_score = [0.720, 0.740, 0.810, 0.815, 0.830]
    empty_acc = [0.850, 0.860, 0.890, 0.892, 0.898]
    # Weighted composite score: 0.7*fg + 0.2*fp + 0.1*empty
    comp_score = [0.70*f + 0.20*p + 0.10*e for f, p, e in zip(fg_dice, fp_score, empty_acc)]
    # Inference time per image (ms)
    latency_ms = [4.2, 4.3, 5.1, 8.9, 17.5]

    x = np.arange(len(stages))
    width = 0.22

    ax1.bar(x - width, fg_dice, width, label="Foreground Dice (70%)", color="#006e6b")
    ax1.bar(x, fp_score, width, label="False Positive Score (20%)", color="#c57924")
    ax1.bar(x + width, comp_score, width, label="Weighted Competition Score", color="#bd2348")

    ax1.set_title("Metric Progression Across the Ablation Ladder")
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages, rotation=25, ha="right", fontsize=9)
    ax1.set_ylabel("Validation Score (0 to 1)")
    ax1.set_ylim(0.4, 0.9)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.25, axis="y")

    # Right: The Cost Axis: Latency vs Performance Gain
    ax2.plot(stages, comp_score, color="#bd2348", marker="o", linewidth=2.2, label="Competition Score")
    ax2.set_ylabel("Competition Score", color="#bd2348")
    ax2.tick_params(axis="y", labelcolor="#bd2348")
    ax2.set_ylim(0.55, 0.70)
    ax2.grid(True, alpha=0.25)

    ax2_twin = ax2.twinx()
    ax2_twin.plot(stages, latency_ms, color="#172d3a", marker="s", linestyle="--", linewidth=2.0, label="Inference Latency (ms/img)")
    ax2_twin.set_ylabel("Inference Latency (ms per image)", color="#172d3a")
    ax2_twin.tick_params(axis="y", labelcolor="#172d3a")
    ax2_twin.set_ylim(0, 25)

    ax2.set_title("Performance Gain vs. Computational Cost")
    ax2.set_xticks(x)
    ax2.set_xticklabels(stages, rotation=25, ha="right", fontsize=9)

    fig.suptitle("The Ablation Ladder: Isolating Gains & Computational Trade-Offs", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig09_inference_ablation_ladder.png", dpi=150)
    plt.close(fig)
    print("Created fig09_inference_ablation_ladder.png")


def main():
    print(f"Generating Milestone 5 figures in {OUTPUT_DIR}...")
    make_fig01_decision_pipeline_overview()
    make_fig02_threshold_precision_recall_tradeoff()
    make_fig03_connected_components_connectivity()
    make_fig04_min_area_filtering_tradeoff()
    make_fig05_tta_flip_inversion_cycle()
    make_fig06_ensemble_blending_vs_voting()
    make_fig07_streaming_inference_memory_profile()
    make_fig08_fortran_rle_traversal()
    make_fig09_inference_ablation_ladder()
    print("All 9 Milestone 5 conceptual figures generated successfully!")


if __name__ == "__main__":
    main()
