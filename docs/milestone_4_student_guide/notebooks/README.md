# Milestone 4: Synthetic Optimization Walkthrough Notebook

This folder contains a standalone, CPU-runnable pedagogical walkthrough of training improvements and model optimization techniques:

- **Jupyter Notebook**: [`synthetic_optimization_walkthrough.ipynb`](synthetic_optimization_walkthrough.ipynb)
- **Paired Python Script**: [`synthetic_optimization_walkthrough.py`](synthetic_optimization_walkthrough.py)

---

## What This Walkthrough Demonstrates

1. **Combination-Group Sampling**:
   - Uses a small fictional group table unrelated to competition counts.
   - Assigns each image its exact group's capped square-root weight and compares expected group mass with a seeded replacement draw.
2. **Learning Rate Schedulers (Linear Warmup + Cosine Decay)**:
   - Configures `torch.optim.lr_scheduler.LambdaLR` to execute a 3-epoch linear warmup followed by smooth cosine decay down to 1% of initial learning rate (`min_lr_ratio = 0.01`).
3. **Early Stopping & Checkpoint Rollback**:
   - Implements an `EarlyStoppingTracker` with `mode='max'`, `min_delta`, and `patience`.
   - Simulates validation metric jitter, triggers early stopping, and restores best checkpoint weights.
4. **PyTorch 2.x Automatic Mixed Precision (AMP) & Gradient Accumulation**:
   - Simulates `torch.amp.autocast` and `torch.amp.GradScaler` on toy tensors.
   - Demonstrates microbatch accumulation, unscaling, gradient norm clipping (`clip_grad_norm_`), and optimizer stepping.
5. **Metric Governance & The "Naive Dice Trap"**:
   - Decomposes segmentation metrics into naive mean Dice, foreground-pair Dice, empty-pair accuracy, and false-positive pixel score.
   - Shows how empty-empty pairs can inflate a mean without supplying evidence of foreground detection.
6. **Spatial Error Auditing**:
   - Stratifies defect recall across area bins (small 1..100 px, medium 101..500 px, large >500 px).
   - Separates presence recall from spatial overlap hit.

---

## How to Run

### Option A: Local Terminal Execution with `uv`
Run the paired Python script directly from your terminal:
```bash
uv run python docs/milestone_4_student_guide/notebooks/synthetic_optimization_walkthrough.py
```
Execution finishes in under 3 seconds on CPU.

### Option B: Jupyter Lab / Notebook
Launch your local Jupyter server:
```bash
uv run jupyter lab docs/milestone_4_student_guide/notebooks/synthetic_optimization_walkthrough.ipynb
```

### Option C: Google Colab
1. Upload `synthetic_optimization_walkthrough.ipynb` to Google Colab.
2. Select a CPU runtime (no GPU is required for this synthetic demonstration).
3. Run the cells in sequential order.

---

## Pedagogical Boundaries & Guarantees

- **Synthetic examples**: All labels, image counts, masks, and scores are invented for practice.
- **Zero Competition Leaks**: No raw images, real test IDs, private EDA numbers, or competition results are included.
- **Hardware Agnostic**: Fully tested on standard CPU environments using PyTorch 2.8.0.
