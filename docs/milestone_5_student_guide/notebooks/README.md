# Milestone 5: Synthetic Inference & Post-Processing Walkthrough

This directory contains standalone, CPU-runnable educational walkthrough materials for **Milestone 5 (Post-Processing & Ensembling)**:

- **Jupyter Notebook**: [`synthetic_inference_pipeline.ipynb`](synthetic_inference_pipeline.ipynb)
- **Paired Python Script**: [`synthetic_inference_pipeline.py`](synthetic_inference_pipeline.py)

---

## What This Walkthrough Demonstrates

1. **Synthetic Strip Generation**: Synthesizes $128 \times 800$ line-scan test images ($6.25:1$ aspect ratio) with background texture and localized defect signatures.
2. **Multi-Model Inference Simulation**: Mocks continuous probability predictions from two distinct model backbones (e.g. U-Net and FPN).
3. **Test-Time Augmentation (TTA)**: Evaluates input flips and demonstrates the spatial probability map inversion cycle before aggregation.
4. **Ensemble Blending**: Combines predictions from multiple models via weighted soft probability averaging.
5. **Post-Processing Decision Pipeline**: Calibrates per-class decision thresholds $\boldsymbol{\tau}$ and applies 8-connected component minimum area filtering ($A_{\min}$).
6. **Memory-Bounded Streaming to RLE**: Encodes binary masks into 1-indexed column-major Fortran RLE format with single-space sentinels (`" "`) and boundary clamping at $102,400$.
7. **Automated Submission Validation**: Runs pre-flight integrity audits verifying header schemas, row counts, and RLE decodability.

---

## How to Run

### Option A: Local Execution with `uv`
Run the paired Python script directly from your terminal:
```bash
uv run python docs/milestone_5_student_guide/notebooks/synthetic_inference_pipeline.py
```
Execution takes under 2 seconds on CPU.

### Option B: Local Jupyter Lab / Notebook
Launch your local Jupyter interface:
```bash
uv run jupyter lab docs/milestone_5_student_guide/notebooks/synthetic_inference_pipeline.ipynb
```

### Option C: Google Colab
1. Upload `synthetic_inference_pipeline.ipynb` to Google Colab.
2. Select a CPU runtime (no GPU is required for this synthetic demonstration).
3. Run the notebook cells sequentially from top to bottom.
