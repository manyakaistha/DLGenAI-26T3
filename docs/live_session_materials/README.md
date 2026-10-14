# Pixels, Masks & Experiments

Explore how surface images become segmentation masks, how data representation affects model design, and how to record a training experiment with Weights & Biases.

[Watch live session recording (12 October 2026)](https://drive.google.com/file/d/1Cx4M0aVOmpj85j4lGFOKg-iId2v5um2X/view?usp=sharing) · [Open the W&B lab](wandb.md) · [Download the notebook](wandb_tutorial.ipynb)

## Live session recording

The live session for DLGenAI 26T3 was conducted on 12 October 2026 covering data representation, output heads, run-length encoding, and experiment tracking with Weights & Biases.

- **Live recording:** [Watch the 12 October 2026 session recording on Google Drive](https://drive.google.com/file/d/1Cx4M0aVOmpj85j4lGFOKg-iId2v5um2X/view?usp=sharing)

## What you will learn

### Track an experiment with W&B

Use the notebook example to start a run, log metrics, tables and plots, and save a versioned model artifact. A tiny built-in dataset supplies values to track.

[Open the W&B lab](wandb.md) · [Download the notebook](wandb_tutorial.ipynb)

### 1. Understand the labels

Read the annotation table, distinguish images from image–class pairs and pixels, identify clean images, and measure class imbalance and label cardinality.

[Explore data sparsity and cardinality](01_data.md)

### 2. Choose what a pixel can mean

Distinguish multiple defects in an image from overlapping labels at a pixel. Compare sigmoid and softmax outputs, background labels, target formats and loss functions.

[Explore masks and output heads](02_heads.md)

### 3. Encode and decode segmentation masks

Follow numbered pixels through column-major traversal, build start–length RLE pairs, reconstruct a mask, and check indexing, empty masks and canvas boundaries.

[Explore RLE with an editable pixel grid](03_rle.md)

## Learn by experimenting

Use the interactive grids and sliders to make a prediction, change one setting, and explain the result. The worked examples and annotated images are available throughout the notes for review.
