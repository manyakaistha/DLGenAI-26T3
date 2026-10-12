# Pixels, Masks & Experiments

Explore how surface images become segmentation masks, how data representation affects model design, and how to record a training experiment with Weights & Biases.

## What you will learn

### Track an experiment with W&B

Train a small model in a notebook, log training and validation metrics, inspect predictions with mask overlays and plots, and save a model as a versioned artifact.

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

### 4. Preserve geometry and small defects

Compare native strips, square resizing and aspect-preserving transforms. Investigate interpolation, aliasing and how sampling position affects thin-defect survival.

[Explore resizing and sampling](04_geometry.md)

### 5. Distinguish contrast from texture

Calculate Cohen’s d, compare inside-mask and background intensities, and discover why regions with identical histograms can have different spatial patterns. Connect these observations to convolutional receptive fields.

[Explore contrast and texture](05_texture.md)

## Learn by experimenting

Use the interactive grids and sliders to make a prediction, change one setting, and explain the result. The worked examples and annotated images are available throughout the notes for review.
