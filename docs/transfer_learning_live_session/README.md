# From a baseline U-Net to a pretrained encoder

**Live session · 26 October 2026 · Milestone 3: transfer learning**

You have already built a Dataset, batched samples, applied paired augmentations and trained a U-Net with BCE–Dice. Now ask a new question: **does starting from learned visual features help our segmentation model, and how can we test that fairly?**

We use [Mateusz Buda’s Brain MRI segmentation dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) as a separate teaching problem. Its input channels are MRI sequences; its target is one binary FLAIR-abnormality mask. You will reuse the same reasoning in the industrial project, where inputs and targets have different channel contracts.

## Your learning path

1. **[Understand the MRI data and patient split](dataset.md).** Pair TIFFs with masks, identify the three sequences, and keep every patient on one side of validation. Explore the patient-leakage lab.
2. **[Trace the encoder and decoder](encoder.md).** Extract multi-scale ResNet features and follow them through a U-Net-style decoder. Explore tensor sizes and concatenate skip connections.
3. **[Understand what pretraining changes](transfer.md).** Separate architecture, starting weights and training policy. Design a comparison that isolates ImageNet initialization.
4. **[Prepare MRI channels for an ImageNet encoder](inputs.md).** Contrast three MRI sequences with repeated FLAIR, inspect normalization, and understand the limits of grayscale weight adaptation.
5. **[Freeze, unfreeze and fine-tune](finetuning.md).** Follow gradients, parameter updates and BatchNorm buffers. Try deliberately incomplete freezing in the interactive lab.
6. **[Evaluate the evidence](experiments.md).** Keep the existing BCE–Dice loss, inspect thresholds and empty masks, and compare patient-level results without changing several factors at once.
7. **[Run the companion notebook](notebook.md).** Execute the same ResNet18 U-Net twice, with random and ImageNet encoder weights. Save the split, configuration, curves and prediction panels.

## What you should be able to explain afterwards

- Which weights come from ImageNet, which layers are newly initialized, and why the decoder still needs training.
- Why a tensor shaped `[3,H,W]` does not automatically mean an RGB photograph.
- Why training on one slice and validating on another slice of the same person gives an optimistic comparison.
- Why freezing gradients, selecting optimizer parameters and choosing BatchNorm mode are three separate decisions.
- What evidence supports a benefit from pretraining under your chosen budget, and what a short classroom run cannot establish.

## Learn by changing one thing

The browser contains **seven interactive labs**. Their small arrays and parameter values are explicitly illustrative; they do not train a network, make medical predictions or show measured MRI performance. A real-data contact sheet shows the actual file format. The notebook reads real TIFF files and performs real optimization.

Every lab has native keyboard-accessible controls, a reset button and a text readout. The surrounding explanation remains available without JavaScript. Predict an outcome before touching a control, then explain any surprise.

## Notebook and setup

[Download the notebook](transfer_learning.ipynb) · [Download the Python script](transfer_learning.py) · [Read setup and execution instructions](notebook.md)

Download the dataset separately or attach it to a Kaggle notebook. No dataset archive, patient metadata table or trained model is bundled with these notes. The notebook starts with a small, clearly labelled classroom subset and can be expanded after the session. Training duration depends on your hardware; completed runs are useful preparation for the live demonstration.

## How this returns to the industrial project

For this MRI lab: image `[B,3,H,W]`, target `[B,1,H,W]`, one logit channel. For industrial strips: preserve the **800 × 128** aspect ratio and use four independent defect targets. Transfer the experimental method and encoder/decoder reasoning; re-check input normalization, output channels, split groups and evaluation rules for that task.

## Dataset attribution and sources

Dataset: Mateusz Buda, *Brain MRI segmentation*, from TCIA / TCGA-LGG, distributed on Kaggle under **CC BY-NC-SA 4.0** as listed in its metadata. The target represents an annotated FLAIR abnormality; it is not a tumor-grade classifier. This is a machine-learning teaching exercise, not a diagnostic system.

The dataset accompanies Buda, Saha and Mazurowski, [*Association of genomic subtypes of lower-grade gliomas with shape features automatically extracted by a deep learning algorithm* (2019)](https://doi.org/10.1016/j.compbiomed.2019.05.002). See the [author’s implementation and dataset description](https://github.com/mateuszbuda/brain-segmentation-pytorch) and [license](https://creativecommons.org/licenses/by-nc-sa/4.0/).
