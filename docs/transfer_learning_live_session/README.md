# From a baseline U-Net to a pretrained encoder

**Live session · 26 October 2026 · Milestone 3: transfer learning**

You have already built a Dataset, batched samples, applied paired augmentations and trained a U-Net with BCE–Dice. Now ask a new question: **does starting from learned visual features help our segmentation model, and how can we test that fairly?**

These notes develop the encoder/decoder mechanics, pretrained initialization, input adaptation, freezing and fine-tuning, loss formulation and controlled evaluation in detail. Read the explanations, work through the tensor/code examples and use the activities to test your reasoning.

The [LGG MRI dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) supplies an application of these ideas. Its dataset preparation, sequence-specific preprocessing and complete training exercise are optional pages placed between the relevant concept pages. Each optional page links straight back to the next concept. You can study the full theory without setting up another project.

## Study path: concepts with optional MRI applications

### 1. Encoder and decoder mechanics

[Trace the encoder and decoder](encoder.md): retain spatial features, follow the ResNet stages, resize and concatenate skip connections, distinguish U-Net skips from residual additions, and check the output contract. The shape trace, decoder code, illustration and interactive activity are part of the main notes.

<div class="optional-route">
<strong>Optional MRI application · after the encoder page</strong>
<a href="dataset.md">MRI images, masks, TIFF pairing and patient-level splitting</a>. Inspect the real contact sheet and explore why patient leakage changes the validation question.
</div>

### 2. Pretrained weights and controlled comparisons

[Understand what pretraining changes](transfer.md): separate architecture, initialization and training policy; identify reused and newly initialized components; design matched experiment arms; and understand why a common seed alone is insufficient.

<div class="optional-route">
<strong>Optional MRI application · after pretraining</strong>
<a href="inputs.md">MRI channels, normalization and grayscale adaptation</a>. Work through sequence order, repeated FLAIR, input scaling, channel normalization and the limits of first-convolution weight equivalence.
</div>

### 3. Freezing, unfreezing and fine-tuning

[Follow parameter and buffer updates](finetuning.md): distinguish gradient flags, optimizer membership and module mode. Study the fixed-encoder implementation, BatchNorm running-statistic behavior, unfreezing policy, parameter groups and debugging checks. Use the interactive lab to separate a parameter update from a buffer update.

### 4. Losses, metrics and evidence

[Evaluate the evidence](experiments.md): retain the BCE–Dice objective, understand its reductions and empty-target behavior, separate soft training loss from thresholded evaluation, compare common validation examples, and state conclusions supported by the experiment. MRI-specific patient aggregation and figure selection are in labelled optional sections alongside the general evaluation principles.

<div class="optional-route">
<strong>Optional MRI application · after evaluation</strong>
<a href="notebook.md">Complete real-data notebook and setup guide</a>. Pair files, hold out patients, run matched models, and save curves, metrics and prediction panels. This is available when you want to apply the notes in code.
</div>

## What you should be able to explain afterwards

- Which weights come from ImageNet, which layers are newly initialized, and why the decoder still needs training.
- How feature dimensions change through the encoder and what a decoder skip concatenation does to channel counts.
- Why freezing gradients, choosing optimizer parameters and setting BatchNorm mode are separate decisions.
- Why training loss, hard Dice, empty-target errors and qualitative predictions reveal different aspects of model behavior.
- What a matched comparison can show about pretraining under a fixed budget, and what a short run cannot establish.

## Interactive activities

The seven labs cover feature shapes, checkpoint provenance, experiment design, freezing/BatchNorm, thresholds, patient splitting and input normalization. General concept labs remain in the main notes; MRI-specific labs accompany their optional application pages.

Their toy arrays and parameter values are explicitly illustrative. They do not train a browser model or invent MRI performance. Every lab has native controls, a reset action and a numeric readout; the source explanations remain readable without JavaScript.

## Two ways to follow the code

[Concept walkthrough notebook](classroom_walkthrough.ipynb) · [Concept Python script](classroom_walkthrough.py)

This small walkthrough traces real ResNet feature shapes, executes one decoder join and checks trainable parameters using a dummy input. It needs no MRI files or training run. Its default is explicitly a random-weight shape demonstration; enabling ImageNet weights requires a download or prepared cache.

[Full MRI experiment notebook](transfer_learning.ipynb) · [Full Python script](transfer_learning.py) · [Optional setup guide](notebook.md)

The complete experiment reads real TIFFs and performs real optimization. Download or attach the dataset separately. All original training code and configuration details remain available in the optional guide; no dataset archive or trained checkpoint is bundled with the website.

## Return to the industrial project

Preserve the industrial images’ **800 × 128** aspect ratio and four independent defect targets. Reuse the feature-extraction and experimental-design reasoning, while checking your own input normalization, split groups, output head and scoring rules.

## Dataset attribution and sources

MRI dataset: Mateusz Buda, *Brain MRI segmentation*, from TCIA / TCGA-LGG, listed on Kaggle under **CC BY-NC-SA 4.0**. The target is an annotated FLAIR abnormality, not a tumor-grade classifier. The [teaching figure’s provenance](figures/provenance.json) records its source and processing. This is a machine-learning teaching example, not a diagnostic system.

See the [author’s implementation](https://github.com/mateuszbuda/brain-segmentation-pytorch), [associated paper](https://doi.org/10.1016/j.compbiomed.2019.05.002) and [license](https://creativecommons.org/licenses/by-nc-sa/4.0/).
