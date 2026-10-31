# Scratch Baseline vs. Pretrained Transfer Learning
*(Milestone 3 · Question 8 Study Guide)*

---

## 1. The Core Concept: Controlled Empirical Benchmarking

In Milestone 2, you trained a basic convolutional U-Net from scratch. In Milestone 3, you introduce transfer learning by integrating an ImageNet-pretrained **ResNet-34** backbone into your segmentation model.

To establish the empirical value of transfer learning scientifically, you must conduct a **controlled comparison**: keeping all dataset splits, augmentations, optimizers, learning rates, loss functions, and batch sizes strictly identical, altering **only the weight initialization (from-scratch vs. ImageNet-pretrained)**.

```text
Fair Benchmarking Protocol:
- Architecture:          U-Net
- Backbones Compared:    From-Scratch (Random Init) vs. Pretrained ResNet-34 (ImageNet)
- Input Adaptation:      Grayscale in_channels=1
- Loss Function:         Compound Loss (0.5 * BCE + 0.5 * SoftDice, smooth=1.0)
- Optimizer:             AdamW (lr=3e-4, weight_decay=1e-2)
- Batch Size:            16
- Training Horizon:      5 Epochs
```

---

## 2. Theoretical Principles: Convergence Dynamics & Generalization

When evaluating learning curves across epochs, machine learning engineers track three primary diagnostic phenomena:

### 1. Feature Acquisition Speed (Epoch 1 Dynamics)
- **From-Scratch Model:** Starts from Gaussian or uniform random noise (e.g., Kaiming/He normal initialization). During Epoch 1, gradients must first organize basic orientation filters, edge detectors, and localized texture responses. Loss starts high, predictions are near zero or noisy, and foreground Dice on rare defects is close to 0.0.
- **Pretrained ResNet-34:** Enters training with over 21 million parameters already tuned to detect universal visual boundaries, ridges, and contrast gradients. Within the very first epoch, the decoder learns to tap into these pre-existing representations, achieving immediate foreground feature activation and substantially higher Epoch 1 Dice scores.

### 2. Generalization Gap ($|L_{\text{val}} - L_{\text{train}}|$)
The **generalization gap** measures the difference between training loss and validation loss:
$$\Delta_{\text{gen}} = |L_{\text{val}} - L_{\text{train}}|$$
- In from-scratch training on small, sparse defect datasets, models are susceptible to memorizing idiosyncratic background noise or specific training sheet artifacts, leading to divergence between training and validation trajectories (overfitting).
- Pretrained weights act as a powerful form of **implicit inductive regularization**: because the feature representations were shaped on over 1.2 million natural images, the network is constrained to robust visual primitives that generalize more reliably to unseen validation splits, yielding a **tighter generalization gap**.

### 3. Asymptotic Performance & Competition Score Gains
By Epoch 5:
- The scratch model is still struggling to optimize both low-level filters and high-level segmentation heads simultaneously.
- The pretrained model converges to a substantially lower validation loss and a markedly higher official competition score (mean Dice over classes with empty-mask evaluation). On typical industrial defect benchmarks, transfer learning delivers an immediate jump of **$+0.02\text{ to }+0.08$** in competition score over a scratch baseline.

---

## 3. Implementation Workflow in PyTorch

To train and evaluate both models consistently:

```python
import torch
import segmentation_models_pytorch as smp

# Scratch U-Net (random initialization):
model_scratch = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,       # None = Train from scratch!
    in_channels=1,
    classes=4
).to(device)

# Pretrained U-Net (ImageNet initialization):
model_pretrained = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet", # ImageNet weights!
    in_channels=1,
    classes=4
).to(device)
```

Run both models through the standard 5-epoch training loop using the official `CompoundLoss` and `AdamW` settings from [`MILESTONE_3.md`](../../MILESTONE_3.md). At the end of each epoch, record:
1. Training Loss ($L_{\text{train}}$)
2. Validation Loss ($L_{\text{val}}$)
3. Validation Foreground Dice
4. Competition Metric Score

---

## 4. Guiding Hints for Question 8

To solve Question 8 in [`MILESTONE_3.md`](../../MILESTONE_3.md):

- **Hint 1 (Identify Winning Model at Epoch 5):** Review your recorded validation curves at Epoch 5. Compare the validation loss and competition score between `model_pretrained` and `model_scratch`. Choose between `pretrained` and `scratch`.
- **Hint 2 (Epoch 1 Acquisition):** Inspect your Epoch 1 logs. Which model achieved a substantially higher foreground Dice and lower loss in the first epoch? Choose between `pretrained` and `scratch`.
- **Hint 3 (Generalization Gap):** Compute $|L_{\text{val}} - L_{\text{train}}|$ at Epoch 5 for both models. Does the pretrained model achieve a smaller difference (tighter gap) between training and validation loss compared to the scratch model? Choose `Yes` or `No`.
- **Hint 4 (Score Improvement Range):** Calculate the difference in competition score:
  $$\Delta_{\text{score}} = \text{Score}_{\text{pretrained}} - \text{Score}_{\text{scratch}}$$
  Select the matching range from the provided options: `[< +0.02, +0.02 to +0.08, > +0.10]`.

---

## 5. Your Assignment Task

1. Open [Milestone 3 Assessment: Question 8](../../MILESTONE_3.md#question-8-scratch-baseline-vs-pretrained-transfer-learning-empirical-training-convergence).
2. Train both models for 5 epochs under identical hyperparameters.
3. Compare Epoch 5 winning performance, Epoch 1 convergence speed, generalization gaps, and competition score improvements.
4. Enter your response in the specified format: `winning_model; faster_epoch1_model; tighter_gap_yes_no; score_improvement_range`.
