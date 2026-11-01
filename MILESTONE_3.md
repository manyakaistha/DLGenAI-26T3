# Milestone 3: Transfer Learning & Stronger Segmentation Models

Industrial Surface Anomaly Segmentation · Milestone 3 Assessment

Read the [student study guides](docs/milestone_3_student_guide/README.md) before attempting the questions. Use the competition data in `data/public/` and compute your own results. This document contains the assessment, model/loss reference definitions and a common experiment protocol. Practice examples live in the study guides; compute assessment results yourself.

For Questions 8 and 9, train your models using the official starter code and standard training protocol provided below.

---

## Overview
In Milestone 2, you built the data foundation: a custom PyTorch `Dataset` and `DataLoader` pipeline and trained a basic from-scratch U-Net baseline. This milestone tests whether pretrained features and alternative loss formulations help under a controlled protocol. Measure imbalance and model outcomes yourself; no fixed improvement or winning architecture is assumed.

**Milestone 3 studies the following modeling techniques:**
1. **Encoder–Decoder Spatial Mechanics:** Feature map downsampling, bottleneck compression, skip-connection concatenation, and receptive field expansion on anisotropic ($128 \times 800$) sensor ribbons.
2. **Transfer Learning with Pretrained Backbones:** Leveraging ImageNet feature hierarchies (ResNet, EfficientNet) via `segmentation_models_pytorch` (SMP), parameter audits, and fine-tuning dynamics (freeze vs. unfreeze).
3. **Single-Channel Grayscale Adaptation:** Mathematically adapting 3-channel RGB pretrained weights to single-channel industrial line-scan sensors (3-channel duplication vs. first-layer weight collapse).
4. **Multi-Scale Context & Atrous Convolutions:** Analyzing DeepLabV3+ with Atrous Spatial Pyramid Pooling (ASPP) to detect both fine cracks and large diffuse stains without spatial resolution collapse.
5. **Compound Loss Formulations:** Replacing plain BCE with differentiable **Soft Dice Loss** and **Compound Loss ($\alpha \text{BCE} + \beta \text{Dice}$)** to handle extreme foreground sparsity.
6. **Output Head Semantics:** Comparing 4-channel independent Sigmoid multi-label detection with 5-channel mutually-exclusive Softmax partitions.
7. **Comparative Benchmarking:** Evaluating performance gains, convergence speed, and parameter trade-offs between scratch U-Net, ResNet-34, and EfficientNet-B0.

---

## 🛠️ Official Starter Code & Default Parameters Reference
Use `segmentation-models-pytorch==0.5.0`, `torch==2.8.0` and `torchvision==0.23.0`; record your Python/CUDA/device versions too. In a local terminal, create a separate environment with `uv venv --python 3.11 .venv-milestone3` and install with `uv pip install --python .venv-milestone3/bin/python torch==2.8.0 torchvision==0.23.0 segmentation-models-pytorch==0.5.0`. Use the equivalent environment Python path on Windows. ImageNet weights require a download/cache; use `encoder_weights=None` for an offline **parameter audit only**, not as a substitute for pretraining in Q8/Q9.

Use the explicit architecture settings below. All models emit raw logits. The four-pooling network in Q1 is a separate mathematical example, not this five-stage SMP encoder.

### 1. Model Instantiation & Freezing (for Questions 2, 8, 9)
```python
import torch
import segmentation_models_pytorch as smp

# ResNet-34 U-Net (Model A)
model_resnet = smp.Unet(
    encoder_name="resnet34",        # Pretrained backbone
    encoder_weights="imagenet",     # Load ImageNet feature hierarchy
    in_channels=1,                  # Grayscale industrial strip
    classes=4,                      # 4 independent defect channels
    encoder_depth=5,
    decoder_channels=(256, 128, 64, 32, 16),
    decoder_use_norm="batchnorm",
    decoder_interpolation="nearest",
    activation=None
)

# EfficientNet-B0 U-Net (Model B)
model_effnet = smp.Unet(
    encoder_name="efficientnet-b0",
    encoder_weights="imagenet",
    in_channels=1,
    classes=4,
    encoder_depth=5,
    decoder_channels=(256, 128, 64, 32, 16),
    decoder_use_norm="batchnorm",
    decoder_interpolation="nearest",
    activation=None
)

# Parameter-count exercise for Q2 only; instantiate fresh unfrozen models for Q8/Q9:
for param in model_resnet.encoder.parameters():
    param.requires_grad = False
```

Setting `requires_grad=False` freezes parameters, not BatchNorm running statistics. For a deliberately fixed feature extractor, call `model.encoder.eval()` after every `model.train()`. Q8/Q9 instead train all parameters with normal training-mode BatchNorm.

### 2. Differentiable Soft Dice & Compound Loss (for Questions 5, 7, 8, 9)

This Dice is one global ratio over the entire batch, four channels and spatial dimensions. BCE uses mean reduction. It is not a per-image/per-class Dice average. The all-empty derivation in Q7 applies to a wholly empty reduction; a clean sample in a mixed batch is coupled to other samples.
```python
import torch.nn as nn

class SoftDiceLoss(nn.Module):
    """Differentiable Soft Dice Loss with Laplace smoothing."""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat = probs.reshape(-1)
        targets_flat = targets.reshape(-1)
        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        return 1.0 - dice

class CompoundLoss(nn.Module):
    """Compound Loss: alpha * BCE + beta * SoftDice."""
    def __init__(self, alpha=0.5, beta=0.5, smooth=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = SoftDiceLoss(smooth=smooth)
        self.alpha = alpha
        self.beta = beta

    def forward(self, logits, targets):
        return self.alpha * self.bce(logits, targets) + self.beta * self.dice(logits, targets)
```

### 3. Common Training Protocol (for Questions 8 & 9)
```python
# Standard hyperparameters for benchmarking
BATCH_SIZE = 16
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-2
EPOCHS = 5
ALPHA = 0.5   # BCE weight
BETA = 0.5    # Soft Dice weight
SMOOTH = 1.0  # Laplace smoothing parameter

criterion = CompoundLoss(alpha=ALPHA, beta=BETA, smooth=SMOOTH)
# For each run, first move a fresh unfrozen model to the selected device:
# model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
```

For Q8 compare two **ResNet-34 U-Nets of identical architecture**, changing only encoder initialization (`None` versus `"imagenet"`). Match initial decoder/head state dictionaries as shown in Guide 8. Q9 compares ImageNet-pretrained ResNet-34 and EfficientNet-B0; the pretrained ResNet run may be reused.

Use this common protocol for all three runs:

- Reuse the image-level train/validation split saved in Milestone 2, with no image ID in both partitions. Record its manifest/hash and class support. Use the same complete partitions for every model; do not use test labels or leaderboard feedback to select a result.
- Load images as grayscale float32 in [0,1] at native height × width 128 × 800; masks are float32 binary [4,128,800] in defect_a through defect_d order. Apply no augmentations or additional mean/std normalization for this controlled baseline. This is a declared input convention, not a claim that [0,1] is optimal for pretraining.
- Set Python, NumPy and torch seeds to 42 before each run. Reset the DataLoader generator to seed 42; use num_workers=0, shuffle=True only for training, and drop_last=False. Save sample ordering with your run metadata when possible.
- Train all parameters, including the encoder, for exactly five epochs in float32 with fresh AdamW state, constant learning rate and no scheduler, early stopping, AMP or gradient accumulation. Use model.train() for optimization and model.eval() with inference_mode() for evaluation. Evaluate the epoch-5 model rather than selecting a best epoch.
- Use batch size 16 for both optimization and evaluation, including the final partial batch. Report a CUDA OOM rather than silently changing one model's batch size. Use batch loss multiplied by batch image count, divided by total image count, for comparable epoch loss logs; note that the Dice term still depends on batch composition.
- After each epoch, evaluate the train and validation partitions in eval mode without shuffling to obtain comparable loss measurements. Define the gap as |validation_eval_loss - train_eval_loss|. Record online training loss separately.
- For hard-mask metrics, apply sigmoid followed by >=0.5 independently per defect channel, with no area filtering or threshold search. Accumulate image-class pairs over the entire validation set.

The official score is **0.70 ForegroundDice + 0.20 FalsePositiveScore + 0.10 EmptyAccuracy**, using [Milestone 1 Q8's component definitions](MILESTONE_1.md#question-8-official-competition-weighted-metric-toy-batch-calculation). ForegroundDice averages binary Dice only over positive target pairs; FalsePositiveScore is one minus mean predicted-positive pixel fraction on empty target pairs; EmptyAccuracy is the fraction of empty target pairs predicted entirely empty. If a denominator is missing, report it rather than fabricate a score.

Treat compared scalar values within absolute 1e-6 as ties. Record measured values and setup even if an observed result differs from your expectation. Hardware and library nondeterminism can prevent identical results across devices.

---

## Question 1: Feature Map Spatial Geometry & Downsampling Arithmetic
In semantic segmentation, the encoder compresses input features through a series of pooling or strided convolution operations. In standard natural image segmentation (e.g. $256 \times 256$), dimensions downsample symmetrically. However, industrial line-scan strips have an extreme anisotropic aspect ratio: $128 \times 800$ ($1:6.25$).

A semantic segmentation network processes an industrial line-scan steel strip of native shape $(1, 128, 800)$ across 4 progressive stride-2 downsampling stages:
$$\text{Input: } (128, 800) \xrightarrow{/2} \text{Stage 1: } (64, 400) \xrightarrow{/2} \text{Stage 2: } (32, 200) \xrightarrow{/2} \text{Stage 3: } (16, 100) \xrightarrow{/2} \text{Stage 4 (Bottleneck): } (H_{\text{bottle}}, W_{\text{bottle}})$$

**Question:**
1. What is the spatial shape $(H_{\text{bottle}}, W_{\text{bottle}})$ of the feature representation at the Stage 4 bottleneck?
2. If an unpadded image strip with an odd width of $128 \times 799$ is passed through this 4-stage encoder with standard integer floor pooling (`kernel_size=2, stride=2`), what is the absolute width mismatch at the **first decoder skip concatenation**, immediately after doubling the Stage 4 bottleneck width and comparing it with the Stage 3 encoder skip? Assume exact 2× upsampling and no padding, cropping or interpolation-to-size before this comparison. Do not report the full-canvas reconstruction deficit.
3. What is the total spatial downsampling factor (ratio of input canvas area to bottleneck spatial area)?

**Answer format:** `H_bottle, W_bottle; first_skip_mismatch_px; downsample_factor`
*(Use the named fields above; example values are intentionally omitted.)*

---

## Question 2: Pretrained Backbone Architecture & Parameter Audit
Training a deep segmentation model from scratch on small domain datasets often struggles because early layers must learn basic visual primitives (edges, gradients, Gabor textures) from scratch. Transfer learning addresses this by using backbones pretrained on ImageNet ($1.2\text{M}$ natural images).

Consider SMP U-Net with ResNet-34 (`encoder_name="resnet34"`) versus EfficientNet-B0 (`encoder_name="efficientnet-b0"`), both instantiated with `in_channels=1` and `classes=4` using the pinned SMP version and explicit settings in Starter Code snippet 1:

**Question:**
1. Compute the total parameter count (weights + biases across encoder, decoder, and segmentation head) of both models. Which architecture is more parameter-efficient, and what is the exact difference in parameter count between them?
2. In feature-extractor transfer learning, the backbone encoder is often frozen (`for p in model.encoder.parameters(): p.requires_grad = False`) while keeping the decoder and segmentation head trainable. For the ResNet-34 U-Net, what exact percentage of Model A's total parameters remain trainable? Round to 1 decimal place.

**Answer format:** `more_efficient_model, param_diff; trainable_pct`  
*(Use the named fields above; example values are intentionally omitted.)*

---

## Question 3: Grayscale Input Adaptation (The Single-Channel Dilemma)
Pretrained ImageNet backbones expect 3-channel RGB inputs ($C_{\text{in}} = 3$). Industrial steel line-scan cameras produce 1-channel grayscale images ($C_{\text{in}} = 1$).
- **Strategy A (Channel Duplication):** Duplicate the grayscale image across 3 channels in the DataLoader ($X_{\text{rep}} = [I, I, I]$) so the 3-channel first layer (`conv1`) can run untouched.
- **Strategy B (First-Layer Weight Collapse):** Modify the first convolutional layer (`conv1`) to accept $C_{\text{in}} = 1$ by mathematically combining the 3-channel pretrained RGB weights: $W_{\text{gray}} = f(W_{\text{rgb}})$.

**Assumptions:** The duplicated channels have identical values (no distinct per-channel RGB normalization), both convolutions have identical padding/stride and the bias is copied unchanged. Compare image tensor bytes only, at the same batch size, dimensions and dtype; exclude masks and downstream activations.

**Question:**
1. Mathematically, what exact operation $f(W_{\text{rgb}})$ across the channel dimension—**sum** or **mean**—must be applied so that the first-layer output activations of Strategy B are strictly identical to Strategy A for any input image $I$?
2. In terms of GPU memory bandwidth and PCIe bus transfer from host to device, by what factor does Strategy B reduce the memory transfer size of the input image batch compared to Strategy A?

**Answer format:** `sum_or_mean, memory_factor`  
*(Use the named fields above; example values are intentionally omitted.)*

---

## Question 4: Multi-Scale Receptive Fields & Atrous Convolutions (DeepLabV3+ vs. U-Net)
Consider fictional thin scratches and broad diffuse patches, without assigning those morphologies to competition class names. A U-Net gathers context through both convolution and downsampling, while skip connections carry higher-resolution features to the decoder.

DeepLabV3+ uses Atrous (Dilated) Convolutions in an Atrous Spatial Pyramid Pooling (ASPP) module. At fixed output size, channels and kernel size, dilation expands footprint without adding kernel parameters. Parallel ASPP branches still add computation and do not restore information already lost by downsampling.
For a $k \times k$ convolutional kernel with dilation rate $r$, the effective spatial footprint $k'$ is given by:
$$k' = k + (k - 1)(r - 1)$$

**Question:**
1. For a standard $3 \times 3$ kernel ($k=3$), what is the effective spatial footprint $k'$ when the dilation rate is $r = 6$, and when $r = 12$?
2. In 1–2 concise sentences: describe a potential advantage and one limitation of ASPP's multi-scale context compared with a single-scale pooling bottleneck when segmenting thin scratches and broad diffuse patches on $128 \times 800$ strips.

**Answer format:** `k_eff_r6, k_eff_r12; brief_advantage`  
*(Use the named fields above; example values are intentionally omitted.)*

---

## Question 5: Differentiable Soft Dice Loss & Numerical Smoothing
In an imbalanced segmentation task, background predictions can dominate pixel accuracy. BCE still penalizes missed positives; Soft Dice offers a different overlap-based objective. Use the following synthetic population for arithmetic, not as a claim about the competition distribution.

Consider a synthetic patch of $10{,}000$ pixels containing exactly **100 defect pixels** ($y_i = 1$) and **9,900 clean background pixels** ($y_i = 0$).
Use the differentiable Soft Dice Loss formula with Laplace additive smoothing ($\epsilon = 1.0$) as defined in Starter Code snippet 2:
$$\mathcal{L}_{\text{Dice}}(P, Y) = 1 - \frac{2 \sum_{i=1}^N p_i y_i + \epsilon}{\sum_{i=1}^N p_i + \sum_{i=1}^N y_i + \epsilon}$$

**Question:**
1. **Lazy Model 1:** Predicts $p_i = 0.01$ uniformly across all $10{,}000$ pixels. What is its exact Soft Dice Loss?
2. **Accurate Model 2:** Predicts $p_i = 0.90$ on all 100 defect pixels, and $p_i = 0.00$ on all 9,900 background pixels. What is its exact Soft Dice Loss?
Round both answers to 4 decimal places.

**Answer format:** `loss_model1, loss_model2`  
*(Use the named fields above; example values are intentionally omitted.)*

---

## Question 6: Output Head Semantics & Mutually Exclusive Loss Formulations
Recall your own pixel-level overlap audit from Milestone 1 Q3. The questions below concern the capabilities of two formulations; do not assume a training-data finding guarantees what future annotations can contain.
- **Formulation A (Multi-Label Sigmoid):** 4 independent output channels with Sigmoid activations, trained with `BCEWithLogitsLoss`. Each defect channel is predicted independently.
- **Formulation B (Multi-Class Softmax):** 5 output channels (Background + 4 defect classes) with Softmax activation, trained with `CrossEntropyLoss`. The network partitions each pixel into exactly one category.

**Question:**
1. Under Formulation B (5-channel Softmax), if a steel strip contains `defect_a` on the left half of the strip and `defect_c` on the right half, can the model correctly segment both defect instances on the same image without mutual interference? (Yes/No)
2. What is the fundamental disadvantage or operational failure mode of Formulation B if test data or future production batches ever contain overlapping defect annotations? (1 sentence).

**Answer format:** `Yes/No; brief_disadvantage`  
*(Use the named fields above; example values are intentionally omitted.)*

---

## Question 7: Compound Loss Behavior on Defect vs. Clean Images
Modern segmentation pipelines combine BCE and Soft Dice into a Compound Loss:
$$\mathcal{L}_{\text{Compound}} = 0.5 \times \mathcal{L}_{\text{BCE}} + 0.5 \times \mathcal{L}_{\text{Dice}}$$

Consider an entirely empty target reduction ($y_i=0$ everywhere), smoothing $\epsilon=1$ and probabilities $p_i=\sigma(z_i)$. Let $S=\sum_i p_i$. This can describe an isolated clean image or an all-clean batch, but not the contribution of a clean image in a mixed batch under the provided global Dice reduction.

**Question:**
1. Derive $L_{Dice}$ as a function of $S$, then its derivatives with respect to $p_j$ and $z_j$. Explain how the logit gradient behaves as total predicted mass increases and as probabilities approach sigmoid saturation. Distinguish a large scalar loss from a large gradient.
2. In 1–2 sentences, explain what BCE can add to the all-empty logit signal, and why this does not prove that the compound loss must always outperform Dice alone on validation data.

**Answer format:** `brief_gradient_explanation; brief_compound_justification`

---

## Question 8: Scratch Baseline vs. Pretrained Transfer Learning (Empirical Training & Convergence)
Train two ResNet-34 U-Nets of identical architecture with random versus ImageNet encoder initialization under the common protocol. This isolates initialization rather than comparing different architectures. There is no assumed winner.

**Question:**
1. At epoch 5, which model has the lower validation loss, and which has the higher official score? Report these separately (`pretrained` / `scratch` / `tie`). Include both models' measured losses and scores in a small results table.
2. At epoch 1, which model has the higher validation ForegroundDice? (`pretrained` / `scratch` / `tie`). Report both values.
3. Which has the smaller eval-mode generalization gap $|L_{val,eval}-L_{train,eval}|$ at epoch 5? (`pretrained` / `scratch` / `tie`). Report both gaps; a smaller gap alone is not proof of better validation performance.
4. Report the signed score difference $\Delta=Score_{pretrained}-Score_{scratch}$ and its band: `delta < 0.02`, `0.02 <= delta <= 0.08`, `0.08 < delta <= 0.10`, or `delta > 0.10`. The first band includes zero and negative differences.

**Answer format:** `loss_winner; score_winner; epoch1_dice_winner; smaller_gap_model; signed_delta; delta_band`, followed by the measured results table and split/seed/version metadata.

---

## Question 9: Pretrained Backbone Architecture Trade-off (ResNet-34 vs. EfficientNet-B0)
Compare ImageNet-pretrained ResNet-34 and EfficientNet-B0 U-Nets trained for five epochs with the same protocol as Q8. Reuse Q2's parameter audit; no fixed memory or score ordering is assumed.

**Question:**
1. Divide the total ResNet-34 model parameter count by the total EfficientNet-B0 count. Round the ratio to the nearest integer.
2. If CUDA is available, measure both models' **training peak allocated CUDA memory** at batch size 16, native resolution and float32, using Guide 9's warm-up/reset/complete-step procedure in fresh processes on the same GPU. Report both peaks in MiB and the lower-memory model (`resnet34` / `efficientnet-b0` / `tie`). Use `not_measured` on CPU-only hardware or `oom` for a model that cannot complete the specified workload. Explain why this training result alone does not establish edge inference suitability.
3. Report both epoch-5 official scores. For this assessment, call EfficientNet-B0 “competitive” if its score is at least ResNet's score minus 0.01; a higher score also qualifies. Does your measurement meet that declared tolerance? (`Yes` / `No`).
4. Explain the convolution mechanism in EfficientNet's MBConv blocks that can reduce parameter count, including why the full block has more components than a depthwise layer alone.

**Answer format:** `param_ratio; resnet_peak_MiB, efficientnet_peak_MiB, lower_memory_model; resnet_score, efficientnet_score, competitive_yes_no; convolution_explanation`, plus the workload/device metadata and memory-versus-deployment caveat.
