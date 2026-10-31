# Milestone 3: Transfer Learning & Stronger Segmentation Models

Industrial Surface Anomaly Segmentation · Milestone 3 Assessment

Read the [student study guides](docs/milestone_3_student_guide/README.md) before attempting the questions. Use the competition data in `data/public/` and compute your own results. This document contains questions and response formats only.

For Questions 8 and 9, train your models using the official starter code and standard training protocol provided below.

---

## Overview
In Milestone 2, you built the data foundation: a custom PyTorch `Dataset` and `DataLoader` pipeline and trained a basic from-scratch U-Net baseline. However, training semantic segmentation models from scratch on highly imbalanced surface anomaly datasets often plateaus early—the network lacks generic edge/texture visual priors and struggles with extreme background dominance (>97% clean pixels).

**Milestone 3 introduces production-grade deep learning vision techniques:**
1. **Encoder–Decoder Spatial Mechanics:** Feature map downsampling, bottleneck compression, skip-connection concatenation, and receptive field expansion on anisotropic ($128 \times 800$) sensor ribbons.
2. **Transfer Learning with Pretrained Backbones:** Leveraging ImageNet feature hierarchies (ResNet, EfficientNet) via `segmentation_models_pytorch` (SMP), parameter audits, and fine-tuning dynamics (freeze vs. unfreeze).
3. **Single-Channel Grayscale Adaptation:** Mathematically adapting 3-channel RGB pretrained weights to single-channel industrial line-scan sensors (3-channel duplication vs. first-layer weight collapse).
4. **Multi-Scale Context & Atrous Convolutions:** Analyzing DeepLabV3+ with Atrous Spatial Pyramid Pooling (ASPP) to detect both fine cracks and large diffuse stains without spatial resolution collapse.
5. **Compound Loss Formulations:** Replacing plain BCE with differentiable **Soft Dice Loss** and **Compound Loss ($\alpha \text{BCE} + \beta \text{Dice}$)** to handle extreme foreground sparsity.
6. **Output Head Semantics:** Comparing 4-channel independent Sigmoid multi-label detection with 5-channel mutually-exclusive Softmax partitions.
7. **Comparative Benchmarking:** Evaluating performance gains, convergence speed, and parameter trade-offs between scratch U-Net, ResNet-34, and EfficientNet-B0.

---

## 🛠️ Official Starter Code & Default Parameters Reference
To eliminate parameter guesswork and ensure complete consistency across implementations, use the standard definitions below.

### 1. Model Instantiation & Freezing (for Questions 2, 8, 9)
```python
import torch
import segmentation_models_pytorch as smp

# ResNet-34 U-Net (Model A)
model_resnet = smp.Unet(
    encoder_name="resnet34",        # Pretrained backbone
    encoder_weights="imagenet",     # Load ImageNet feature hierarchy
    in_channels=1,                  # Grayscale industrial strip
    classes=4                       # 4 independent defect channels
)

# EfficientNet-B0 U-Net (Model B)
model_effnet = smp.Unet(
    encoder_name="efficientnet-b0",
    encoder_weights="imagenet",
    in_channels=1,
    classes=4
)

# Freezing Encoder (for Q2):
for param in model_resnet.encoder.parameters():
    param.requires_grad = False
```

### 2. Differentiable Soft Dice & Compound Loss (for Questions 5, 7, 8, 9)
```python
import torch.nn as nn

class SoftDiceLoss(nn.Module):
    """Differentiable Soft Dice Loss with Laplace smoothing."""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
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
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
```

---

## Question 1: Feature Map Spatial Geometry & Downsampling Arithmetic
In semantic segmentation, the encoder compresses input features through a series of pooling or strided convolution operations. In standard natural image segmentation (e.g. $256 \times 256$), dimensions downsample symmetrically. However, industrial line-scan strips have an extreme anisotropic aspect ratio: $128 \times 800$ ($1:6.25$).

A semantic segmentation network processes an industrial line-scan steel strip of native shape $(1, 128, 800)$ across 4 progressive stride-2 downsampling stages:
$$\text{Input: } (128, 800) \xrightarrow{/2} \text{Stage 1: } (64, 400) \xrightarrow{/2} \text{Stage 2: } (32, 200) \xrightarrow{/2} \text{Stage 3: } (16, 100) \xrightarrow{/2} \text{Stage 4 (Bottleneck): } (H_{\text{bottle}}, W_{\text{bottle}})$$

**Question:**
1. What is the spatial shape $(H_{\text{bottle}}, W_{\text{bottle}})$ of the feature representation at the Stage 4 bottleneck?
2. If an unpadded image strip with an odd width of $128 \times 799$ is passed through this 4-stage encoder with standard integer floor pooling (`kernel_size=2, stride=2`), what is the spatial dimension mismatch (in pixels along width) that occurs when attempting the final skip-connection concatenation at the decoder?
3. What is the total spatial downsampling factor (ratio of input canvas area to bottleneck spatial area)?

**Answer format:** `H_bottle, W_bottle; mismatch_px; downsample_factor`  
*(Example format: `16, 100; 0; 64`)*

---

## Question 2: Pretrained Backbone Architecture & Parameter Audit
Training a deep segmentation model from scratch on small domain datasets often struggles because early layers must learn basic visual primitives (edges, gradients, Gabor textures) from scratch. Transfer learning addresses this by using backbones pretrained on ImageNet ($1.2\text{M}$ natural images).

Consider SMP U-Net with ResNet-34 (`encoder_name="resnet34"`) versus EfficientNet-B0 (`encoder_name="efficientnet-b0"`), both instantiated with `in_channels=1` and `classes=4` using default SMP configurations (see Starter Code snippet 1):

**Question:**
1. Compute the total parameter count (weights + biases across encoder, decoder, and segmentation head) of both models. Which architecture is more parameter-efficient, and what is the exact difference in parameter count between them?
2. In feature-extractor transfer learning, the backbone encoder is often frozen (`for p in model.encoder.parameters(): p.requires_grad = False`) while keeping the decoder and segmentation head trainable. For the ResNet-34 U-Net, what exact percentage of Model A's total parameters remain trainable? Round to 1 decimal place.

**Answer format:** `more_efficient_model, param_diff; trainable_pct`  
*(Example format: `resnet18, 5000000; 25.0%`)*

---

## Question 3: Grayscale Input Adaptation (The Single-Channel Dilemma)
Pretrained ImageNet backbones expect 3-channel RGB inputs ($C_{\text{in}} = 3$). Industrial steel line-scan cameras produce 1-channel grayscale images ($C_{\text{in}} = 1$).
- **Strategy A (Channel Duplication):** Duplicate the grayscale image across 3 channels in the DataLoader ($X_{\text{rep}} = [I, I, I]$) so the 3-channel first layer (`conv1`) can run untouched.
- **Strategy B (First-Layer Weight Collapse):** Modify the first convolutional layer (`conv1`) to accept $C_{\text{in}} = 1$ by mathematically combining the 3-channel pretrained RGB weights: $W_{\text{gray}} = f(W_{\text{rgb}})$.

**Question:**
1. Mathematically, what exact operation $f(W_{\text{rgb}})$ across the channel dimension—**sum** or **mean**—must be applied so that the first-layer output activations of Strategy B are strictly identical to Strategy A for any input image $I$?
2. In terms of GPU memory bandwidth and PCIe bus transfer from host to device, by what factor does Strategy B reduce the memory transfer size of the input image batch compared to Strategy A?

**Answer format:** `sum_or_mean, memory_factor`  
*(Example format: `mean, 2x`)*

---

## Question 4: Multi-Scale Receptive Fields & Atrous Convolutions (DeepLabV3+ vs. U-Net)
In surface defect inspection, anomalies span radically different scales: `defect_a` consists of hairline cracks (only 1–3 pixels wide), whereas `defect_c` consists of massive diffuse patches spanning hundreds of pixels. Traditional U-Nets expand their receptive field solely by aggressive downsampling (pooling), which destroys fine spatial boundary details.

DeepLabV3+ solves this trade-off using Atrous (Dilated) Convolutions in an Atrous Spatial Pyramid Pooling (ASPP) module, expanding the effective receptive field without downsampling or increasing parameter count.
For a $k \times k$ convolutional kernel with dilation rate $r$, the effective spatial footprint $k'$ is given by:
$$k' = k + (k - 1)(r - 1)$$

**Question:**
1. For a standard $3 \times 3$ kernel ($k=3$), what is the effective spatial footprint $k'$ when the dilation rate is $r = 6$, and when $r = 12$?
2. In 1–2 concise sentences: what is the fundamental architectural advantage of ASPP's multi-scale dilated convolutions over a traditional pooling bottleneck when segmenting both thin scratches (`defect_a`) and large diffuse patches (`defect_c`) on $128 \times 800$ steel sheets?

**Answer format:** `k_eff_r6, k_eff_r12; brief_advantage`  
*(Example format: `7x7, 15x15; Dilated convolutions expand receptive field without resolution loss`)*

---

## Question 5: Differentiable Soft Dice Loss & Numerical Smoothing
In surface anomaly detection, over 97% of canvas pixels are clean background. Standard Binary Cross-Entropy (BCE) computes pixel-wise loss independently, allowing the massive volume of background pixels to overwhelm gradients and biasing models toward predicting zero everywhere.

Consider a synthetic patch of $10{,}000$ pixels containing exactly **100 defect pixels** ($y_i = 1$) and **9,900 clean background pixels** ($y_i = 0$).
Use the differentiable Soft Dice Loss formula with Laplace additive smoothing ($\epsilon = 1.0$) as defined in Starter Code snippet 2:
$$\mathcal{L}_{\text{Dice}}(P, Y) = 1 - \frac{2 \sum_{i=1}^N p_i y_i + \epsilon}{\sum_{i=1}^N p_i + \sum_{i=1}^N y_i + \epsilon}$$

**Question:**
1. **Lazy Model 1:** Predicts $p_i = 0.01$ uniformly across all $10{,}000$ pixels. What is its exact Soft Dice Loss?
2. **Accurate Model 2:** Predicts $p_i = 0.90$ on all 100 defect pixels, and $p_i = 0.00$ on all 9,900 background pixels. What is its exact Soft Dice Loss?
Round both answers to 4 decimal places.

**Answer format:** `loss_model1, loss_model2`  
*(Example format: `0.8521, 0.1245`)*

---

## Question 6: Output Head Semantics & Mutually Exclusive Loss Formulations
In Milestone 1 (Q3), empirical data auditing verified that different defect classes are strictly spatially mutually exclusive at the pixel level (0 overlapping pixels across the entire dataset).
- **Formulation A (Multi-Label Sigmoid):** 4 independent output channels with Sigmoid activations, trained with `BCEWithLogitsLoss`. Each defect channel is predicted independently.
- **Formulation B (Multi-Class Softmax):** 5 output channels (Background + 4 defect classes) with Softmax activation, trained with `CrossEntropyLoss`. The network partitions each pixel into exactly one category.

**Question:**
1. Under Formulation B (5-channel Softmax), if a steel strip contains `defect_a` on the left half of the strip and `defect_c` on the right half, can the model correctly segment both defect instances on the same image without mutual interference? (Yes/No)
2. What is the fundamental disadvantage or operational failure mode of Formulation B if test data or future production batches ever contain overlapping defect annotations? (1 sentence).

**Answer format:** `Yes/No; brief_disadvantage`  
*(Example format: `Yes; Softmax prevents multiple classes from occupying the same pixel`)*

---

## Question 7: Compound Loss Behavior on Defect vs. Clean Images
Modern segmentation pipelines combine BCE and Soft Dice into a Compound Loss:
$$\mathcal{L}_{\text{Compound}} = 0.5 \times \mathcal{L}_{\text{BCE}} + 0.5 \times \mathcal{L}_{\text{Dice}}$$

Consider a completely clean image ($|Y| = 0$, no defect pixels):
- If predictions are completely empty ($\sum p_i = 0$), smoothed Dice is $1.0 \implies \mathcal{L}_{\text{Dice}} = 0.0$.
- If the model predicts even 50 stray false-positive noise pixels ($\sum p_i = 50$), smoothed Dice is $\frac{0 + 1}{50 + 0 + 1} = \frac{1}{51} \approx 0.0196 \implies \mathcal{L}_{\text{Dice}} \approx 0.9804$.

**Question:**
1. For a clean image ($|Y| = 0$), describe the gradient behavior that $\mathcal{L}_{\text{Dice}}$ produces when stray noise pixels appear versus when predictions are clean.
2. Why is combining BCE with Soft Dice strictly superior to training with Soft Dice alone on highly imbalanced surface anomaly datasets where many images are clean? (1–2 sentences).

**Answer format:** `brief_gradient_explanation; brief_compound_justification`

---

## Question 8: Scratch Baseline vs. Pretrained Transfer Learning (Empirical Training & Convergence)
Train both a from-scratch U-Net and an ImageNet-pretrained ResNet-34 U-Net for 5 epochs under identical settings using the standard training protocol defined in Starter Code snippet 3: AdamW optimizer (learning rate = $3 \times 10^{-4}$), batch size = 16, and Compound Loss ($0.5 \times \text{BCE} + 0.5 \times \text{SoftDice}$).

**Question:**
1. By Epoch 5, which model achieves a lower validation loss and higher official competition score? (`pretrained` / `scratch`)
2. Convergence Speed: In the very first epoch (Epoch 1), which model demonstrates faster foreground feature acquisition and substantially higher foreground Dice? (`pretrained` / `scratch`)
3. Generalization: Does the pretrained model exhibit smaller overfitting / a tighter generalization gap ($|L_{\text{val}} - L_{\text{train}}|$) compared to training from scratch? (`Yes` / `No`)
4. What is the approximate range of competition score improvement gained by using the pretrained ResNet-34 backbone over the scratch baseline on the validation split? Choose one: `[< +0.02, +0.02 to +0.08, > +0.10]`.

**Answer format:** `winning_model; faster_epoch1_model; tighter_gap_yes_no; score_improvement_range`  
*(Example format: `scratch; scratch; No; < +0.02`)*

---

## Question 9: Pretrained Backbone Architecture Trade-off (ResNet-34 vs. EfficientNet-B0)
Compare ResNet-34 U-Net ($24.4\text{M}$ parameters) and EfficientNet-B0 U-Net ($6.25\text{M}$ parameters) trained for 5 epochs using the same protocol as Question 8.

**Question:**
1. Parameter Ratio: Approximately how many times more parameters does ResNet-34 have compared to EfficientNet-B0? (Round to nearest integer).
2. Resource Footprint: Which architecture consumes significantly less peak GPU VRAM and is better suited for edge deployment with strict memory budgets? (`resnet34` / `efficientnet-b0`)
3. Performance: Can the lightweight EfficientNet-B0 achieve a competitive or higher official competition score than ResNet-34 despite having $\sim 74\%$ fewer parameters? (`Yes` / `No`)
4. Architectural Mechanism: Name the specific convolution technique used in EfficientNet that drastically reduces parameter count compared to standard convolutions in ResNet.

**Answer format:** `param_ratio; edge_backbone; competitive_yes_no; convolution_technique`  
*(Example format: `2x; resnet34; No; standard residual convolutions`)*
