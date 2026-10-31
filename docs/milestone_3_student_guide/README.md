# Milestone 3 Student Study Guides
## Transfer Learning & Stronger Segmentation Models

Welcome to Milestone 3! In Milestone 1, you inspected raw annotations, unraveled Run-Length Encoding (RLE) memory layouts, explored line-scan sensor geometry, and studied metric sensitivity. In Milestone 2, you moved from analytical observation to active systems engineering: building custom PyTorch `Dataset` and `DataLoader` pipelines, implementing industrial geometric augmentations, establishing leakage-free validation splits, and training a baseline from-scratch U-Net.

In **Milestone 3**, you elevate your modeling capability to industrial production standards:
- **Spatial Geometry & Bottleneck Compression:** Tracing feature map dimensions through progressive stride-2 downsampling, understanding receptive field expansion, and diagnosing odd-dimension tensor parity mismatches along anisotropic ($128 \times 800$) ribbons.
- **Transfer Learning with Pretrained Vision Backbones:** Leveraging generic visual primitives (edges, gradients, textures) learned from ImageNet ($1.2\text{M}$ natural images) across ResNet and EfficientNet architectures, auditing model parameters, and mastering feature-extractor freezing dynamics.
- **Grayscale Sensor Adaptation:** Solving the impedance mismatch between 3-channel RGB pretrained weights and 1-channel monochromatic industrial cameras, proving activation equivalence between channel duplication and first-layer weight collapse, and evaluating PCIe memory bandwidth efficiency.
- **Multi-Scale Context via Atrous (Dilated) Convolutions:** Analyzing DeepLabV3+ with Atrous Spatial Pyramid Pooling (ASPP) to detect both microscopic hairline cracks (1–3 pixels wide) and massive diffuse patches without sacrificing spatial resolution or multiplying parameters.
- **Differentiable Soft Dice & Compound Loss Formulations:** Taming extreme foreground sparsity (>97% clean canvas) using smoothed Soft Dice loss ($\epsilon = 1.0$), analyzing gradient behavior on clean sheets, and anchoring training dynamics with Compound Loss ($\alpha \text{BCE} + \beta \text{Dice}$).
- **Output Head Semantics & Structural Partitioning:** Evaluating 4-channel independent Sigmoid multi-label detection against 5-channel mutually exclusive Softmax partitions, and identifying catastrophic failure modes under label overlap.
- **Empirical Benchmarking & Architecture Trade-offs:** Systematically measuring convergence speed, generalization gaps, parameter throughput, and memory trade-offs between scratch baselines, ResNet-34, and EfficientNet-B0.

---

## Curriculum Map: Questions & Guided Learning Modules

Every study guide corresponds directly to an assessment question in [`MILESTONE_3.md`](../../MILESTONE_3.md). Study the core theoretical concepts, review the step-by-step diagnostic hints, and compute your results using the official starter code.

| Milestone Question | Guided Learning Document | Theoretical Principles | Guiding Focus & Hints |
| :--- | :--- | :--- | :--- |
| **Question 1** | [Guide 1: Feature Map Spatial Geometry & Downsampling](guide_01_feature_map_geometry_and_downsampling.md) | Contracting paths, stride-2 downsampling, bottleneck compression, skip-connection concatenation, odd-width floor-division parity mismatch. | Tracing tensor shapes $(H_s, W_s)$, odd parity truncation formula $\lfloor W/2 \rfloor$, canvas vs. bottleneck area ratio. |
| **Question 2** | [Guide 2: Pretrained Backbone Architecture & Parameter Audit](guide_02_pretrained_backbone_and_parameter_audit.md) | Inductive priors, ImageNet transfer learning, parameter accounting across encoder, decoder, and segmentation head, feature-extractor freezing dynamics. | Counting total vs. trainable parameters via `numel()`, freezing `model.encoder`, computing trainable ratio percentage. |
| **Question 3** | [Guide 3: Grayscale Input Adaptation & Channel Physics](guide_03_grayscale_adaptation_and_channel_physics.md) | Monochromatic line-scan cameras vs. RGB pretrained filters, channel duplication ($[I,I,I]$) vs. weight collapse ($W_{\text{gray}} = f(W_{\text{rgb}})$), mathematical activation equivalence proof, host-to-device PCIe bandwidth. | Mathematical derivation of channel summation, PyTorch `torch.allclose` verification, memory transfer scaling factor. |
| **Question 4** | [Guide 4: Multi-Scale Receptive Fields & Atrous Convolutions](guide_04_atrous_convolutions_and_multiscale_aspp.md) | Spatial receptive field expansion, dilated (atrous) convolutions, effective kernel footprint $k' = k + (k-1)(r-1)$, DeepLabV3+ Atrous Spatial Pyramid Pooling (ASPP). | Derivation of effective spatial footprint for dilation rates $r=6, 12$, multi-scale context without resolution collapse. |
| **Question 5** | [Guide 5: Differentiable Soft Dice Loss & Numerical Smoothing](guide_05_soft_dice_loss_and_laplace_smoothing.md) | Extreme background imbalance (>97% clean canvas), differentiable continuous relaxation of Dice, Laplace additive smoothing ($\epsilon=1.0$), false-positive background penalties. | Computing Soft Dice step-by-step on synthetic patches, Lazy Model vs. Accurate Model loss dynamics. |
| **Question 6** | [Guide 6: Output Head Semantics & Loss Formulations](guide_06_output_heads_and_target_semantics.md) | 4-channel independent Sigmoid (`BCEWithLogitsLoss`) vs. 5-channel mutually exclusive Softmax (`CrossEntropyLoss`), spatial disjointness vs. pixel-level overlap failure modes. | Evaluating spatial co-occurrence across disjoint regions, identifying catastrophic suppression under annotation collisions. |
| **Question 7** | [Guide 7: Compound Loss Behavior on Defect vs. Clean Images](guide_07_compound_loss_and_clean_image_dynamics.md) | Compound Loss formulation ($0.5 \text{BCE} + 0.5 \text{Dice}$), gradient behavior on completely clean images ($|Y|=0$), false alarm sensitivity, gradient stabilization. | Deriving Dice gradient response to stray noise on clean sheets, justifying compound BCE + Dice balance. |
| **Question 8** | [Guide 8: Scratch Baseline vs. Pretrained Transfer Learning](guide_08_scratch_vs_pretrained_convergence.md) | Empirical fine-tuning dynamics, generic visual feature acquisition, Epoch 1 convergence acceleration, generalization gap $|L_{\text{val}} - L_{\text{train}}|$, competition score gains. | Controlled 5-epoch training protocol, evaluating validation trajectories, measuring score improvements. |
| **Question 9** | [Guide 9: Pretrained Backbone Architecture Trade-off](guide_09_backbone_architecture_tradeoffs.md) | ResNet-34 vs. EfficientNet-B0 parameter ratios, depthwise separable convolutions (MBConv), peak GPU VRAM footprint, edge deployment feasibility. | Comparing model capacity and throughput, analyzing parameter efficiency vs. segmentation fidelity. |

---

## 🛠️ Official Starter Code & Parameter Reference

All models and loss functions throughout Milestone 3 use the standardized definitions below:

```python
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

# 1. Model Definitions (SMP)
model_resnet = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=1,
    classes=4
)

model_effnet = smp.Unet(
    encoder_name="efficientnet-b0",
    encoder_weights="imagenet",
    in_channels=1,
    classes=4
)

# 2. Differentiable Soft Dice & Compound Loss
class SoftDiceLoss(nn.Module):
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
    def __init__(self, alpha=0.5, beta=0.5, smooth=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = SoftDiceLoss(smooth=smooth)
        self.alpha = alpha
        self.beta = beta

    def forward(self, logits, targets):
        return self.alpha * self.bce(logits, targets) + self.beta * self.dice(logits, targets)

# 3. Standard Benchmarking Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-2
EPOCHS = 5
ALPHA = 0.5
BETA = 0.5
SMOOTH = 1.0
```

---

## How to Work Through Milestone 3:

1. **Read the corresponding Concept Guide** before attempting each question to master the underlying vision mechanics and mathematical derivations.
2. **Review the Guiding Hints** to structure your scripts, avoid subtle PyTorch tensor pitfalls, and verify your logic without leaking solutions.
3. **Write and run your own code** in your notebook or local environment against `data/public/` using the standard starter definitions.
4. Enter your verified results into the assessment sheet:
   👉 [`MILESTONE_3.md`](../../MILESTONE_3.md)
