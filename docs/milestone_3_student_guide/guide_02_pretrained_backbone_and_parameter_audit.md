# Pretrained Backbone Architecture & Parameter Audit
*(Milestone 3 · Question 2 Study Guide)*

---

## 1. The Core Concept: The Transfer Learning Advantage

Training deep semantic segmentation networks from scratch requires tens of thousands of diverse, annotated samples. On specialized industrial datasets like surface anomaly inspection, annotations are scarce, expensive to produce, and heavily skewed toward clean background material.

When a convolutional network is initialized randomly (e.g., via Kaiming/He normal initialization), the earliest convolutional filters possess no prior concept of visual reality. The network must spend dozens of epochs learning basic image processing primitives:
- Directional edge filters (horizontal, vertical, diagonal)
- Color and intensity gradients
- Local frequency and Gabor textures
- Corner and boundary detectors

### Inductive Visual Priors from ImageNet
Transfer learning addresses this cold-start bottleneck by utilizing feature extractors pretrained on large-scale natural image datasets such as **ImageNet-1k** ($1.28\text{M}$ images across 1,000 object categories). Even though steel strips look radically different from dogs, cars, and landscapes, the **low-level physics of light, edge contrast, and spatial continuity are universal across optical sensors**.

By adopting a pretrained encoder backbone, your model starts training with mature, robust feature hierarchies already formed:
- **Stages 1–2 (Shallow features):** Generic edge, ridge, and scratch primitives that transfer immediately to surface cracks.
- **Stages 3–5 (Deep features):** High-level spatial groupings that adapt rapidly to diffuse stains and structural variations.

---

## 2. Theoretical Principles: ResNet vs. EfficientNet Architectures

In modern semantic segmentation toolkits like `segmentation_models_pytorch` (SMP), a standard segmentation model decomposes into three distinct structural modules:

$$\text{Model} = \underbrace{\text{Encoder (Backbone)}}_{\text{Pretrained ImageNet Features}} + \underbrace{\text{Decoder}}_{\text{Feature Upsampling \& Skip Fusion}} + \underbrace{\text{Segmentation Head}}_{1 \times 1 \text{ Conv to } K \text{ Defect Classes}}$$

### Backbone Architecture 1: ResNet-34 (Residual Networks)
ResNet-34 employs standard $3 \times 3$ convolutional layers organized into residual blocks with additive identity shortcut connections:
$$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}$$
The identity shortcut eliminates vanishing gradients across deep stacks, allowing stable optimization. However, standard 2D convolutions apply dense spatial-channel matrix multiplications:
$$\text{Parameters} = K_h \times K_w \times C_{\text{in}} \times C_{\text{out}}$$
For $3 \times 3$ kernels across hundreds of channels, parameter counts scale quadratic with channel depth.

### Backbone Architecture 2: EfficientNet-B0 (Compound Scaling & MBConv)
EfficientNet-B0 was engineered using Neural Architecture Search (NAS) and compound coefficient scaling. Instead of standard convolutions, EfficientNet utilizes **Mobile Inverted Bottleneck Convolutions (MBConv)** built from:
1. **$1 \times 1$ Pointwise Expansion:** Expands channel capacity.
2. **$k \times k$ Depthwise Convolutions:** Applies spatial filtering independently per channel ($C_{\text{in}} \times K_h \times K_w$ parameters, eliminating dense cross-channel multiplying).
3. **Squeeze-and-Excitation (SE) Attention:** Adaptively recalibrates channel-wise feature responses.
4. **$1 \times 1$ Pointwise Projection:** Compresses channels back down to match linear bottlenecks.

This factorization separates spatial filtering from channel mixing, slashing parameter count and FLOPs by **over $70\%$** while preserving representational capacity.

---

## 3. Parameter Accounting & Freezing Dynamics in PyTorch

In deep learning auditing, total parameters represent the storage footprint and parameter capacity of a model. In PyTorch, parameter tensors are queried using `model.parameters()`:

```python
# Total parameters (weights + biases) across all layers:
total_params = sum(p.numel() for p in model.parameters())

# Trainable parameters (requiring gradient computation during backward pass):
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
```

### Feature-Extractor Fine-Tuning (Encoder Freezing)
In the first phase of transfer learning on small datasets, it is common practice to **freeze the encoder backbone**. Because ImageNet features are already well-conditioned, updating encoder weights with aggressive gradients can cause **catastrophic forgetting** (destabilizing generic edge priors).

By setting `requires_grad = False` on the encoder:
```python
for param in model.encoder.parameters():
    param.requires_grad = False
```
- The encoder acts as a fixed, frozen feature extractor.
- Only the randomly initialized decoder blocks and the segmentation head receive gradient updates.
- GPU VRAM consumption during training decreases because intermediate encoder activation gradients do not need to be retained in memory.

### Trainable Parameter Ratio Formula
When the encoder is frozen, the percentage of remaining trainable parameters is:

$$\text{Trainable Percentage} = \frac{\sum_{p \in \text{trainable}} \text{numel}(p)}{\sum_{p \in \text{all}} \text{numel}(p)} \times 100\% = \frac{\text{Decoder Params} + \text{Head Params}}{\text{Total Model Params}} \times 100\%$$

---

## 4. Guiding Hints for Question 2

To solve Question 2 in [`MILESTONE_3.md`](../../MILESTONE_3.md):

- **Hint 1 (Instantiate Models Consistently):** Instantiate both `smp.Unet` models using the exact starter code parameters:
  ```python
  import segmentation_models_pytorch as smp
  model_resnet = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=1, classes=4)
  model_effnet = smp.Unet(encoder_name="efficientnet-b0", encoder_weights="imagenet", in_channels=1, classes=4)
  ```
- **Hint 2 (Compute Total Parameter Counts):** For both `model_resnet` and `model_effnet`, write a helper function that iterates through `model.parameters()` and sums `p.numel()`. Compare the two totals to determine which model is more parameter-efficient, and compute the exact numerical difference:
  $$\Delta_{\text{params}} = |\text{params}_{\text{resnet}} - \text{params}_{\text{effnet}}|$$
- **Hint 3 (Apply Encoder Freezing):** For `model_resnet`, loop over `model_resnet.encoder.parameters()` and set `param.requires_grad = False`. Verify that `model_resnet.decoder` and `model_resnet.segmentation_head` parameters still have `requires_grad == True`.
- **Hint 4 (Calculate Percentage):** Divide the count of parameters with `requires_grad == True` by the total parameter count of `model_resnet`, multiply by 100, and round to 1 decimal place.

---

## 5. Your Assignment Task

1. Open [Milestone 3 Assessment: Question 2](../../MILESTONE_3.md#question-2-pretrained-backbone-architecture-parameter-audit).
2. Instantiate `model_resnet` and `model_effnet` via SMP.
3. Compute the exact total parameter counts, identify the more efficient model, and find their difference.
4. Freeze the ResNet-34 encoder and compute the exact percentage of trainable parameters remaining.
5. Enter your response in the specified format: `more_efficient_model, param_diff; trainable_pct`.
