# Feature Map Spatial Geometry & Downsampling Arithmetic
*(Milestone 3 · Question 1 Study Guide)*

---

## 1. The Engineering Context: Anisotropic Sensor Ribbons

In standard computer vision benchmarks (e.g., ImageNet, COCO, Cityscapes), images are typically square or near-square (e.g., $224 \times 224$, $512 \times 512$). In industrial quality control, however, steel strips pass under continuous line-scan cameras at speeds exceeding 15 m/s. The resulting digital inspection canvases are wide, elongated ribbons with an anisotropic aspect ratio:

$$(H, W) = (128, 800) \quad \implies \quad \text{Aspect Ratio} = \frac{W}{H} = \frac{800}{128} = 6.25 : 1$$

In Milestone 2, you learned that naively resizing these images to square dimensions ($256 \times 256$) stretches vertical features by $2\times$ while squishing horizontal features by over $3.1\times$, severely distorting thin scratches and longitudinal cracks. Therefore, industrial segmentation networks must process strips at their **native anisotropic dimensions**.

To design an effective encoder–decoder architecture, you must understand how spatial dimensions transform through progressive downsampling stages, how bottleneck representations compress spatial information, and how odd-dimension rounding introduces fatal skip-connection tensor alignment bugs.

---

## 2. Theoretical Principles: Progressive Downsampling & Bottleneck Compression

A semantic segmentation encoder progressively compresses 2D spatial dimensions while expanding channel capacity to capture hierarchical visual abstractions:
1. **Early layers** operate at high spatial resolution and small receptive fields, capturing high-frequency primitives (microscopic edges, pixel gradients, localized textures).
2. **Deep layers (Bottleneck)** operate at low spatial resolution and massive receptive fields, capturing semantic context (global sheet morphology, large diffuse stains).

```text
Progressive 4-Stage Downsampling Pipeline:
Input:    (B, C_in,  128, 800)
             │  stride=2
             ▼
Stage 1:  (B, C_1,    64, 400)
             │  stride=2
             ▼
Stage 2:  (B, C_2,    32, 200)
             │  stride=2
             ▼
Stage 3:  (B, C_3,    16, 100)
             │  stride=2
             ▼
Stage 4:  (B, C_4,    H_bottle, W_bottle)   <--- Bottleneck Representation
```

### Stride-2 Downsampling Arithmetic
At each stage $s \in \{1, 2, 3, 4\}$, spatial dimensions are halved using either:
- Strided convolutions (`Conv2d(stride=2, padding=1, kernel_size=3)`)
- Max pooling (`MaxPool2d(kernel_size=2, stride=2)`)

For an input dimension $D_{\text{in}}$, standard PyTorch pooling applies integer floor division:
$$D_{\text{out}} = \left\lfloor \frac{D_{\text{in}} + 2 \times p - k}{s} \right\rfloor + 1$$
For standard non-overlapping $2 \times 2$ pooling ($k=2, s=2, p=0$):
$$D_{\text{out}} = \left\lfloor \frac{D_{\text{in}}}{2} \right\rfloor$$

### Spatial Area Downsampling Factor
The spatial area of a 2D feature map is given by $A = H \times W$. When an image undergoes $N$ successive stride-2 downsampling operations:
- The height is reduced by $2^N$.
- The width is reduced by $2^N$.
- The total spatial footprint area is reduced by:
$$\text{Area Downsampling Factor} = \frac{A_{\text{input}}}{A_{\text{bottleneck}}} = \frac{H_{\text{input}} \times W_{\text{input}}}{H_{\text{bottle}} \times W_{\text{bottle}}} = (2^N) \times (2^N) = 2^{2N} = 4^N$$

For a 4-stage encoder ($N = 4$):
$$\text{Area Factor} = 2^4 \times 2^4 = 16 \times 16 = 256$$
This means that at the Stage 4 bottleneck, each spatial location in the feature tensor summarizes an area equivalent to **256 native canvas pixels**.

---

## 3. The Odd-Dimension Parity Trap: Skip-Connection Tensor Mismatches

In U-Net architectures, the decoder reconstructs fine spatial boundaries by concatenating high-resolution encoder feature maps with upsampled decoder feature maps via **lateral skip connections**:

$$\mathbf{X}_{\text{fused}} = \text{Concat}\big(\mathbf{X}_{\text{skip}}, \, \text{Upsample}(\mathbf{X}_{\text{decoder}})\big)$$

In PyTorch, `torch.cat([tensor_a, tensor_b], dim=1)` requires that all non-concatenated dimensions match **exactly**:
$$H_a == H_b \quad \text{and} \quad W_a == W_b$$
If there is even a single-pixel mismatch, PyTorch raises a runtime exception:
`RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 100 but got size 99 for dimension 3.`

### Why Odd Dimensions Cause Phase Errors
Consider what happens when an input dimension is **odd** (e.g., $W = 799$):
- Stage 0 (Input): $W_0 = 799$
- Stage 1: $W_1 = \lfloor 799 / 2 \rfloor = 399$ (1 pixel dropped by floor division)
- Stage 2: $W_2 = \lfloor 399 / 2 \rfloor = 199$
- Stage 3: $W_3 = \lfloor 199 / 2 \rfloor = 99$
- Stage 4 (Bottleneck): $W_4 = \lfloor 99 / 2 \rfloor = 49$

Now trace what happens during decoder upsampling using standard $2\times$ bilinear interpolation or transposed convolution (`ConvTranspose2d(stride=2)`):
- From Stage 4 to Stage 3: $49 \times 2 = 98$. But Stage 3 encoder skip feature map has width $W_3 = 99$!
- **Mismatch at Stage 3 skip connection:** $99 - 98 = 1\text{ pixel}$!
- Propagating all the way back up to the canvas: $49 \times 2^4 = 49 \times 16 = 784 \neq 799$.

This fundamental parity mismatch occurs whenever an image dimension is not divisible by $2^N$ (where $N$ is the number of downsampling stages). For $N=4$, input dimensions must be exact multiples of $2^4 = 16$. Because $800$ is cleanly divisible by 16 ($800 / 16 = 50$), native ribbons process without mismatch. But an unpadded $799$-pixel strip causes immediate tensor misalignment.

---

## 4. Guiding Hints for Question 1

To solve Question 1 in [`MILESTONE_3.md`](../../MILESTONE_3.md):

- **Hint 1 (Trace Bottleneck Shape):** Take the native canvas dimensions $(H, W) = (128, 800)$. Apply integer division by 2 successively across the 4 stages:
  $$\text{Stage 1: } (128/2, 800/2) \to \text{Stage 2: } (\dots) \to \text{Stage 3: } (\dots) \to \text{Stage 4: } (H_{\text{bottle}}, W_{\text{bottle}})$$
  Verify that both $H_{\text{bottle}}$ and $W_{\text{bottle}}$ are clean integers.
- **Hint 2 (Calculate Odd-Width Truncation):** For an odd width of $W = 799$, trace the width through the 4 stages using integer floor division $\lfloor W / 2 \rfloor$. At Stage 4, multiply the bottleneck width by $2^4 = 16$ to find the reconstructed width $\hat{W}_{\text{recon}}$. The dimension mismatch along width is:
  $$\Delta W = W_{\text{native}} - \hat{W}_{\text{recon}} = 799 - (W_{\text{bottle}} \times 16)$$
- **Hint 3 (Compute Downsampling Factor):** Compute the ratio of total input pixels to total bottleneck feature locations:
  $$\text{Factor} = \frac{128 \times 800}{H_{\text{bottle}} \times W_{\text{bottle}}}$$
  Confirm that this matches the mathematical identity $(2^4)^2 = 16 \times 16$.

---

## 5. Your Assignment Task

1. Open [Milestone 3 Assessment: Question 1](../../MILESTONE_3.md#question-1-feature-map-spatial-geometry-downsampling-arithmetic).
2. Trace the exact Stage 4 bottleneck shape $(H_{\text{bottle}}, W_{\text{bottle}})$.
3. Compute the spatial mismatch (in pixels along width) for an odd-width strip of $128 \times 799$.
4. Calculate the total spatial downsampling factor.
5. Enter your response in the specified format: `H_bottle, W_bottle; mismatch_px; downsample_factor`.
