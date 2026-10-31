# Output Head Semantics & Loss Formulations
*(Milestone 3 · Question 6 Study Guide)*

---

## 1. The Core Architectural Choice: Sigmoid vs. Softmax

When configuring the final output layer of a semantic segmentation model, deep learning practitioners face a foundational design choice:

```text
               ┌──► Formulation A: 4-Channel Independent Sigmoid (Multi-Label)
Decoder Logits │    Shape: [B, 4, H, W]  │  Trained with BCEWithLogitsLoss
               │
               └──► Formulation B: 5-Channel Mutually Exclusive Softmax (Multi-Class)
                    Shape: [B, 5, H, W]  │  Trained with CrossEntropyLoss
```

This choice governs not only the loss function and activation mathematics, but also the fundamental representational capacity of your inspection system.

---

## 2. Mathematical Principles: Activation & Target Semantics

### Formulation A: 4-Channel Independent Sigmoid
In Formulation A, the network outputs 4 continuous logit feature maps matching the 4 defect classes (`defect_a`, `defect_b`, `defect_c`, `defect_d`):
$$\mathbf{Z} \in \mathbb{R}^{B \times 4 \times H \times W}$$
Each channel is passed through an independent Sigmoid function:
$$P_c(y, x) = \sigma\big(z_c(y, x)\big) = \frac{1}{1 + e^{-z_c(y, x)}} \in [0, 1]$$
- **Target Representation:** Dense binary floating-point tensor of shape `[B, 4, H, W]`.
- **Loss Function:** `nn.BCEWithLogitsLoss()` computed independently across all 4 channels and all pixels.
- **Semantic Freedom:** Channel $c_1$ exerts zero mathematical suppression on channel $c_2$. A pixel $(y, x)$ can simultaneously have high probabilities for multiple defect categories (e.g., $P_a(y, x) = 0.92$ and $P_c(y, x) = 0.88$). If all channels are zero ($P_c < \tau$), the pixel is implicitly classified as background.

### Formulation B: 5-Channel Mutually Exclusive Softmax
In Formulation B, an explicit **Background Class (Channel 0)** is added, creating 5 total output channels:
$$\mathbf{Z} \in \mathbb{R}^{B \times 5 \times H \times W} \quad (\text{0: Background, 1: Defect A, 2: Defect B, 3: Defect C, 4: Defect D})$$
Probabilities are computed via the Softmax operator across the channel dimension at every spatial coordinate $(y, x)$:
$$P_c(y, x) = \frac{e^{z_c(y, x)}}{\sum_{k=0}^4 e^{z_k(y, x)}}$$
- **Target Representation:** 2D integer class index map of shape `[B, H, W]` where each pixel holds an integer class label $Y(y, x) \in \{0, 1, 2, 3, 4\}$.
- **Loss Function:** `nn.CrossEntropyLoss()`.
- **Probability Simplex Constraint:** At every single pixel coordinate $(y, x)$, the channel probabilities strictly sum to 1:
  $$\sum_{c=0}^4 P_c(y, x) = 1.0$$
  By definition, an increase in the probability of one defect class *strictly forces* a mathematical decrease in all other classes.

---

## 3. Spatial Disjointness vs. Pixel-Level Overlap

In Milestone 1 (Question 3), your data audit verified an empirical property of the training dataset:
$$\text{Overlapping Defect Pixels in Training Set} = 0$$
Across all 11,393 training images, whenever multiple defect classes appeared on the same steel sheet, they occurred at **distinct, non-overlapping spatial locations** (e.g., a scratch on the left edge and an oxide patch on the right edge).

### Can Softmax Segment Multiple Defects on the Same Sheet?
A frequent misconception is that Softmax restricts a model to predicting only one defect class per image. **This is false.**
- Softmax is applied **independently at each spatial pixel $(y, x)$**.
- At pixel $(y, x) = (50, 100)$ on the left side of the sheet, the model can predict:
  $$P_{\text{defect\_a}}(50, 100) = 0.95, \quad P_{\text{bg}} = 0.04, \quad P_{\text{others}} \approx 0.01$$
- At pixel $(y, x) = (50, 700)$ on the right side of the sheet, the model can independently predict:
  $$P_{\text{defect\_c}}(50, 700) = 0.94, \quad P_{\text{bg}} = 0.05, \quad P_{\text{others}} \approx 0.01$$
Because spatial coordinates are decoupled, Formulation B can cleanly segment multiple defect instances across disjoint regions of the same steel sheet without mutual interference.

### The Operational Failure Mode of Formulation B
However, Formulation B has a critical operational vulnerability: **it fundamentally cannot represent overlapping annotations at the same spatial pixel**.

```text
The Overlap Breakdown:
Suppose a hairline scratch (Defect A) cuts across a large oxide inclusion (Defect C) at pixel (y, x):
- True physical reality: Both Defect A and Defect C are present simultaneously.
- Integer Class Target: Target tensor must choose ONE integer: y(y, x) = 1 OR y(y, x) = 3.
                        Whichever defect is omitted is treated as FALSE (loss penalizes it!).
- Softmax Probabilities: sum(P_c) = 1.0. If P_A = 0.8, P_C CANNOT exceed 0.2!
```

If test data, re-annotated batches, or real-world production runs contain even a single pixel where two defects overlap, Formulation B experiences **catastrophic mutual suppression**: the model is forced into a winner-take-all competition, guaranteeing missed detections (false negatives) for whichever defect loses the competition at that pixel.

---

## 4. Comparative Decision Matrix

| Dimension | Formulation A (4-Ch Sigmoid) | Formulation B (5-Ch Softmax) |
| :--- | :--- | :--- |
| **Output Activation** | Channel-wise Sigmoid $\sigma(z_c) \in [0, 1]$ | Cross-channel Softmax $\sum P_c = 1.0$ |
| **Loss Function** | `BCEWithLogitsLoss` / Compound Loss | `CrossEntropyLoss` |
| **Target Tensor Format** | Float `[B, 4, 128, 800]` | Long `[B, 128, 800]` (integers 0..4) |
| **Spatially Disjoint Defects** | Fully supported | Fully supported |
| **Overlapping Pixel Annotations** | Fully supported (each channel independent) | **Catastrophic failure** (winner-take-all suppression) |
| **Background Representation** | Implicit (all 4 channels $< \tau$) | Explicit (dedicated Channel 0) |

---

## 5. Guiding Hints for Question 6

To solve Question 6 in [`MILESTONE_3.md`](../../MILESTONE_3.md):

- **Hint 1 (Spatially Separated Defects):** Consider whether Softmax normalization is computed across spatial coordinates or across the channel depth at each individual coordinate $(y, x)$. Since Softmax operates per-pixel, can the left half of an image predict class A while the right half predicts class C? Choose `Yes` or `No`.
- **Hint 2 (Failure Mode Under Overlap):** In 1 concise sentence, identify why Formulation B fails if defects overlap at the exact same pixel $(y, x)$. Note that integer target representation and the Softmax probability simplex ($\sum P_c = 1$) force mutual exclusivity at the pixel level, preventing a single pixel from being classified into multiple categories simultaneously.

---

## 6. Your Assignment Task

1. Open [Milestone 3 Assessment: Question 6](../../MILESTONE_3.md#question-6-output-head-semantics-mutually-exclusive-loss-formulations).
2. Answer whether Formulation B can segment spatially disjoint defects on the same sheet without mutual interference (`Yes` / `No`).
3. Formulate a 1-sentence statement describing the operational failure mode of Formulation B when defect annotations overlap.
4. Enter your response in the specified format: `Yes/No; brief_disadvantage`.
