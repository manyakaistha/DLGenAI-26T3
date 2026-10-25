# PyTorch Dataset Architecture & Loss Typing (Paradigms 1 & 2)
*(Milestone 2 · Question 7 Study Guide)*

---

## 1. Architectural Paradigms for Multi-Defect Segmentation

When designing a segmentation system for multi-label industrial surface defects, there are two fundamentally different formulations:

### Paradigm 1: Multi-Channel Joint Prediction (Concurrent Multi-Task)
A single neural network ingests the raw strip image and concurrently outputs segmentation predictions for all four defect categories in a single forward pass:

$$f_\theta : \mathbf{X} \in \mathbb{R}^{3 \times H \times W} \longrightarrow \mathbf{\hat{Y}} \in \mathbb{R}^{4 \times H \times W}$$

- **Advantage (High Throughput):** Requires only one forward pass per image to evaluate all four defect classes, making it fast for high-speed industrial line-scan inspection.
- **Trade-off (Gradient Competition):** Gradients from all four heads backpropagate through the shared backbone simultaneously; dominant, high-prevalence classes can overpower gradients from rare classes.

### Paradigm 2: Conditioned / Prompted Segmentation (Query-Based)
Inspired by prompt-conditioned architectures such as **OpenAI's Whisper** (which uses prefix prompt tokens like `<|en|>` to dynamically steer task execution), segmentation is reframed as an interactive query: *"Here is a steel strip image; segment only defect class $k$."*

$$f_\theta : (\mathbf{X} \in \mathbb{R}^{3 \times H \times W}, c \in \{\text{defect_a}, \dots\}) \longrightarrow \mathbf{\hat{Y}} \in \mathbb{R}^{1 \times H \times W}$$

- **Advantage (Class-Directed Prediction):** The condition tells the model which defect to segment, and training batches can be balanced across class queries. A conditioned model normally shares learned parameters across queries; conditioning alone does not allocate a dedicated set of weights to each class.
- **Trade-off (Inference Cost):** Producing all four classes requires four class-conditioned outputs. A simple implementation may run the entire model once per class, but other implementations can cache shared image features or batch class queries. The cost depends on where conditioning occurs; it is not necessarily four complete image-encoder passes.

---

## 2. The PyTorch Dataset Contract for Paradigm 1

To train under **Paradigm 1**, your custom PyTorch `Dataset` subclass must implement:
- `__len__`: Returns the number of unique image IDs in **this Dataset partition**. The full training corpus contains $11{,}393$ images, but separate training and validation Datasets report their own subset sizes, not the full-corpus count.
- `__getitem__(idx)`: Loads the $idx$-th image from disk, normalizes pixel values to $[0.0, 1.0]$ in shape `(3, 128, 800)`, decodes the four defect RLE masks into shape `(4, 128, 800)` with `torch.float32` dtype, and returns the tuple `(image, masks)`.

When wrapped inside a PyTorch `DataLoader` with `batch_size = 4`, individual samples are collated along a new batch dimension, yielding 4-dimensional tensors of the form $[B, C, H, W]$.

---

## 3. Why Use Floating-Point Target Masks for BCE?

A common question in PyTorch segmentation is:
> *"Since defect masks contain only binary values (0 for background and 1 for defect), why can't we store target masks as `torch.uint8` or `torch.bool`?"*

The answer lies in the mathematical formulation and PyTorch kernel implementation of `nn.BCEWithLogitsLoss()`:

1. **Loss Formulation:** For predicted logit $z \in \mathbb{R}$ and ground-truth target $y \in \{0.0, 1.0\}$, the loss is:
   $$\mathcal{L}(z, y) = \max(z, 0) - z \cdot y + \log\left(1 + e^{-|z|}\right)$$
2. **Analytical Gradient:** During backpropagation, the gradient of the loss with respect to logit $z$ is:
   $$\frac{\partial \mathcal{L}}{\partial z} = \sigma(z) - y$$
3. **Target Representation and PyTorch Operations:** `BCEWithLogitsLoss` expects floating-point targets with the same shape as the logits and values in $[0, 1]$. Binary labels are therefore represented as $0.0$ and $1.0$. Passing integer or boolean mask tensors is unsuitable for this loss's floating-point operations and can raise dtype/casting errors. Convert the masks before evaluating the loss.

For this assignment, use `torch.float32` masks as specified in Q7. A simple full-precision training loop uses float32 images, logits and masks. **Float32 is a practical contract here, not the only floating-point type PyTorch supports:** float64 loss calculations and supported mixed-precision combinations are also valid. The target and logit dtypes do not universally have to match exactly.

The gradient formula above explains how predictions are adjusted toward the labels; it does not prove a requirement for identical dtypes. Targets normally do not require gradients. Feed raw logits into `BCEWithLogitsLoss` rather than applying sigmoid first; its fused formulation handles the sigmoid and binary cross-entropy stably.

---

## 4. Guiding Hints for Question 7

To answer Question 7:

- **Hint 1 (Identify 4D Batch Shapes):** Given a batch size of 4, 3 color channels, 4 defect classes, height 128, and width 800, write out the exact shapes $[B, C, H, W]$ for both `images` and `masks`.
- **Hint 2 (Verify Channel Order):** In Paradigm 1, mask channels strictly map to: Channel 0 $\to$ `defect_a`, Channel 1 $\to$ `defect_b`, Channel 2 $\to$ `defect_c`, Channel 3 $\to$ `defect_d`.
- **Hint 3 (Formulate the Loss Explanation):** Explain why BCE uses floating-point target values, state the float32 mask contract for this assignment, and distinguish that choice from a universal requirement that every target and logit dtype match exactly.
- **Hint 4 (Explore Paradigm 2 Differences):** For the optional question, consider how the dataset item would change if each sample yielded a single-channel mask and a condition prompt token rather than a 4-channel tensor.

---

## 5. Your Assignment Task

1. Open [Milestone 2 Assessment: Question 7](milestone-2.html#question-7-pytorch-dataset-for-paradigm-1).
2. Implement your PyTorch `SurfaceAnomalyDataset` and DataLoader under Paradigm 1.
3. Extract the first batch and report `images_shape, masks_shape, dtype, brief_loss_explanation`.


Reference: PyTorch’s [Dataset and DataLoader documentation](https://docs.pytorch.org/docs/2.8/data.html) and [`BCEWithLogitsLoss` documentation](https://docs.pytorch.org/docs/2.8/generated/torch.nn.BCEWithLogitsLoss.html) describe these interface and loss contracts.
