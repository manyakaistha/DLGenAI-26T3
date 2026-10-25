# Pixel-Level Spatial Overlap & Mutual Exclusivity
*(Milestone 2 · Question 6 Study Guide)*

---

## 1. The Core Concept: Image-Level Co-Occurrence vs. Pixel-Level Overlap

In Questions 4 and 5, you analyzed defect co-occurrence at the **image level**: establishing how frequently distinct defect classes appear anywhere on the same $128 \times 800$ steel strip.

However, sharing an image file does not mean sharing physical space:
- A steel strip could have a longitudinal scratch on the left edge and an oxide patch in the center. They appear in the same image, but their physical footprints are completely disjoint.
- Alternatively, do two defect categories ever physically overlap at the exact same pixel coordinates $(y, x)$?

Answering this empirical question is critical because it dictates the mathematical validity of your neural network's output head.

---

## 2. Formulating Pixel-Level Mask Intersection

For an image containing multiple defects, let $M_i, M_j \in \{0, 1\}^{128 \times 800}$ denote the decoded 2D binary masks for two distinct defect classes $i \neq j$.

### Pairwise Pixel Intersection
The number of shared, overlapping pixels between the two masks is defined as their dot product:

$$|M_i \cap M_j| = \sum_{y=0}^{H-1} \sum_{x=0}^{W-1} \left( M_i(y, x) \cdot M_j(y, x) \right)$$

### Pairwise Dice Similarity Coefficient
The Dice similarity between the two ground-truth masks measures their relative spatial overlap:

$$\operatorname{Dice}(M_i, M_j) = \frac{2 |M_i \cap M_j|}{|M_i| + |M_j|}$$

- If $|M_i \cap M_j| = 0$, then $\operatorname{Dice}(M_i, M_j) = 0.0$ (the masks are strictly disjoint).
- If the two masks share even a single pixel, then $|M_i \cap M_j| > 0$.

---

## 3. The Architectural Consequence: Output Head Design

Why does pixel overlap dictate model architecture?
- **Configuration A (4 Independent Sigmoids):** Each output channel uses an independent Sigmoid activation ($\hat{P}_k = \sigma(z_k)$) evaluated with `BCEWithLogitsLoss`. This formulation is valid whether masks overlap or not, because each pixel can independently belong to multiple classes.
- **Configuration B (5-Class Softmax Partition):** Softmax across channels enforces $\sum_{k=0}^4 \hat{P}_k = 1.0$. This formulation is valid **only if** classes are strictly mutually exclusive ($|M_i \cap M_j| = 0$ everywhere). If two defects share a pixel, forcing a single Softmax distribution creates conflicting target labels and corrupts training gradients.

---

## 4. Guiding Hints for Question 6

To evaluate spatial overlap across the training data:

- **Hint 1 (Focus on Multi-Label Images):** You only need to check images that contain at least two defects (identified in Question 4). Images with zero or one defect cannot have cross-defect pixel overlap.
- **Hint 2 (Decode 2D Binary Masks):** For each multi-label image, decode the RLE masks for its present defect classes into $(128, 800)$ binary arrays.
- **Hint 3 (Compute Pairwise Intersections):** For every pair of active defect classes ($i \neq j$) on that image, compute the intersection $|M_i \cap M_j|$.
- **Hint 4 (Find the Maximum Overlap):** Track the maximum number of overlapping pixels (or max Dice score) observed across any pair in the entire training set.
- **Hint 5 (Assess Mutual Exclusivity):** If the maximum overlap is 0, the defect classes are spatially mutually exclusive at the pixel level (`Yes`). If it is greater than 0, they are not (`No`).

---

## 5. Your Assignment Task

1. Open [Milestone 2 Assessment: Question 6](milestone-2.html#question-6-pixel-level-spatial-overlap-and-mutual-exclusivity).
2. Compute the pairwise pixel intersection across all multi-label images in `train.csv`.
3. Report the maximum number of overlapping pixels (or max Dice) and state whether defect classes are spatially mutually exclusive: `max_overlap_pixels, Yes/No`.
