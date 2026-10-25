# Multi-Label Defect Co-Occurrence Matrix
*(Milestone 2 · Question 4 Study Guide)*

---

## 1. The Core Concept: Multi-Label Co-Occurrence

In standard single-label image classification (such as ImageNet), each image belongs to exactly one category: an image depicts either a cat or a dog, but never both.

In industrial quality control, this mutual exclusivity assumption breaks down. Continuous manufacturing processes expose materials to thermal, chemical, and mechanical forces simultaneously. A single physical steel strip can easily exhibit multiple defect types at the same time:
- A mechanical gouge from roller wear, and
- Surface oxide discoloration from localized cooling failure.

When multiple distinct defect categories appear together on the same physical image, this phenomenon is called **multi-label co-occurrence**. Analyzing co-occurrence reveals whether defect types occur in clusters or independently.

---

## 2. Formulating the $4 \times 4$ Co-Occurrence Matrix

To quantify joint failure patterns across all $N = 11{,}393$ training images, we construct a $4 \times 4$ defect co-occurrence matrix $C$.

### Mathematical Representation
For each image $k$, defect presence across the four classes (`defect_a`, `defect_b`, `defect_c`, `defect_d`) can be represented as a binary indicator vector:

$$\mathbf{y}_k = [y_{k, 1}, \; y_{k, 2}, \; y_{k, 3}, \; y_{k, 4}]^T \in \{0, 1\}^4$$

where $y_{k, i} = 1$ if defect $i$ is present on image $k$, and $0$ otherwise.

Stacking these vectors into an $N \times 4$ binary indicator matrix $\mathbf{Y}$, the $4 \times 4$ co-occurrence matrix $C$ is defined as:

$$C = \mathbf{Y}^T \mathbf{Y} \in \mathbb{R}^{4 \times 4}$$

### Key Matrix Properties
1. **Diagonal Entries ($C_{ii}$):** $C_{ii} = \sum_{k=1}^N y_{k, i}$ represents the **marginal count** of images containing defect $i$ (the values computed in Question 3).
2. **Off-Diagonal Entries ($C_{ij}$ where $i \neq j$):** $C_{ij} = \sum_{k=1}^N y_{k, i} y_{k, j}$ represents the exact count of images where **both defect $i$ and defect $j$ appear simultaneously**.
3. **Symmetry:** Because joint presence is commutative ($y_i y_j = y_j y_i$), the matrix is strictly symmetric: $C_{ij} = C_{ji}$.

---

## 3. Guiding Hints for Question 4

To compute your matrix and isolate the top co-occurring pair:

- **Hint 1 (Form Binary Presence Indicators):** For every unique image, determine which of the four defect categories are present ($1$) and which are absent ($0$).
- **Hint 2 (Count Pairwise Intersections):** For any two defect classes $i$ and $j$, count the number of images where both classes have indicator value $1$.
- **Hint 3 (Focus Strictly on Off-Diagonal Pairs):** Question 4 asks for distinct defect classes ($i \neq j$). Inspect only the off-diagonal entries of your matrix to find the pair with the highest co-occurrence count.
- **Pitfall to Watch For:** Do not confuse the diagonal entries (single-defect counts, which can be very large) with the off-diagonal joint co-occurrence counts!

---

## 4. Your Assignment Task

1. Open [Milestone 2 Assessment: Question 4](milestone-2.html#question-4-multi-label-defect-co-occurrence-matrix).
2. Compute the $4 \times 4$ defect co-occurrence matrix across all 11,393 training images.
3. Identify which distinct pair of defect classes ($i \neq j$) co-occurs the most and state their exact joint image count: `defect_pair, count`.
