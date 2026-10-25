# RLE Standards: Column-Major Encoding on a 2D Grid
*(Milestone 2 · Question 1 Study Guide)*

---

## 1. The Core Concept: Why Run-Length Encoding?

In automated surface inspection, segmentation masks are mostly background (clean metal), with defect anomalies occupying only a tiny fraction of pixels. Storing uncompressed 2D binary grids for thousands of images consumes excessive storage and network bandwidth.

**Run-Length Encoding (RLE)** solves this by storing only the locations of active pixels as pairs of space-separated integers:

$$\text{"start length start length ..."}$$

- **`start`**: The index where a contiguous sequence of defect pixels begins.
- **`length`**: The number of consecutive defect pixels in that run.

---

## 2. The Two Non-Negotiable Rules

When encoding masks for this competition, your index calculations must adhere strictly to two standards:

### Rule 1: 1-Based Indexing
Unlike Python arrays which begin at index 0, RLE strings are **1-based**:
- The first pixel in memory is assigned index **1**.
- The final pixel on an $H \times W$ canvas is index **$H \times W$**.

### Rule 2: Column-Major Traversal (`order='F'`)
Pixels are flattened into a 1D sequence by traversing **down columns first** (Fortran order), rather than across rows (C order):
1. Walk down Column 0 from top to bottom (row 0 to row $H-1$).
2. Move to Column 1 and walk top to bottom.
3. Continue through all columns until Column $W-1$.

```text
2D Grid Flattening Order (H=4, W=4, 1-based Fortran):
       Col 0    Col 1    Col 2    Col 3
Row 0:   1        5        9       13
Row 1:   2        6       10       14
Row 2:   3        7       11       15
Row 3:   4        8       12       16
```

### The Coordinate Conversion Formula
For any 0-indexed coordinate $(\text{row } r, \text{col } c)$ on a grid of height $H$:

$$k = c \times H + r + 1$$

---

## 3. Guiding Hints for Question 1

To determine the exact RLE string for the $8 \times 8$ grid in Question 1:

- **Hint 1 (Determine Canvas Height):** The grid is of size $8 \times 8$, so $H = 8$. Each column contains 8 pixels. Column 0 spans indices 1–8, Column 1 spans indices 9–16, and so on.
- **Hint 2 (Traverse Column-by-Column):** Inspect the grid column-by-column from left to right. Note the row positions of every highlighted (active) cell.
- **Hint 3 (Compute Linear Indices):** Use the formula $k = c \times 8 + r + 1$ to convert each active cell's $(r, c)$ position into its 1D linear index.
- **Hint 4 (Group Consecutive Integers):** Sort the active indices in ascending order. Any run of consecutive integers $[k, k+1, \dots, k+L-1]$ forms a single run: `k L`.
- **Pitfall to Watch For:** In column-major order, the bottom cell of column $c$ (index $c \times 8 + 8$) and the top cell of column $c+1$ (index $(c+1) \times 8 + 1$) are adjacent integers in 1D memory! If both are active, they merge into a single continuous run.

---

## 4. Your Assignment Task

1. Open [Milestone 2 Assessment: Question 1](milestone-2.html#question-1-rle-standards-column-major-encoding-on-a-2d-grid).
2. Inspect the $8 \times 8$ pixel grid in `assets/fig_q1_rle_grid_8x8.png`.
3. Compute the 1-based Fortran linear indices for all active cells, group them into contiguous `start length` pairs, and enter your final RLE string.
