# RLE Decoding & Multi-Defect Pixel Mass
*(Milestone 2 · Question 2 Study Guide)*

---

## 1. The Core Concept: Inverting RLE to 2D Spatial Masks

While Run-Length Encoding (RLE) is compact for disk storage, deep learning segmentation pipelines require dense 2D binary tensors matching the native image dimensions ($(H, W) = (128, 800)$). Loss functions (such as Binary Cross-Entropy and Dice Loss) compute pixel-wise errors against these dense 2D targets.

To use the annotations in `train.csv`, your data loading pipeline must invert RLE strings back into 2D NumPy arrays on the fly.

---

## 2. Conceptual Steps for Building an RLE Decoder

When writing your decoding function, think through these core operations:

1. **Handling Clean Slices:** In `train.csv`, absent defects are marked as `NaN` or empty strings (`""`). Your function should detect these and immediately return an all-zero array of shape $(128, 800)$.
2. **Parsing Integer Pairs:** Split the string on whitespace to extract alternating pairs:
   $$\text{starts} = \text{tokens}[0::2], \quad \text{lengths} = \text{tokens}[1::2]$$
3. **Accounting for 1-Based Offsets:** Remember from Guide 1 that the start indices in the file are 1-based. To index into a 0-based Python array, subtract 1 from each start value (`start - 1`).
4. **Populating Flat Memory:** Allocate a 1D flat array of zeros of size $H \times W = 102{,}400$. For each run, set the slice from `start - 1` to `start - 1 + length` to 1.
5. **Reshaping to 2D:** Reshape the 1D flat array to shape $(H, W) = (128, 800)$ using **Fortran order (`order='F'`)** so that column-major flattening is correctly inverted back into native coordinates.

---

## 3. Guiding Hints for Question 2

To answer Question 2:

- **Hint 1 (Locate the Image Rows):** Filter `train.csv` for rows where `image_id == "0ZUU8LH57Y"`. You should locate exactly four rows, corresponding to `defect_a`, `defect_b`, `defect_c`, and `defect_d`.
- **Hint 2 (Inspect Mask Presence):** Examine the `mask_rle` field for each of the four defect classes on this image. Note which classes contain non-empty strings and which are `NaN`.
- **Hint 3 (Decode Each Class Separately):** Pass each row's `mask_rle` through your decoding function to obtain a 2D binary mask $M_c \in \{0, 1\}^{128 \times 800}$.
- **Hint 4 (Calculate Total Active Pixel Mass):** The question asks for the total sum of all positive (active) pixel values across all four defect masks. For each mask, count its active pixels ($\sum M_c == 1$) and sum these counts across the four categories:
  $$\text{Total Pixel Mass} = \sum_{c \in \{a, b, c, d\}} \sum_{y=0}^{127} \sum_{x=0}^{799} M_c(y, x)$$

---

## 4. Your Assignment Task

1. Open [Milestone 2 Assessment: Question 2](milestone-2.html#question-2-rle-decoding-in-practice-multi-defect-pixel-mass).
2. Write your RLE decoding function that converts `mask_rle` into a $(128, 800)$ binary mask.
3. Decode all four defect classes for image `"0ZUU8LH57Y"`, sum the active pixels across all four masks, and enter the resulting integer.
