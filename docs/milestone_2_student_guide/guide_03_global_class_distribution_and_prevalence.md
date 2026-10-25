# Global Class Distribution & Defect Prevalence
*(Milestone 2 · Question 3 Study Guide)*

---

## 1. The Core Concept: Severe Class Imbalance in Manufacturing

In automated visual surface inspection, defect categories rarely occur with equal probability. Modern continuous manufacturing lines are engineered for high yields ($>90\%$), meaning the vast majority of surface area is pristine.

Furthermore, different defect mechanisms have vastly different occurrence probabilities:
- **High-prevalence wear defects:** Surface friction, microscopic roll wear, or oxide scaling occur continuously during mill operations. These defects appear across thousands of production meters.
- **Low-prevalence catastrophic flaws:** Critical weld fissures, severe delamination, or structural tears happen only during severe equipment malfunctions. Consequently, only a tiny handful of images capture these events.

Before designing neural network architectures, weighting loss functions, or partitioning validation folds, you must quantify the exact marginal prevalence of each defect class across the entire training corpus. If a rare defect appears in only a fraction of a percent of images, standard training loss will be dominated by frequent classes, leading to **gradient starvation** for the rare class.

---

## 2. Understanding the Long-Format Annotations

In `train.csv`, annotations are provided in a long-format tabular schema:
- Exactly **11,393 unique images**.
- Exactly **4 rows per image** (`defect_a`, `defect_b`, `defect_c`, `defect_d`).
- Total rows in the table: $11{,}393 \times 4 = 45{,}572$.

### Identifying Defect Presence
In this schema:
- An entry where `mask_rle` is `NaN`, empty (`""`), or whitespace indicates that the defect is **absent** ($y = 0$).
- An entry with a valid, non-empty RLE string indicates that the defect is **present** in that image ($y = 1$).

---

## 3. Guiding Hints for Question 3

To determine the class distribution and identify the extremes:

- **Hint 1 (Filter for Defect Presence):** Focus only on rows in `train.csv` that represent actual defect instances—meaning `mask_rle` is neither null nor empty.
- **Hint 2 (Count Unique Images per Defect Class):** For each of the four categories (`defect_a`, `defect_b`, `defect_c`, `defect_d`), count how many distinct images contain that defect.
- **Hint 3 (Identify Highest and Lowest Prevalence):** Compare the counts across all four categories to determine:
  1. The **most common** defect class and its exact image count.
  2. The **rarest** defect class and its exact image count.
- **Pitfall to Watch For:** Do not simply count the total number of rows per defect class in the raw CSV! Because every image has 4 rows (one per defect class), counting raw rows without filtering for active `mask_rle` will give exactly 11,393 for every single class.

---

## 4. Your Assignment Task

1. Open [Milestone 2 Assessment: Question 3](milestone-2.html#question-3-global-class-distribution-and-defect-prevalence).
2. Load `data/public/train.csv` and filter for active `mask_rle` entries across all four defect classes.
3. Determine the image count for each defect type, and enter the most common and rarest classes: `most_common (count), rarest (count)`.
