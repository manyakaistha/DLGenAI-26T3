# Milestone 1: Data Understanding and Baseline Segmentation

Industrial Surface Anomaly Segmentation · Revised assessment

Read the [student study guides](docs/milestone_1_student_guide_v2/README.md) before attempting the questions. Use the competition data in `data/public/` and compute your own results. This document contains questions and response formats only.

For Questions 11 and 12, use the [Colab starter notebook](https://colab.research.google.com/drive/1Jf4f7jKddRSMpiZ1D8oVqmR8LDK-t86Q?usp=sharing) and follow the [setup instructions](docs/starter_code_q11_q12/README.md).

Questions 1–11 use short answers. Question 12 requires a descriptive answer in bullet points. Image dimensions below are height × width unless stated otherwise.

## Question 1: Global Class Distribution & Clean Sample Mass

In `data/public/train.csv`, each of the 11,393 unique training images has 4 rows corresponding to the 4 defect classes (`defect_a`, `defect_b`, `defect_c`, `defect_d`), totaling 45,572 rows. A row with empty/null `mask_rle` indicates the absence of that defect.

Compute the frequency distribution of positive occurrences for each defect class, as well as the total number of **completely clean images** (images with no defect across all 4 classes).

**Question:** Based on your calculations, what is the exact sum of the count of positive samples in the **most frequent defect class** and the total count of **completely clean images**?

**Answer format:** Integer.

## Question 2: Multi-Label Cardinality & Co-occurrence Complexity

While defect classes are reported across separate rows, industrial steel sheets can suffer from simultaneous manufacturing errors. Analyze the label cardinality (the number of distinct defect classes present per image) across all 11,393 images in `train.csv`.

**Question:** Exactly how many images in the training set exhibit multi-label defect co-occurrence (i.e., contain **strictly more than 1 defect class**)?

**Answer format:** Integer.

## Question 3: The Spatial Mutual-Exclusivity Proof (Architectural Insight)

Even if an image has multiple defects at the image level, the physical defects may or may not overlap spatially. Decode all masks for the multi-label images identified in Question 2 onto their native `(128, 800)` canvas.

Check whether any individual pixel carries annotations for two or more defect classes at the exact same spatial coordinate $(y, x)$.

**Question:** What is the total count of overlapping defect pixels (pixels with $\ge 2$ defect labels) across the entire training dataset?

**Answer format:** Short answer (exact count).

## Question 4: RLE Standards & The Boundary Edge Case (Production Data Cleaning)

The competition uses Kaggle/Severstal-style 1-based Fortran-order (column-major) Run-Length Encoding on a canvas of shape $(128, 800) = 102{,}400$ pixels. A naive decoder assumes every run index strictly satisfies:

$$1 \le \text{start} + \text{length} - 1 \le 102{,}400.$$

Inspect every non-empty RLE string in `train.csv`.

**Question:** Exactly how many rows in `train.csv` contain a malformed run where the end pixel index exceeds the canvas boundary (102,400), and what is the maximum pixel index encountered?

**Answer format:** `X rows, max index Y`.

## Question 5: Geometric Distortion & Anisotropic Aspect Ratio

Images have native resolution $128 \times 800$ (height × width). A standard deep learning practice in natural image segmentation is to resize inputs to a square $256 \times 256$ using bilinear interpolation, train the model, and resize predicted masks back to $128 \times 800$ via nearest neighbors. The forward resize applies its own scale factor to each axis: the horizontal scale factor $s_x$ acts on width, and the vertical scale factor $s_y$ acts on height.

**Question:** Compute $s_x$ and $s_y$ for the $128 \times 800 \rightarrow 256 \times 256$ resize, then the ratio of vertical stretching to horizontal compression, $s_y/s_x$. (Mind which dimension — width or height — belongs in each ratio.)

**Answer format:** `s_x, s_y, ratio`.

## Question 6: Photometric Invisibility & Defect Feature Types (Cohen's d)

For each defect class, calculate the pixel intensity distribution of the defect region versus the surrounding local background (outside defect, within the same image). The standardized mean difference is measured by Cohen's $d$:

$$d = \frac{\mu_{\text{defect}} - \mu_{\text{bg}}}{\sigma_{\text{pooled}}}.$$

**Question:** Which defect class has a Cohen's $d \le 0.02$ relative to its local background, rendering it photometrically invisible to pure grayscale thresholding methods, and what type of features (intensity vs. texture/frequency) are strictly required to detect it?

**Answer format:** Defect name and feature type: `defect_name, feature_type`.

## Question 7: Metric Behavior — The Discontinuous Step Function of Naive Dice

In standard semantic segmentation, Dice between predicted binary mask $X$ and ground truth mask $Y$ is defined as:

$$\operatorname{Dice}(X,Y) = \frac{2|X \cap Y|}{|X| + |Y|}.$$

Use the standard empty convention: $\operatorname{Dice}(\emptyset,\emptyset) = 1.0$.

Suppose an image has no defect for `defect_b` ($Y = \emptyset$).

- Model 1 predicts no defect ($\hat{X}_1 = \emptyset$).
- Model 2 predicts just 2 stray noise pixels ($|\hat{X}_2| = 2$).

**Question:** What is the Dice score for Model 1? What is the Dice score for Model 2?

**Answer format:** Numerical scores: `Dice1, Dice2`.

## Question 8: Official Competition Weighted Metric — Toy Batch Calculation

The competition organizers evaluate submissions using a weighted composite metric. For this assessment, use these component definitions:

$$\text{Score} = 0.70 \times \text{ForegroundDice} + 0.20 \times \text{FalsePositiveScore} + 0.10 \times \text{EmptyAccuracy}.$$

1. **ForegroundDice** is the average Dice score computed strictly across image-defect pairs where a ground-truth defect exists ($|Y| > 0$).
2. **FalsePositiveScore** is $1 - \text{mean fp pixel fraction}$ across all ground-truth empty pairs ($|Y| = 0$), where the false-positive pixel fraction is $|\hat{X}|/102{,}400$.
3. **EmptyAccuracy** is the proportion of ground-truth empty pairs ($|Y| = 0$) where the model correctly predicts an entirely empty mask ($|\hat{X}| = 0$).

Consider a validation set of **50 image-class pairs**:

- **10 pairs are positive** ($|Y| > 0$). The model achieves individual Dice scores of 0.80 on 6 pairs and 0.40 on 4 pairs.
- **40 pairs are empty** ($|Y| = 0$). The model predicts completely empty masks on 36 pairs. On the remaining 4 pairs, it predicts false-positive blobs of 1,024 pixels each on a 102,400-pixel canvas.

**Question:** Calculate the exact values of:

(a) ForegroundDice  
(b) FalsePositiveScore  
(c) EmptyAccuracy  
(d) Final weighted Score (rounded to 4 decimal places).

**Answer format:** `a, b, c, d`.

## Question 9: The “All-Empty Baseline” on the Full Training Set (No Splits Required)

You have only the raw Kaggle training data; no validation folds or trained models are needed for this question.

Consider the trivial **“Always Abstain / All-Zeros”** baseline: a predictor that outputs an empty mask $\hat{X} = \emptyset$ for every image-class pair. Evaluate this baseline over **all 45,572 image-class pairs** in `train.csv`, using the empty convention from Question 7 ($\operatorname{Dice}(\emptyset,\emptyset) = 1.0$).

**Question:**

1. How many of the 45,572 image-class pairs have an empty ground-truth mask? Using that count, what is the naive empty-aware mean Dice score of the all-empty baseline (round to 4 decimal places)?
2. What is the official competition score of this exact same baseline (using the three component definitions from Question 8)?

**Answer format:** `Score_naive, Score_official`. Compute the empty-pair count as part of your working.

## Question 10: Morphological Noise Filtering — Minimum Area Recall Ceilings

Model predictions often contain small spurious speckles. A standard post-processing step is to apply 8-connected component analysis (`cv2.connectedComponentsWithStats`) and discard all components with pixel area $< N$. However, this introduces a recall ceiling if ground-truth defects are smaller than $N$.

Inspect the connected components of all ground-truth masks in `train.csv`.

**Question:**

1. What is the smallest connected component instance area (in pixels) present in the ground truth for `defect_b`?
2. If a student applies a filter with $N = 50$ pixels, what percentage of true ground-truth pixels is discarded for `defect_b`?
3. If the filter is increased to $N = 200$ pixels, what percentage of ground-truth pixels are lost for `defect_a`?

**Answer format:** `Area_min_b, Lost_pct_b_50, Lost_pct_a_200`.

## Question 11: Multi-Label Combination Stratification

Build **one simple 80/20 train/validation holdout** over the 11,393 training images. Defect prevalence is highly skewed: `defect_b` has only 313 positive images (2.75% prevalence), while `defect_c` has 4,100. To protect rare classes and their co-occurrences, stratify on each image's multi-label combination string (e.g., `clean`, `defect_c`, `defect_c+defect_d`, …) instead of splitting at random.

**Question:**

1. How many unique combination strings exist across the 11,393 training images?
2. If every combination stratum is split in exact 80/20 proportions (fractional counts allowed), how many `defect_b`-positive images fall into the validation holdout? Round to the nearest integer.

**Answer format:** `n_combos, b_in_val`.

## Question 12: Baseline Model Implementation & Decision Threshold Sweep

**This question needs a descriptive answer. Use bullet points.**

Train a baseline segmentation model (e.g., ResNet-34 U-Net or EfficientNet-B0 at native $128 \times 800$ resolution) on the 80% training partition and evaluate on the 20% stratified holdout from Question 11 for **5–10 epochs**. On the holdout, evaluate raw output sigmoid probabilities across a grid of decision thresholds $\tau \in \{0.3, 0.5, 0.7, 0.9\}$ **without minimum-area filtering**.

**Question:**

1. As threshold $\tau$ increases from 0.3 to 0.9, what happens to the false-positive rate on empty pairs versus the Foreground Dice on positive pairs?
2. Which metric — naive mean Dice or the official competition score — tends to prefer the higher thresholds, and why? Connect your answer to the holdout emptiness you would measure using the counting method from Question 9.

**Answer format:** Bullet points describing the trends and analyzing the peak $\tau$ for each metric.
