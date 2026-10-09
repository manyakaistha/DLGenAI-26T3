# The Competition Scorer and The "All-Empty Baseline Paradox"
*(Preparing for Milestone 1: Questions 8 & 9)*

---

## 1. The Official Competition Weighted Metric
Because naive Dice creates a brutal step penalty on negative images (Guide 6) while inflating scores on datasets with many clean images, industrial anomaly competitions (e.g., Kaggle Industrial Surface Defect Detection) implement a **weighted composite evaluation metric**:

$$\mathbf{\text{Score} = 0.70 \times \text{ForegroundDice} + 0.20 \times \text{FalsePositiveScore} + 0.10 \times \text{EmptyAccuracy}}$$

*Official Source:* [Competition Evaluation Overview](https://www.kaggle.com/competitions/industrial-surface-defect-detection/overview/evaluation).

**Design philosophy in one sentence:** separate the metric into "how well do you find real defects?" (70%) and "how well do you avoid hallucinating?" (30%), so that neither behavior can masquerade as the other.

### The Three Components Explained:

1. **Foreground Dice ($70\%$ of Leaderboard Weight):**
   * Computed **only over positive pairs** ($|Y| > 0$) — images where a defect actually exists.
   * Measures defect localization quality when a defect is actually present. Empty images do not artificially inflate this term, because they are excluded from this average entirely!
   * This is the "main event": finding defects is what the task is about, hence 70%.

2. **False Positive Score ($20\%$ of Leaderboard Weight):**
   * Computed **only over empty pairs** ($|Y| = 0$).
   * Penalizes the total volume of hallucinated pixels:
     $$\text{FalsePositiveScore} = 1 - \frac{1}{N_{\text{empty}}} \sum_{j \in \mathcal{D}_{\text{empty}}} \frac{|X_j|}{\Omega_{\text{canvas}}}$$
   * **Reading the formula:** for each empty image, $\frac{|X_j|}{\Omega_{\text{canvas}}}$ is the fraction of the canvas the model wrongly marked as defect (canvas size $\Omega = 128 \times 800 = 102{,}400$ px). The average of these fractions is subtracted from 1, so a model predicting nothing on empty images scores 1.0, and a model that paints the whole canvas scores 0.
   * **Why smoother than naive Dice?** Instead of dropping from 1.0 to 0.0 for a single noise pixel (the Guide 6 cliff), the score decreases *proportionally* to the false-alarm volume. One speckle costs almost nothing; flooding the canvas costs everything.

3. **Empty Prediction Accuracy ($10\%$ of Leaderboard Weight):**
   * Computed **only over empty pairs** ($|Y| = 0$).
   * Measures discrete abstention: what fraction of empty images were predicted completely empty ($|X| = 0$)?
   * This is an all-or-nothing check *on top of* the smooth False Positive Score: even if your false-alarm volume is tiny, you get extra credit only for perfectly clean abstentions.

### How the pieces fit: a worked micro-example
Suppose a validation set has 10 pairs: 6 empty, 4 with defects. A model predicts:
* Perfect overlap on all 4 defect images → $\text{FG} = 1.0$.
* On empty images: predicts 0 pixels on 5 of them, and 102 pixels ($0.1\%$ of canvas) on 1.
  * $\text{FP} = 1 - \frac{0+0+0+0+0+0.001}{6} \approx 1 - 0.000167 \approx 0.99983$
  * $\text{Empty} = 5/6 \approx 0.8333$
* $\text{Score} = 0.70(1.0) + 0.20(0.99983) + 0.10(0.8333) \approx 0.70 + 0.19997 + 0.08333 \approx 0.9833$

Work through this yourself with different numbers until the arithmetic feels automatic — Practice Problem 7.1 gives you exactly that, and Question 8 gives you a larger graded toy batch.

---

## 2. The "All-Empty Baseline Paradox"
In classification, everyone knows the "Majority Class Trap": if $90\%$ of patients are healthy, a doctor who diagnoses everyone as healthy gets $90\%$ accuracy.

A very similar trap exists in segmentation under naive empty-aware Dice. The paradox: **a model that does literally nothing can look excellent under naive mean Dice, yet score poorly under the official metric.** These two numbers can be worlds apart for the exact same predictions.

### Derivation for an All-Empty Model ($\hat{X} = \emptyset$):
Suppose an evaluation pool contains $N$ total pairs, of which $N_{\text{pos}}$ are positive and $N_{\text{empty}}$ are empty (the pool can be a validation split — or simply the entire training set; this effect needs no cross-validation):
1. **Under Naive Empty-Aware Dice:**
   * Every positive pair scores $0.0$ (model missed the defect — case 2 of the convention: $|X| = 0$, $|Y| > 0$).
   * Every empty pair scores $1.0$ (model correctly predicted nothing — case 1 of the convention).
   $$\text{Naive Mean Dice} = \frac{N_{\text{empty}} \times 1.0 + N_{\text{pos}} \times 0.0}{N} = \frac{N_{\text{empty}}}{N}$$
   A completely blind model that predicts nothing scores exactly the *emptiness rate* of the pool it is evaluated on. Whether that number looks flattering depends entirely on how empty that pool is — a value you can count directly from the raw labels in `train.csv`.

2. **Under the Official Competition Scorer:**
   * $\text{ForegroundDice} = 0.0$ — no defects detected, so the 70% term contributes nothing at all.
   * $\text{FalsePositiveScore} = 1.0$ — zero false-positive pixels on empty images, perfect marks.
   * $\text{EmptyAccuracy} = 1.0$ — every empty pair predicted empty, perfect marks.
   The baseline therefore receives **zero** on the dominant term and **full credit** on both abstention terms; its score is a small constant fixed by the metric's weights alone — it does not depend on how empty the dataset is, and it does not improve if you feed the model more data. (Evaluate this expression numerically for the official weights as part of your Question 9 task; Practice Problem 7.2 rehearses the same evaluation with different weights first.)

The official scorer cuts through the illusion of the naive number: a model that does not detect defects cannot score well on it, no matter how much of the dataset is empty.

> **The lesson:** never trust a single aggregate metric in isolation. Always ask: *"what would a trivial all-empty baseline score under this metric?"* If your real model barely beats the all-empty baseline, your model is not actually learning defect detection.

---

## 3. Official Trusted Resources & Recommended Reading
* **Kaggle Competition Rules & Metric:**
  [Industrial Surface Defect Detection Evaluation](https://www.kaggle.com/competitions/industrial-surface-defect-detection/overview/evaluation).
* **Maier-Hein, L., et al. (2024).** *Metrics reloaded: recommendations for image analysis validation*. Nature Methods, 21(2), 195–212. [DOI: 10.1038/s41592-023-02151-z](https://doi.org/10.1038/s41592-023-02151-z).
* **Reinke, A., et al. (2024).** *Understanding metric-related pitfalls in image analysis validation*. Nature Methods, 21(2), 182–194. [DOI: 10.1038/s41592-023-02150-0](https://doi.org/10.1038/s41592-023-02150-0).

---

## 4. Practice Problems: Score Toy Batches Yourself

### Practice Problem 7.1 — Score a 20-pair toy batch under the official formula
A toy validation set has **20 image-class pairs**:
* **8 positive pairs** with individual Dice scores: five pairs at $0.90$, three pairs at $0.50$.
* **12 empty pairs:** the model predicts a fully empty mask on 10 of them; on the remaining 2 it predicts blobs of **512 pixels** each (canvas $= 102{,}400$ px).

Compute (a) $\text{ForegroundDice}$, (b) $\text{FalsePositiveScore}$, (c) $\text{EmptyAccuracy}$, (d) the final weighted Score. Round each to 4 decimal places.

<details>
<summary><b>Worked Solution</b></summary>

**(a)** Positive pairs only:
$$\text{FG} = \frac{5(0.90) + 3(0.50)}{8} = \frac{4.50 + 1.50}{8} = \frac{6.00}{8} = \mathbf{0.7500}$$

**(b)** Empty pairs only. Each 512-px blob: $\frac{512}{102{,}400} = 0.005$.
$$\text{mean fp fraction} = \frac{10(0) + 2(0.005)}{12} = \frac{0.01}{12} \approx 0.0008\overline{3}$$
$$\text{FP} = 1 - 0.0008\overline{3} = \mathbf{0.9992}$$

**(c)** Empty pairs predicted perfectly empty:
$$\text{EA} = \frac{10}{12} = \mathbf{0.8333}$$

**(d)** Combine:
$$\text{Score} = 0.70(0.7500) + 0.20(0.9992) + 0.10(0.8333)$$
$$= 0.5250 + 0.1998 + 0.0833 = \mathbf{0.8082}$$

**Pitfall check:** did you average Dice over *all* 20 pairs in step (a)? That would drag the positive-only average toward the empty pairs' scores and corrupt the component — the three components live on **disjoint** subsets by design.

</details>

### Practice Problem 7.2 — The all-empty baseline under hypothetical weights
A *different* competition scores with weights $0.50 \times \text{FG} + 0.30 \times \text{FP} + 0.10 \times \text{EA}$. A student submits an all-empty model ($\hat{X} = \emptyset$ for every pair) on a split of their choice.

1. What are the model's three component values on this split? (Use the component definitions from Section 1 — think, do not look up.)
2. What Score does it receive under the hypothetical weights?
3. Does your answer change if the split had a different emptiness rate? Why?
4. Reflect: under naive empty-aware mean Dice, what does the same baseline score equal? (Write the expression, not a number.) Why is that one data-dependent while the weighted score is not?

<details>
<summary><b>Worked Solution</b></summary>

1. $\text{FG} = 0.0$ (it never overlaps any positive mask), $\text{FP} = 1.0$ (no false-positive pixels at all), $\text{EA} = 1.0$ (every empty pair predicted empty).
2. $\text{Score} = 0.50(0) + 0.30(1) + 0.10(1) = \mathbf{0.40}$.
3. **No.** The component values $(0, 1, 1)$ hold for *any* split, and the weights are constants — the product never sees the data. Feed in any emptiness rate and the baseline still scores 0.40.
4. Naive mean Dice $= N_{\text{empty}} / N$ — the split's **emptiness rate**. Change the split and this number moves with it, which is why a blind model can look strong on empty-dominated validation sets. The weighted score cannot be inflated that way because 70% of its weight is reserved for a term the blind model scores 0 on.

</details>

---

## 5. Your Assignment Task

Open the graded assignment: [`MILESTONE_1.md`](../../MILESTONE_1.md)

* **Question 8** gives you a larger toy batch with its own counts and asks for the three components and the final weighted score. Practice Problem 7.1 walked the identical arithmetic with different numbers — watch the component definitions (positive pairs vs. empty pairs) as you go.
* **Question 9** asks you to evaluate the all-empty baseline over the **entire training set** using only `train.csv`: count the ground-truth empty pairs yourself (Section 2's $N_{\text{empty}}/N$ expression, with $N = 45{,}572$), then apply Section 2's component argument with the official weights for the competition score — plus your own explanation of why the two diverge. No validation folds, fold files, or trained models are involved: the paradox is fully visible in the raw label statistics.

Compute both numbers yourself from the raw data; this guide intentionally stops short of evaluating them.
