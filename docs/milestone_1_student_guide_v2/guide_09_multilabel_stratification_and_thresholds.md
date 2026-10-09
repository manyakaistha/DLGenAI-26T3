# Multi-Label Stratification and Decision Threshold Sweeps
*(Preparing for Milestone 1: Questions 11 & 12)*

---

## 1. Why a Naive Random Holdout Fails for Rare Defects
In this milestone you evaluate with a **single 80/20 train/validation holdout** — no k-fold cross-validation, no repeated splits. One split keeps the pipeline simple, but it makes *how* you draw that split critically important: you get exactly one draw, so any imbalance it introduces persists through every metric you report.

When working with imbalanced multi-label datasets, a naive random holdout causes severe problems:
* In our dataset, `defect_b` appears in only 313 training images (2.75% prevalence — a small fraction; `defect_c`, by contrast, has thousands).
* If you split purely at random, the number of positive `defect_b` images that land in the 20% validation partition is a random draw: with so few positives, chance alone can put noticeably more or fewer than the exact 20% proportion into validation.
* Any deviation means your validation metrics reflect *split luck* as much as model quality — and for a rare class, the relative deviation can be enormous.

**The statistical picture:** think of the holdout draw as a lottery. With a rare item and random drawing, one draw can be lucky or unlucky — and unlike k-fold averaging, you do not get redraws to wash it out. Stratification is a *rigged* lottery that guarantees the exact proportion.

Quantify both sides yourself: the expected validation count under exact 80/20 proportions, and the standard deviation $\sigma = \sqrt{n \, p(1-p)}$ of the random draw. Question 11 asks for the exact-proportion validation count. The standard deviation is optional background; Practice Problem 9.1 rehearses it on toy counts.

---

## 2. Multi-Label Combination Stratification
To fix this, we apply **Combination Stratification** to the single holdout:
1. For each image, encode its active defect combination into a single string (e.g., `"clean"`, `"defect_c"`, `"defect_c+defect_d"`, `"defect_a"`, `"defect_b+defect_c"`, etc.).
2. Count the unique defect combinations present in the dataset (the assignment asks you for the exact number — count, do not guess).
3. Allocate the holdout *within* each combination stratum so that the validation partition receives the exact same proportion of clean images, single defects, and rare multi-label co-occurrences as the full dataset.

**Why stratify on the *combination* rather than each class separately?** Per-class stratification handles one label at a time, but our data is multi-label: it can silently distort *co-occurrence* patterns (the holdout might get enough `defect_b` images but in the wrong combinations). Treating each distinct combination as its own stratum preserves both marginal frequencies and co-occurrence structure in one step.

For Question 11's idealized calculation, fractional counts are allowed, so each stratum contributes exactly 20% to validation. An actual holdout needs integer allocations: fix a random seed, keep images disjoint, and record any rounding or rare-stratum exceptions.

---

## 3. Baseline Model & Decision Threshold Sweeps ($\tau$)
When a baseline U-Net outputs predicted probabilities $\hat{P}(y, x) \in [0, 1]$, how do you convert them into a binary mask? You need a cutoff — the **decision threshold** $\tau$:
$$\hat{M}(y, x) = \begin{cases} 1 & \text{if } \hat{P}(y, x) \ge \tau \\ 0 & \text{if } \hat{P}(y, x) < \tau \end{cases}$$

**Why is $\tau$ not always 0.5?** The default 0.5 assumes the model's probabilities are perfectly calibrated and that false positives and false negatives cost the same. In this dataset they do *not* cost the same: false positives on the many empty images are punished by the metric structure, while false negatives on rare defects destroy your 70% Foreground Dice term. Sweeping $\tau$ lets you find the operating point that best matches *the actual scoring function*.

### The Precision-Recall Trade-off across $\tau$:
* **Low Thresholds ($\tau = 0.3$):** Captures faint defect pixels (high recall), but predicts noisy false alarms on clean images (poor precision).
* **High Thresholds ($\tau = 0.8 - 0.9$):** Extremely conservative. Clean images are safely predicted empty (high specificity), but faint or thin defects are missed.

### The Threshold Divergence Between Naive Dice and Competition Score:
The two metrics weight "empty-image behavior" and "defect-image behavior" very differently, so they do **not** reward the same operating point. Before you read anyone else's sweep results, commit to predictions on paper for each question below — then run your own sweep and check them:

1. **As $\tau$ climbs from 0.3 to 0.9,** what happens to the false-positive rate on empty images? And to Foreground Dice on positive images — does it move monotonically, or does it rise and then fall? *Sketch your predicted curves.*
2. **Where would you expect naive mean Dice to prefer to sit** — at low $\tau$, moderate $\tau$, or high $\tau$? Recall from Guide 7 that naive mean Dice averages over *all* pairs, and empty pairs are the majority. What behavior does that majority reward?
3. **Where would you expect the official competition score to prefer to sit?** Its largest weight (70%) lives on Foreground Dice computed *only over positive pairs*. What happens to your overall score if you abstain so aggressively that foreground recall collapses?
4. **Do the two optima coincide?** If not, which metric's "best $\tau$" would you trust for the leaderboard, and why?

> **Takeaway:** the "best threshold" depends entirely on which metric you optimize. Always sweep $\tau$ and report both metrics — picking $\tau$ by naive Dice alone will systematically mis-shape your predictions. Your graded task is to produce the actual sweep and report where each metric peaks *on your model's outputs*; the reasoning framework above is how you will explain the divergence.

---

## 4. Official Trusted Resources & Recommended Reading
* **Sechidis, K., Tsoumakas, G., & Vlahavas, I. (2011).** *On the Stratification of Multi-label Data*. ECML PKDD 2011, LNCS 6913, 145–158. [DOI: 10.1007/978-3-642-23808-6_10](https://doi.org/10.1007/978-3-642-23808-6_10).
* **Szymański, P., & Kajdanowicz, T. (2017).** *A Network Perspective on Stratification of Multi-Label Data*. PMLR 74: 22–35. [arXiv:1704.08756](https://arxiv.org/abs/1704.08756).
* **Baretta, Trent.** [`iterative-stratification` Library on GitHub](https://github.com/trent-b/iterative-stratification).

---

## 5. Practice Problems: Holdout Composition and Threshold Reasoning

### Practice Problem 9.1 — Expected count and $\sigma$ for a toy holdout
A toy dataset has 40 images: a rare class $R$ appears in exactly 10 of them, a common class $C$ in 20. A **random 20% holdout** draws 8 images uniformly.

1. Under exact 80/20 stratification, how many $R$-positive images land in the holdout? And $C$-positive?
2. Under the random draw, what is the **expected** number of $R$-positives in the holdout?
3. Compute the standard deviation $\sigma = \sqrt{n \, p(1-p)}$ of the $R$-count ($n = 8$, $p = 10/40$) and of the $C$-count ($p = 20/40$). Round to 4 decimals.
4. Compare the *relative* deviation $\sigma / \text{expectation}$ for $R$ vs $C$. Which class's holdout support is more fragile, and why does stratification matter more for it?
5. Suppose bad luck gave the random holdout **zero** $R$-positives. What happens to a foreground-Dice-style metric computed on class $R$ — defined or not? What does that do to your trust in the split?

<details>
<summary><b>Worked Solution</b></summary>

1. $R$: $0.2 \times 10 = \mathbf{2}$ exactly; $C$: $0.2 \times 20 = \mathbf{4}$ exactly.
2. $E[R] = 8 \times \frac{10}{40} = 8 \times 0.25 = \mathbf{2.0}$ — the same center as stratification; what changes is the spread around it.
3. $\sigma_R = \sqrt{8 \times 0.25 \times 0.75} = \sqrt{1.5} \approx \mathbf{1.2247}$; $\quad \sigma_C = \sqrt{8 \times 0.5 \times 0.5} = \sqrt{2} \approx \mathbf{1.4142}$.
4. Relative deviation: $R$: $1.2247 / 2.0 \approx \mathbf{61\%}$; $C$: $1.4142 / 4.0 \approx \mathbf{35\%}$. The rare class is far more fragile — the absolute sigmas are comparable, but $R$'s expectation is half of $C$'s, so a single $\pm 1\sigma$ draw can nearly halve or triple its validation support. Stratification sets both counts to exactly 2 and 4 ($\sigma = 0$ by construction), which is precisely the guarantee Question 11 asks you to argue for.
5. **Undefined** — foreground Dice over zero positive samples is $0/0$ (Guide 6's dilemma, now at split level). Frameworks skip or coerce it, so the metric depends on split luck rather than model quality. Stable, guaranteed rare-class support in the holdout is *why* you stratify.

</details>

### Practice Problem 9.2 — Predict the metric-optimal $\tau$ direction
Two competing objectives on a sweep of $\tau \in [0.3, 0.9]$:

1. Naive mean Dice over *all* image-class pairs, where empty pairs vastly outnumber positive pairs.
2. The official competition score, whose 70% term is Foreground Dice on *positive pairs only*.

For each objective, answer without running any code:

* In which **direction** does moving $\tau$ upward shift the trade-off (more abstention vs. more recall)?
* Which objective **rewards aggressive abstention more strongly**, and why?
* If your model's raw probabilities were poorly calibrated (say, systematically too low), how would that distort the naive-Dice-optimal $\tau$ compared to the competition-optimal one?

<details>
<summary><b>Worked Solution</b></summary>

* Raising $\tau$ predicts fewer positive pixels: **false positives on empty images fall**, and **foreground recall (hence Foreground Dice) eventually falls** too. Lowering $\tau$ does the reverse.
* **Naive mean Dice rewards abstention far more strongly:** every empty pair it converts to a clean 1.0 counts fully in the average, while the positive pairs (where abstention destroys score) are the minority. The competition score also values empty-image cleanliness, but only for 30% of the weight — 70% is reserved for foreground overlap, which aggressive abstention directly harms.
* Poorly calibrated (too-low) probabilities push the *naive*-optimal $\tau$ downward relative to where good calibration would put it — you must lower the cutoff just to keep predicting anything — and the two optima can drift apart by different amounts, because one metric is far more sensitive to the abstention/recall balance than the other. This is why you sweep on **raw outputs** and report **both** metrics: each one exposes a different failure mode of your operating point.

</details>

---

## 6. Your Assignment Task

Open the graded assignment: [`MILESTONE_1.md`](../../MILESTONE_1.md)

* **Question 11** asks you to design a single stratified 80/20 holdout over the real training set: count the unique combination strings and compute the exact-proportion `defect_b` count in validation, rounded to the nearest integer. Practice Problem 9.1 rehearsed the expected-count/$\sigma$ mechanics on toy numbers.
* **Question 12** asks you to train your baseline on the 80% partition, sweep thresholds on the 20% holdout (no min-area filtering), describe both metrics' trends, and analyze which metric prefers the higher thresholds and why — using the prediction framework from Section 3 and Practice Problem 9.2.

The holdout counts, the training run, and the sweep curves are all things you must produce; this guide gives you the reasoning scaffold and verifies your statistical mechanics on toy numbers only.


## Starter Notebook for Questions 11 and 12

**[Open in Google Colab](https://colab.research.google.com/drive/1Jf4f7jKddRSMpiZ1D8oVqmR8LDK-t86Q?usp=sharing)**

Save a copy in Drive, choose a GPU runtime, install the dependencies, and set `DATA_ROOT` to your competition data folder. Then run the notebook cells in order: complete the Q11 calculations, inspect the masks, train for 5–10 epochs, and evaluate thresholds 0.3, 0.5, 0.7 and 0.9 without minimum-area filtering. Write your Q12 bullet points from your own results.

[Full setup instructions and notebook downloads](../starter_code_q11_q12/README.md) include the Colab dependency cell, data-path examples, and how to save your experiment outputs.
