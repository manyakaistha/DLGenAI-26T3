# Milestone 1 Student Study Guides
## Industrial Surface Anomaly Segmentation

Welcome to Milestone 1! This curriculum prepares you to understand the data, master the evaluation metrics, and build your baseline segmentation models from scratch.

This is the **assessment edition**: like the learning edition, it teaches you the **underlying engineering and mathematical principles** of every milestone question — but it deliberately **contains no computed answers or answer hints for the 12 graded questions**. Instead, each guide ends with:

1. **Practice Problems** — fully worked exercises on small *toy* datasets that teach you the exact method and intuition you need, with different numbers from the graded task.
2. **Your Assignment Task** — a short pointer telling you which question the guide prepares you for, so you apply the practiced method to the real data yourself.

---

## Concept Guides & Reading Modules

| Milestone Question | Guided Learning Document | Theoretical Principles | Practice Focus |
| :--- | :--- | :--- | :--- |
| **Q1 & Q2** | [Guide 1: Anomaly Sparsity & Multi-Label Cardinality](guide_01_data_sparsity_and_cardinality.md) | Industrial yield rates, clean sheet dominance, long-format CSV structure, multi-label cardinality distributions. | Count per-class positives, clean images, and cardinality on a toy long-format CSV. |
| **Q3** | [Guide 2: Spatial Mutual Exclusivity & Output Architecture](guide_02_spatial_exclusivity_and_output_heads.md) | Surface defect physics, pixel-level intersection testing, 4-Sigmoid vs. 5-Softmax head design. | Compute pairwise mask overlaps on toy masks and reason about which output head fits the evidence. |
| **Q4** | [Guide 3: RLE Encodings and Memory Traversal](guide_03_rle_decoding_and_boundary_checks.md) | Run-Length Encoding, Fortran (`order='F'`) vs. C order, 1-based indexing, buffer boundary clamping. | Decode a toy RLE string on a tiny canvas; detect and clamp boundary-violating runs. |
| **Q5** | [Guide 4: Anisotropic Sensors and Aspect Ratio Distortion](guide_04_sensor_aspect_ratios_and_aliasing.md) | Industrial line-scan cameras, anisotropic strip geometry ($128 \times 800$), Nyquist-Shannon aliasing, thin crack collapse. | Compute per-axis scale factors and the distortion ratio for non-square resize targets. |
| **Q6** | [Guide 5: Photometric Separation vs. Pure Texture](guide_05_photometric_contrast_vs_texture.md) | Cohen's $d$ effect size, 1st-order intensity vs. 2nd-order texture, why simple thresholding fails on texture defects. | Compute Cohen's $d$ from toy intensity statistics and classify photometric vs. texture regimes. |
| **Q7** | [Guide 6: The Sørensen–Dice Index and Negative Step Functions](guide_06_dice_metric_and_empty_mask_dilemma.md) | Dice/$F_1$ formulation, $0/0$ empty convention, the discontinuous Heaviside step penalty on clean samples. | Score toy masks under the empty convention, including empty-GT edge cases. |
| **Q8 & Q9** | [Guide 7: The Competition Scorer & All-Empty Paradox](guide_07_competition_metric_and_baseline_pitfalls.md) | Official 70/20/10 weighted competition scorer, the "All-Empty Baseline Paradox" (naive vs. competition score). | Score a toy batch under the three components; count GT-empty pairs from raw `train.csv` and evaluate an all-empty baseline (official + hypothetical weights). |
| **Q10** | [Guide 8: Morphological Post-Processing & Recall Ceilings](guide_08_morphological_postprocessing_and_recall.md) | 8-connected components (`cv2.connectedComponentsWithStats`), noise speckle filtering, mathematical recall ceilings. | Compute the recall-ceiling formula from a toy list of component areas. |
| **Q11 & Q12** | [Guide 9: Multi-Label Stratification & Decision Thresholds](guide_09_multilabel_stratification_and_thresholds.md) | Combination-stratified 80/20 holdout (no k-fold), rare-class split-luck $\sigma$, baseline U-Net, decision threshold $\tau$ sweep. | Compute expected holdout counts and $\sigma$ for toy splits; reason qualitatively about threshold movement under each metric. |

---

## How to Work Through Milestone 1:
1. **Read the corresponding Concept Guide** before attempting each question.
2. **Work every Practice Problem on paper or in a scratch notebook first**, and check yourself against the provided worked solution — the practice uses toy numbers, so it does not give away your graded answers.
3. **Write your own Python scripts** against `data/public/` to solve the real task pointed to in the "Your Assignment Task" section.
4. Enter your answers into the assignment form:
   👉 [`MILESTONE_1.md`](../../MILESTONE_1.md)
