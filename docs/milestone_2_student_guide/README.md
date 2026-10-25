# Milestone 2 Student Study Guides
## Building the Segmentation Pipeline & First Submission

Welcome to Milestone 2! In Milestone 1, you inspected raw annotations, unraveled Run-Length Encoding (RLE) memory layouts, explored line-scan sensor geometry, and studied metric sensitivity. 

In Milestone 2, you move from analytical observation to **active systems engineering**: mastering RLE encoding and decoding on 2D grids, analyzing global defect prevalence and multi-label co-occurrence patterns, testing physical pixel-level exclusivity, choosing between multi-channel and prompt-conditioned architectures, and implementing a robust PyTorch training pipeline to diagnose model convergence.

---

## Curriculum Map: Questions & Guided Learning Modules

Every study guide corresponds directly to an assessment question in [`MILESTONE_2.md`](../../MILESTONE_2.md). Study the core concepts, review the step-by-step guiding hints, and then compute your results on the competition data in `data/public/`.

| Milestone Question | Guided Learning Document | Theoretical Principles | Guiding Focus & Hints |
| :--- | :--- | :--- | :--- |
| **Question 1** | [Guide 1: RLE Standards & Column-Major Grid Encoding](guide_01_rle_column_major_encoding.md) | 1-based indexing, Fortran column-major (`order='F'`) traversal, linear coordinate mapping, active run pair formation. | Coordinate conversion formula, boundary run merging, manual tracing hints. |
| **Question 2** | [Guide 2: RLE Decoding & Multi-Defect Pixel Mass](guide_02_rle_decoding_and_pixel_mass.md) | Inverting RLE strings into 2D $(128, 800)$ masks, handling empty/NaN entries, multi-defect row aggregation, total positive pixel mass. | 5-step decoding logic, parsing token pairs, multi-defect mass summation. |
| **Question 3** | [Guide 3: Global Class Distribution & Defect Prevalence](guide_03_global_class_distribution_and_prevalence.md) | Long-format row structure (45,572 rows across 11,393 images), marginal defect frequencies, severe class imbalance in manufacturing. | Filtering active defect entries, aggregating unique images per class, identifying extremes. |
| **Question 4** | [Guide 4: Multi-Label Defect Co-Occurrence Matrix](guide_04_defect_cooccurrence_matrix.md) | Compound manufacturing errors, multi-label indicator vectors, $4 \times 4$ co-occurrence matrix formulation $C = \mathbf{Y}^T \mathbf{Y}$, joint failure patterns. | Formulating symmetric indicator matrices, diagonal vs. off-diagonal pairs. |
| **Question 5** | [Guide 5: Co-Occurrence vs. Correlation (The Base-Rate Effect)](guide_05_cooccurrence_vs_correlation_base_rate.md) | The Base-Rate Fallacy in defect associations, expected independent co-occurrence, Pearson correlation coefficient ($\phi$ coefficient) for binary vectors. | Normalizing co-occurrence for marginal base rates, interpreting correlation values. |
| **Question 6** | [Guide 6: Pixel-Level Spatial Overlap & Mutual Exclusivity](guide_06_pixel_level_spatial_overlap_and_mutual_exclusivity.md) | Image-level co-occurrence vs. pixel-level spatial intersection, pairwise mask overlap $|M_i \cap M_j|$, pairwise Dice, 4-Sigmoid vs. 5-Softmax head validity. | Multi-label mask decoding, pairwise pixel intersection checks, architectural validity. |
| **Question 7** | [Guide 7: PyTorch Dataset Architecture & Loss Typing](guide_07_pytorch_dataset_and_architectural_paradigms.md) | Multi-Channel Joint Prediction (Paradigm 1) vs. Prompt-Conditioned Segmentation (Paradigm 2), PyTorch `Dataset` contract, 4D tensor collation $[B, C, H, W]$, floating-point targets for `BCEWithLogitsLoss` and the assignment’s `torch.float32` mask contract. | Dataset methods contract, tensor dimensions, floating-point target representation in BCE. |
| **Question 8** | [Guide 8: Baseline Model Training Dynamics & Convergence](guide_08_training_dynamics_and_convergence.md) | 5-epoch baseline U-Net training lifecycle, monitoring train vs. val loss trajectories, diagnosing underfitting vs. overfitting, optimal checkpoint selection. | Tracking epoch loss pairs, diagnosing generalization regimes, checkpoint selection. |

---

## Architectural Formulations Overview

In Milestone 2, you also explore two fundamentally different paradigms for formulating multi-defect segmentation:

1. **Paradigm 1: Multi-Channel Joint Prediction (Concurrent Multi-Task):**
   A shared encoder-decoder backbone processes the steel strip and concurrently predicts all four defect channels in a single forward pass ($f_\theta: \mathbf{X} \in \mathbb{R}^{3 \times H \times W} \to \mathbf{\hat{Y}} \in \mathbb{R}^{4 \times H \times W}$). Highly throughput-efficient, but subject to gradient competition between dominant and rare classes.
2. **Paradigm 2: Conditioned / Prompted Segmentation (Query-Based):**
   Inspired by Whisper's prefix prompt conditioning tokens, segmentation is reframed as an interactive query ($f_\theta: (\mathbf{X}, c) \to \mathbf{\hat{Y}} \in \mathbb{R}^{1 \times H \times W}$). Class-directed outputs normally share learned parameters. Predicting all four classes needs four conditioned outputs, but queries may be batched or shared image features cached; inference cost depends on the implementation.

---

## How to Work Through Milestone 2:

1. **Read the corresponding Concept Guide** before attempting each question to understand the engineering context and theoretical principles.
2. **Review the Guiding Hints** to understand how to approach the calculation or implementation without leaking solutions.
3. **Write and run your own Python scripts** against `data/public/` to solve the real task described in the "Your Assignment Task" section of each guide.
4. Enter your final answers into the assessment sheet:
   👉 [`MILESTONE_2.md`](../../MILESTONE_2.md)


## Previously Shared Resources

Earlier pipeline-oriented URLs remain available through compatibility pages. The [legacy synthetic walkthrough and downloads](walkthrough-notebook.html) use a grayscale/dictionary contract; they are reference material rather than the RGB/tuple Dataset contract required by current Q7.
