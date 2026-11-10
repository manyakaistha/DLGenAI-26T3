# Milestone 4: Training Improvements & Model Optimization

Industrial Surface Anomaly Segmentation · Student Assessment

Read the [Milestone 4 study guides](docs/milestone_4_student_guide/README.md) before attempting these questions. This is an assessment sheet, not a worked answer set: compute data-derived values from the specified subset and report measured training outcomes from your own run. Synthetic practice examples are in the guides. Do not use full-dataset counts or leaderboard feedback.

## Instructions

- Work locally, in Google Colab, or in a Kaggle notebook with the official competition data.
- Follow each answer format exactly and round only where requested.
- Log every training run in your W&B project named `YourRollNo-t32026`; include the configuration, seed, split, and epoch metrics.
- Keep the assessment questions and responses in this document. Study guides teach the methods and use different toy values.

## Common Setup (data-driven questions)

**Working subset:** From `train.csv`, take the unique `image_id` values, sort them alphabetically in ASCII order, retain the first 1,200 IDs, and keep all four class rows for each image (4,800 image-class pairs). Decode masks on the native `128 × 800` canvas in 1-based Fortran order. Clip any run extending beyond the 102,400-pixel canvas, as in Milestone 1.

**Split:** For each image, construct a combination string from its present classes (`clean`, `defect_c`, or a joined combination such as `defect_a+defect_c`). For the stratified split, merge combinations with fewer than 3 images into `rare_other`; use an 80/20 image-level split stratified by this string with `random_state=42`. This yields 960 training and 240 validation images. Questions 1–3 and 7–9 use the full 1,200-image subset where stated; training questions use this split.

**Training budget (Questions 8 and 10):** Use the same model as your Milestone 2/3 baseline and the following fixed settings: native `128 × 800` input, AdamW with learning rate `3e-4`, batch size 16, AMP enabled, seed 42, decision threshold 0.5, and no minimum-area filtering. Keep the architecture, data pipeline, split, loss, optimizer, and evaluation code fixed unless a question explicitly varies one component. Use the official validation score defined in Question 7.

---

## Question 1 · Image-Level and Class-Level Imbalance

Use the full 1,200-image subset (not the train/validation split).

1. How many images are completely clean?
2. How many images are positive for each of `defect_a`, `defect_b`, `defect_c`, and `defect_d`?
3. How many unique class-combination strings exist before merging rare groups?
4. What are the sizes of the largest and smallest combination groups?

**Answer format:** `clean; defect_a, defect_b, defect_c, defect_d; unique_combinations; largest, smallest` (integers)

## Question 2 · Pixel-Level Imbalance and `pos_weight`

Use the same 1,200-image subset and decode every `defect_b` mask.

1. What percentage of all `defect_b` mask pixels (1,200 × 102,400) are positive? Give 4 decimal places.
2. What is the mean `defect_b` area, in pixels, among images with a positive `defect_b` mask? Give 1 decimal place.
3. What is the negative-to-positive pixel ratio, a candidate `pos_weight` for BCE? Round to the nearest integer.

**Answer format:** `positive_percent, mean_positive_area, negative_to_positive_ratio` (%, 4 decimals; pixels, 1 decimal; integer)

## Question 3 · Bounded Square-Root Sampling

Use the unmerged combination groups from Question 1 (all 1,200 images). Let $n_g$ be the size of combination group $g$, and $n_{clean}$ the clean-group size. Every image in group $g$ receives per-image weight $w_g=\min(\sqrt{n_{clean}/n_g},5)$. Draw with replacement in proportion to per-image weights; one epoch contains 1,200 draws.

1. What percentage of draws is expected to be clean under this sampler and under natural uniform sampling?
2. What is the expected number of `defect_b`-positive image draws per epoch (any group containing `defect_b`) under the sampler and under natural sampling?
3. By what factor does the sampler enrich `defect_b`-positive images?
4. How many combination groups reach the weight cap of 5?

Remember that a group's total sampling mass is $n_g w_g$, not merely $w_g$.

**Answer format:** `clean_sampler%, clean_natural%; defect_b_sampler, defect_b_natural; enrichment_factor; capped_groups` (percentages, expected draw counts, and enrichment factor to 2 decimals; capped group count as an integer)

## Question 4 · Warmup + Cosine Schedule and Stepping Cadence

On the 960-image training split, batch size 16 gives 60 optimizer steps per epoch. Train for 8 epochs (480 optimizer steps) with base learning rate $3×10^{-4}$. The schedule multiplier is $m(t)=t/60$ for scheduler index $t<60$, and $m(t)=\frac{1}{2}[1+\cos(\pi(t-60)/(480-60))]$ for $60≤t≤480$. The learning rate is base LR times the multiplier. For this question, `t` is the scheduler's index: requested step values are indices, and after eight calls to a step-indexed scheduler the index is 8.

1. What learning rate is used at steps 15, 270, and 400?
2. A student writes the schedule in optimizer-step units but calls `scheduler.step()` only once per epoch. What learning rate is used after epoch 8 under that bug?
3. In one sentence, describe the symptom this stepping bug would create in training curves.

**Answer format:** `lr15, lr270, lr400; lr_after_epoch8` (scientific notation, 3 significant figures); then one sentence.

## Question 5 · Early-Stopping Trace

Monitor official validation score in `max` mode with patience 3 and `min_delta=0.002`. An epoch improves only when `score > best + min_delta`; otherwise increment the patience counter. Reset it on improvement. Save the best checkpoint only on improvement.

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation score | 0.512 | 0.601 | 0.648 | 0.661 | 0.664 | 0.668 | 0.671 | 0.670 | 0.672 | 0.669 | 0.673 | 0.665 |

1. Stop as soon as the bad-epoch counter reaches patience. At what epoch does training stop? Which epoch's checkpoint is restored, and what is its score?
2. With `min_delta=0`, does training stop within these 12 epochs? Which epoch holds the best checkpoint at epoch 12?

**Answer format:** `stop_epoch, restored_epoch, restored_score; stops_with_zero_delta (yes/no), best_epoch_at_12` (score to 3 decimals)

## Question 6 · Mixed Precision, GPU Memory, and Gradient Accumulation

A U-Net has 25,000,000 parameters. Use decimal MB (1 MB = (10^6) bytes). Weights, gradients, and both AdamW moment buffers use FP32 (4 bytes each), including under AMP. Activations use 150 MB per sample in FP32 and half that under FP16 autocast. Ignore workspace, autocast weight cache, and framework overhead.

**Part A — Memory estimate**

1. How many MB do weights, gradients, and AdamW state require?
2. What is estimated total memory at batch size 16 in FP32 and under AMP?
3. With a 3,400 MB budget, what is the largest batch size fitting under each precision assumption?

**Part B — Accumulation**

Use the 960-image training split, microbatch size 8, and 4 accumulation steps. Do not drop a final partial batch (verify whether one occurs). Count optimizer steps per epoch. A 2-epoch warmup is counted in optimizer steps.

4. What is the effective batch size?
5. How many microbatches and optimizer steps occur per epoch?
6. By what factor should each microbatch loss be divided before `backward()`?
7. How many optimizer steps does the 2-epoch warmup contain?

**Answer format:** `state_MB; FP32_MB, AMP_MB; FP32_batch, AMP_batch; effective_batch; microbatches, optimizer_steps; loss_divisor; warmup_steps` (integer MB/batches/steps unless fractional arithmetic is required)

## Question 7 · The Naive Dice Trap and Official Metric

Use the empty-pair convention (Dice(∅,∅)=1) and (Dice(∅, nonempty)=0). The official score is (0.70·ForegroundDice + 0.20·FalsePositiveScore + 0.10·EmptyAccuracy), where `FalsePositiveScore = 1 − mean(|P|/102,400)` over empty target pairs and `EmptyAccuracy` is the share of empty pairs predicted fully empty.

**Part A — Data-driven:** Evaluate an all-empty baseline on all 4,800 image-class pairs in the 1,200-image subset.

1. How many pairs have empty ground truth?
2. What is the baseline's naive empty-aware mean Dice?
3. What is its official score?

**Part B — Toy calculation:** A validation set has 400 pairs: 60 positive pairs with `ForegroundDice=0.50`, and 340 empty pairs. The model predicts a fully empty mask on 306 empty pairs and a 1,024-pixel false blob on each of the other 34.

4. What are this model's official score and naive Dice?
5. Which metric ranks the model below the all-empty baseline on this set? What is the baseline's naive Dice?

**Answer format:** `empty_pairs; baseline_naive, baseline_official; toy_official, toy_naive; metric_name, toy_baseline_naive` (scores to 4 decimals)

## Question 8 · Training Dynamics and Checkpoint Selection

Train the baseline on the 960/240 split for 6 epochs using the Common Setup and the warmup + cosine schedule from Question 4, retaining that schedule's 8-epoch/480-step horizon. Log training loss, validation loss, and official validation score at every epoch to W&B.

1. At which epoch is validation loss lowest?
2. At which epoch is official validation score highest, and what is that score? (3 decimals)
3. What is the generalization gap `validation loss − training loss` at epoch 6? (3 decimals)
4. Which quantity should guide checkpoint selection, which epoch's checkpoint would you keep, and what learning-curve pattern best describes the run (for example, still underfitting, healthy convergence, or overfitting)?

**Answer format:** `lowest_val_loss_epoch; best_score_epoch, best_score; gap; selection_metric, kept_epoch, curve_pattern`

## Question 9 · Defect-Size Bins and Presence vs. Overlap

Use the full 1,200-image subset. For every positive image-class pair (pool all four classes), compute the pair's total decoded-mask area and place it in one bin: 1–100 px, 101–500 px, or 501+ px.

1. How many positive pairs fall in each bin?
2. What percentage of all positive ground-truth pixels lies in the 501+ bin?
3. Suppose a hypothetical model has presence recall of 30% for small, 70% for medium, and 95% for large pairs. What is the expected total number of missed positive pairs, and which bin contributes the most missed pairs?
4. In one sentence, explain why the bin with the most missed pairs can differ from the bin with the most missed pixels.

**Answer format:** `small, medium, large; large_bin_pixel_percent; missed_pairs, largest_missed_bin; one_sentence` (percent to 2 decimals; expected missed count to 2 decimals)

## Question 10 · Controlled Experimentation (OFAT)

Use the Common Setup split and 6-epoch budget. Log every run in W&B. `R0` is the baseline (constant learning rate, natural sampling, AMP on). `R0b` is identical to R0 but uses seed 7 and estimates seed noise. `R1` is R0 plus warmup/cosine only. `R2` is R0 plus the bounded square-root sampler only, using weights computed on the 960 training images as in Question 3. All other settings stay fixed.

Treat `|score(R0) − score(R0b)|` as the noise band. A change counts as a real improvement only if its gain is greater than twice this band.

1. What is each run's best official validation score: R0, R0b, R1, and R2?
2. What is the noise band and the isolated gain of R1 and R2 over R0?
3. Does the R1 gain exceed twice the noise band? Does the R2 gain? Answer yes/no for each.
4. For R0 and R2, what percentage of empty validation pairs has at least one predicted pixel (clean-image false-alarm rate)? Use each run's best checkpoint and threshold 0.5.
5. In one sentence, explain why a single run that enables the sampler and warmup/cosine together cannot identify which change helped.
6. In one sentence, explain what the change in false-alarm rate between R0 and R2 suggests about prior shift from sampling.

**Answer format:** `R0, R0b, R1, R2; noise_band, R1_gain, R2_gain; R1_yes/no, R2_yes/no; R0_false_alarm%, R2_false_alarm%; sentence; sentence` (scores/gains to 3 decimals; rates to 1 decimal)
