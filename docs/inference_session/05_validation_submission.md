# Validation, runtime and submission

## Read every denominator

The published competition description assigns **70% Foreground Dice, 20% False Positive Score and 10% Empty Prediction Accuracy**. The prose does not fully specify every reduction or sub-score formula. Use an organizer scorer when available. A locally invented combination must be labelled an estimate.

This lesson's notebook reports two explicit diagnostics:

- **Positive-pair Dice:** compute Dice per target-positive image–class pair, then take the mean. Dice is $2TP/(2TP+FP+FN)$.
- **Empty-pair specificity:** fraction of target-empty pairs whose predicted mask is completely empty. This is a pair-level diagnostic, not pixel specificity $TN/(TN+FP)$.

An absent subgroup produces `None`, not an invented perfect or zero score. Keep support counts, per-class results and qualitative overlays beside aggregates. Do not report these diagnostics as the official score.

## An ablation ladder

| Candidate | Single change | Evidence to record |
| --- | --- | --- |
| A | Raw checkpoint, identity view | Baseline quality, latency, peak memory |
| B | EMA checkpoint | Same split and decisions, checkpoint identity |
| C | Two-view TTA on selected checkpoint | Alignment check and added latency |
| D | Small probability ensemble | Common holdout and complementary errors |
| E | Tuned thresholds | Selection/confirmation split and search grid |
| F | Component filtering | Small-defect recall and empty-pair changes |

This ordering is one teaching path. Each rung must justify its own cost. If a change fails the confirmation criterion, keep the simpler recipe. Record model hashes, software versions, normalization, precision, view list, aggregation, thresholds, filter connectivity and output resolution.

## Runtime and memory

$M$ models with $K$ views need $M\times K$ forward evaluations per batch. EMA alone uses one evaluation per view. This is an evaluation count, not a measured wall-clock speed claim. Benchmark warm runs on the same device and batch size. Synchronize CUDA around timing. Include preprocessing and RLE if claiming end-to-end latency.

One FP32 probability tensor for an image is `4 × 128 × 800 × 4 = 1,638,400` bytes, about **1.56 MiB**. Caching 3,000 images costs about **4.58 GiB** before metadata or extra copies. More models multiply the storage if every cache is retained.

Stream a batch at a time. Accumulate probabilities in FP32, threshold/filter, encode and write rows, then release intermediates. Sequential model loading can reduce resident weight memory but adds loading overhead. Streaming bounds memory with respect to dataset size, not independently of batch size, model weights or the native image dimensions. Reduced precision can change near-threshold decisions, so validate it too.

## RLE and the sample submission

Flatten each binary native-size mask in **Fortran (column-major) order**. Runs use **one-based start positions** and lengths. For the toy mask `[[0,1,0],[1,1,0]]`, F-order flattening is `[0,1,1,1,0,0]`, giving RLE `2 3`. The project submission contract uses one ASCII space (`" "`) for an empty mask.

Use the **actual sample submission** for column names, image–class pairs, row order and row count. For 3,000 test images and four classes, that is 12,000 data rows plus one header line. Parse CSV rows rather than trusting a line-count shortcut. Never confuse header-inclusive lines with data records.

Check unique expected pairs, no missing/extra pairs, correct class names, native shape, binary masks, legal run bounds through pixel 102,400, exact RLE encode/decode round trips, and empty-mask handling. Avoid publishing raw competition images, credentials, checkpoints or training caches alongside the lesson.

## Discussion

A four-view, three-model ensemble has twelve forward evaluations per batch. Which candidate would you deploy if its extra latency brought no confirmed gain? What additional evidence would justify retaining it?
