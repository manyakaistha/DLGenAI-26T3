# Inference techniques for segmentation

**9 November 2026 · Live Session 5 · 60 minutes**

A trained network returns scores. An inference pipeline turns those scores into masks that you can validate, reproduce and submit. This session connects Milestone 5's inference topics with EMA checkpoint selection.

[Open the interactive slides](presentation/index.html) · [Start the study notes](01_pipeline.md) · [Notebook and setup](06_practical.md) · [Quick reference](07_reference.md)

## What you will learn

By the end, explain and implement the inverse-transform step in TTA, distinguish EMA parameter averaging from prediction ensembling, choose thresholds using held-out evidence, and verify the final mask-to-CSV pipeline.

## The one-hour route

| Minutes | Topic | Participation |
| --- | --- | --- |
| 00–06 | Inference contract and baseline | Identify shapes and activation |
| 06–20 | TTA and spatial inversion | Mirror-location activity + notebook A |
| 20–33 | EMA, SWA and model ensembles | Scalar EMA activity + notebook B |
| 33–45 | Thresholds and component filtering | Threshold activity + notebook C |
| 45–54 | Validation, cost and submission | Audit a proposed experiment |
| 54–60 | Exit questions and discussion | Explain one defensible next experiment |

The notebook uses **deterministic synthetic arrays and a toy predictor**, runs on CPU, and needs no checkpoint, account, dataset or GPU. Its outputs demonstrate mechanics, not measured competition performance. No private dataset findings or model results are included.

## Study material

1. [The inference contract](01_pipeline.md)
2. [Test-time augmentation](02_tta.md)
3. [EMA and prediction ensembles](03_ema_ensembles.md)
4. [Thresholds and connected components](04_decisions.md)
5. [Validation, runtime and submission](05_validation_submission.md)
6. [Runnable notebook and exercises](06_practical.md)
7. [Quick reference and reading](07_reference.md)

This is a dedicated live lesson. It does not change the milestone assessments or require students to copy a particular inference recipe.
