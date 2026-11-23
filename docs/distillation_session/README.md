# Distillation, pseudo-labeling & data-centric tricks

**23 November 2026 · Live Session 6**

Use an ensemble to train one segmentation model, generate training targets for unlabeled images, and check how data quality affects your results.

[Open the interactive slides](presentation/index.html) · [1 · Ensemble distillation](01_distillation.md) · [2 · Pseudo-labeling](02_pseudo_labeling.md) · [3 · Data-centric tricks](03_data_tricks.md) · [4 · Calibration & safety](04_calibration_and_safety.md) · [5 · Quick reference](05_reference.md)

## What you will learn

- Explain why an ensemble's **soft probability maps** teach a single student model more than hard 0/1 masks alone.
- Run a safe **pseudo-labeling** loop on unlabeled images without falling into confirmation bias or validation leakage.
- Spot domain shift, resolution loss on small defects, visible copy-paste seams, and variation in annotated boundaries.
- Use simple **probability calibration** and a 4-step safety checklist before trusting any new gain.

The interactive examples use synthetic scores and masks.

## Study guides

1. [Distilling an ensemble into one model](01_distillation.md)
2. [Pseudo-labeling and self-training](02_pseudo_labeling.md)
3. [Data quality and augmentation](03_data_tricks.md)
4. [Calibration, confirmation bias and safety](04_calibration_and_safety.md)
5. [Quick reference and further reading](05_reference.md)
