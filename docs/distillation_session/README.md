# Distillation, pseudo-labeling & data-centric tricks

**23 November 2026 · Live Session 6 · 40 minutes**

Once you have a strong ensemble, two practical questions show up right away: *how do you shrink that accuracy into one fast model*, and *how do you squeeze more signal out of your data — including unlabeled images*? This 40-minute session walks through ensemble distillation, pseudo-labeling, and the data-centric habits that actually move segmentation scores.

[Open the interactive slides](presentation/index.html) · [1 · Ensemble distillation](01_distillation.md) · [2 · Pseudo-labeling](02_pseudo_labeling.md) · [3 · Data-centric tricks](03_data_tricks.md) · [4 · Calibration & safety](04_calibration_and_safety.md) · [5 · Quick reference](05_reference.md)

## What you will learn

By the end of these 40 minutes, you will be able to:

- Explain why an ensemble's **soft probability maps** teach a single student model more than hard 0/1 masks alone.
- Run a safe **pseudo-labeling** loop on unlabeled images without falling into confirmation bias or validation leakage.
- Spot **data-centric bottlenecks** — domain shift, resolution loss on tiny defects, rare-class copy-paste pitfalls, and annotation jitter.
- Use simple **probability calibration** and a 4-step safety checklist before trusting any new gain.

## The 40-minute route

| Minutes | Topic | Interactive check |
| --- | --- | --- |
| 00–03 | The ensemble dilemma | Why great ensembles are hard to ship |
| 03–14 | Knowledge distillation from ensembles | Temperature slider + animated teacher → student flow |
| 14–24 | Pseudo-labeling & self-training | Confidence-gate & confirmation-bias simulator |
| 24–33 | Data-centric tricks (shift, scale, copy-paste, jitter) | Spot the hidden failure mode |
| 33–38 | Calibration & the 4-step safety checklist | Calibrated vs. overconfident teacher toggle |
| 38–40 | Exit ticket & Q&A | Pick one defensible next step |

All examples in these notes and slides use **small, clean synthetic numbers** so the mechanics stay transparent. No private dataset counts or competition scores are assumed.

## Study guides

1. [Distilling an ensemble into one model](01_distillation.md)
2. [Pseudo-labeling and self-training](02_pseudo_labeling.md)
3. [Data-centric tricks that move the needle](03_data_tricks.md)
4. [Calibration, confirmation bias and safety](04_calibration_and_safety.md)
5. [Quick reference and further reading](05_reference.md)

This is a standalone live session guide. It complements Milestones 4 and 5 without changing any assessment requirements.
