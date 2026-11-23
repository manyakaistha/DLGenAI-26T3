# Calibration, confirmation bias & when to stop

Calibration measures how often predictions at a given confidence level are correct.

## Why calibration matters

A well-calibrated model's `0.70` means "pixels with this score are true defects about 70% of the time." Modern deep networks—especially ones trained with Dice loss or heavy augmentation—are often **overconfident**: they output `0.95` when the true accuracy is `0.75`, or one model in an ensemble is much sharper than another.

When probabilities are miscalibrated:

- **Ensembling breaks**: an overconfident weak model can drown out a well-calibrated strong model.
- **Distillation breaks**: the teacher's soft targets look like hard 0/1 masks, erasing the dark knowledge.
- **Pseudo-labeling breaks**: a confidence gate at `0.90` lets through plenty of wrong predictions.

## Simple fixes: temperature & Platt scaling

You do not need a complex post-processor to calibrate a segmentation model. On a **held-out validation set**, you can fit just one or two scalars per class on the raw logits $z$:

- **Temperature scaling** (1 parameter per class):
  $$p_{\text{cal}} = \sigma\!\left(\frac{z}{T}\right)$$
- **Platt scaling** (2 parameters per class—slope $a$ and bias $b$):
  $$p_{\text{cal}} = \sigma(a \cdot z + b)$$

With positive $T$ and $a$, both methods preserve pixel ranking within a class. Fit the parameters on out-of-fold predictions and measure calibration on separate data.

## Guarding against confirmation bias

When you combine pseudo-labeling with multiple rounds (Gen 1 → Gen 2 → Gen 3), errors can quietly compound. Watch for these warning signs:

- **Validation score plateaus or drops** while training loss keeps falling.
- **Empty-image false positives creep up**: the model starts hallucinating faint defects on clean metal.
- **Rare-class recall collapses**: if the confidence gate is too strict, rare defects never get pseudo-labeled, making the class imbalance even worse!

## A 4-step safety checklist

Before you trust any distillation, pseudo-labeling, or data-centric change:

1. **Keep validation fixed**: same held-out folds, never touched by teachers or pseudo-labels, evaluated on the target-like slice if domain shift exists.
2. **Check calibration first**: inspect whether teacher probabilities are overconfident before blending or distilling.
3. **Change one thing at a time**: test distillation *or* pseudo-labeling *or* copy-paste separately so you know what actually helped.
4. **Check both sides of the metric**: track positive-mask Dice *and* empty-image false positives (and per-class recall for rare defects).

If a fancier loop does not beat the simpler baseline on your locked validation set, **keep the simpler recipe**.

**Check yourself:** Why can an overconfident teacher hurt both ensembling and distillation? And how can a strict confidence gate accidentally hurt a rare defect class?
