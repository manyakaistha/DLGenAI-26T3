# Pseudo-labeling and self-training

## Turning unlabeled images into training signal

In real projects, collecting raw images is cheap, while drawing pixel-accurate masks is slow and expensive. You often end up with:

- A **labeled** training set (images + ground-truth masks).
- An **unlabeled** pool (such as extra factory strips from the deployment camera).

**Pseudo-labeling** (also called **self-training**) uses your best current model or ensemble—the *teacher*—to predict masks on those unlabeled images, then trains a new *student* model on **labeled + pseudo-labeled** data together.

When it works, pseudo-labeling does two things at once:

1. **More data**: the student sees more surface textures, lighting variations, and defect shapes.
2. **Domain adaptation**: if the unlabeled images come from the target test distribution, the student adapts to that exact camera, contrast, and background style.

## The self-training loop (Noisy Student intuition)

A Noisy Student training round has four steps.

1. **Train a strong teacher** (ideally a diverse ensemble) on the labeled training set.
2. **Predict on the unlabeled pool** using your best inference setup (e.g., ensemble + TTA).
3. **Filter the predictions** by confidence or keep them as soft targets.
4. **Train a fresh student** on the combined dataset—using **stronger data augmentation** (flips, brightness/contrast jitter, random crops, dropout) on the student than on the teacher.

Why add noise/augmentation to the student? Because the teacher predicted on a *clean* image (with TTA), while the student has to match that same mask on a *harder, augmented* version of the image. Apply each geometric transform to both the image and the target mask.

## Hard vs. soft pseudo-labels

Once the teacher outputs a probability map $p_t \in [0, 1]$ on an unlabeled image, you have two choices:

| Choice | How it works | Strength | Risk |
| --- | --- | --- | --- |
| **Hard pseudo-labels** | Threshold $p_t \ge \tau$ into `0`/`1` masks | Simple, works with any standard loss | Locks in borderline mistakes as 100% certain truth |
| **Soft pseudo-labels** | Keep continuous $p_t$ (or temperature-scaled $p_t$) | Preserves teacher uncertainty at edges | Slightly larger storage; needs soft-target loss |

In segmentation, **soft pseudo-labels** are usually safer than hard 0/1 masks—especially around fuzzy defect borders where even the teacher is only `0.55` sure.

## Confidence gating and confirmation bias

The biggest danger in pseudo-labeling is **confirmation bias**: if the teacher hallucinates a false positive on an unlabeled image and you train the student on it, the student becomes *more* confident in that exact mistake.

Two simple guards keep confirmation bias in check:

- **Confidence filtering**: only keep unlabeled images (or pixels) where the teacher is clearly confident—for example, high max probability for positive masks, or very low max probability for clean masks—and ignore the ambiguous middle band.
- **Controlled mixing ratio**: do not let pseudo-labeled images overwhelm real human labels. Start with a modest mix (e.g., 20–30% of each batch or a lower loss weight like `0.5×` on pseudo-labeled samples) and verify on your held-out validation set.

## Keep validation separate

- **Never** include validation images in the unlabeled pseudo-labeling pool.
- **Never** use a model trained on fold $k$'s validation images to generate pseudo-labels for fold $k$.
- Always compare `baseline student` vs. `pseudo-labeled student` on the **exact same untouched validation split**.

**Check yourself:** Why is it helpful to apply *stronger* augmentations to the student than the teacher during self-training? What happens if you set the confidence threshold too low?
