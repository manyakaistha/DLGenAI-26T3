# Data quality and augmentation

A model can miss defects because the camera changed, the input lost detail, the class has too few examples, or annotators drew different boundaries. Check which problem shows up in your validation errors.

## 1. Domain shift and source mix

Real-world datasets are frequently assembled from more than one batch, camera, or production line. Imagine your training set has 70% images from **Source A** and 30% from **Source B**, while your deployment or test stream is 100% **Source B**.

If you only look at your overall validation score, you are mostly measuring performance on Source A. A change that helps Source A by `+0.02` and hurts Source B by `-0.01` will look like a win overall—and then **drop** in production!

**The habit:** Always check whether your training and test distributions differ (simple image statistics, resolution, brightness, or an adversarial "train vs. test" classifier). If they do, **track validation separately on the target-like slice** and make your tuning decisions there.

## 2. Resolution and small-defect visibility

Industrial strips are wide and thin (`800 × 128`, a `6.25:1` aspect ratio). Many defects are only a few pixels tall or wide.

When you feed an image through a standard encoder with stride 32, a `4 × 4` defect shrinks to a fraction of a single feature-map cell. If you also downscale the input image to save memory, tiny defects can vanish completely before the decoder ever sees them.

**The habit:**

- Preserve the native aspect ratio—never squash `800 × 128` into a square.
- If small defects dominate your error budget, test **higher input resolution** (for example, `2×` upscaling during training/inference, or cropping native-resolution windows). Even a modest resolution boost can rescue small-defect recall that no loss tweak can fix.

## 3. Copy-paste augmentation for rare classes

In multi-class anomaly segmentation, one defect class might appear in 25% of images while a rare class appears in only 1–2%. Standard flips and brightness shifts never give the model new examples of that rare class.

**Copy-paste augmentation** takes a real defect mask (plus its pixels) from one image and pastes it onto a clean background strip, updating the target mask at the same time.

It is a great lever for rare-class recall, **but watch out for the edge-halo trap**:

- If you paste a rectangular crop with a hard straight border or a mismatched background brightness, the model can cheat by learning to detect the **paste seam** instead of the actual defect!
- **Fix**: match the local background brightness/contrast around the pasted patch, feather or tighten the mask boundary, and always validate on **real, unpasted** held-out images.

## 4. Annotation jitter and label noise

Even careful human labelers rarely agree to the exact pixel on a blurry scratch edge. A mask boundary commonly jitters by **±1 to ±2 pixels** from one annotator to another.

Two practical consequences follow:

1. **Diminishing returns on razor-sharp binary losses**: pushing a model to predict `0.999` vs `0.001` on a ±2px uncertain border forces it to overfit annotator quirks.
2. **Why ensembles + distillation help**: averaging diverse models smooths out random boundary jitter, and distilling those soft probabilities teaches the student where the boundary is genuinely fuzzy.

**Check yourself:** If your training data is a mix of two camera sources but test images come from only one, why can overall CV mislead you? And what shortcut can a model learn if copy-paste augmentation leaves a visible rectangular seam?
