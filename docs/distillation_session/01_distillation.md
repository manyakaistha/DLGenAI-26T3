# Distilling an ensemble into one model

## The ensemble dilemma

Averaging probability maps can improve segmentation when the models make different errors.

The catch is **cost**. If you combine 5 models with 4-view test-time augmentation (TTA), every single image requires **20 forward passes**. That is fine for an offline batch job, but painful if you have a tight latency budget, a small GPU, or a live factory camera.

**Knowledge distillation** gives you a way out: keep the ensemble as a **teacher** during training, and train one fast **student** model to imitate the teacher's output maps. Deploy the student on its own.

## Why soft targets beat hard 0/1 masks

A human-annotated segmentation mask is binary: every pixel is either `0` (background) or `1` (defect). That throws away nuance.

An ensemble's predicted probability map is **soft**:

- The solid center of a scratch might be `0.94`.
- The blurry edge of the scratch might be `0.62`.
- A faint smudge that *almost* looks like a defect might be `0.18`.
- Clean metal far away is `0.01`.

Geoffrey Hinton called this extra signal **dark knowledge**. Hard labels treat a clear defect and a borderline edge pixel identically (`1` vs `1`), and they treat a tricky smudge and clean steel identically (`0` vs `0`). Soft targets tell the student *where* the boundary is uncertain and *which* background regions are hard negatives.

## Temperature scaling

When a teacher is very confident, its probabilities can look almost binary anyway (`0.999` vs `0.001`). Dividing the logits $z$ by a **temperature** $T > 1$ before applying sigmoid or softmax softens the distribution:

$$
p_T = \sigma\!\left(\frac{z}{T}\right)
$$

- At $T = 1$, you get the standard probability.
- At $T = 2$ or $T = 3$, extreme logits relax toward `0.5`, making subtle differences between `0.02` and `0.15` much easier for the student's gradients to see.
- At inference time, the student goes back to $T = 1$.

## The distillation loss

You rarely want the student to ignore the ground-truth labels completely—after all, the teacher can still make mistakes. Instead, blend the **hard ground-truth loss** with the **soft teacher loss**:

$$
\mathcal{L}_{\text{total}} = \alpha \, \mathcal{L}_{\text{hard}}(y, p_s) + (1 - \alpha) \, \mathcal{L}_{\text{soft}}(p_t, p_s)
$$

For pixel-wise segmentation:

- **Hard loss**: your usual BCE + Soft Dice loss against the true mask $y$.
- **Soft loss**: binary cross-entropy (or KL divergence) between the teacher's soft probability map $p_t$ and the student's predicted probability map $p_s$.
- **Balance $\alpha$**: a common starting point is $\alpha \in [0.3, 0.5]$ so the student respects true labels while absorbing the ensemble's smoother boundaries.

```python
# Simple pixel-wise distillation step (independent sigmoid channels)
with torch.no_grad():
    teacher_probs = torch.sigmoid(teacher_logits / T)

student_probs_T = torch.sigmoid(student_logits / T)
loss_hard = bce_dice_loss(student_logits, true_mask)
loss_soft = F.binary_cross_entropy(student_probs_T, teacher_probs) * (T * T)
loss = alpha * loss_hard + (1.0 - alpha) * loss_soft
```

Notice the $T^2$ factor on the soft loss: it helps compensate for smaller soft-loss gradients at higher temperatures.

## Out-of-fold teacher predictions

If you average all five teachers to label a training image, some teachers will have trained on that image. An out-of-fold target uses only teachers that held the image out.

For a training image in fold $k$, generate its out-of-fold target with a teacher that held out fold $k$. When evaluating a student, also keep its validation fold out of the teachers’ training data.

**Check yourself:** If you distill a 4-model ensemble into a single student, how many forward passes does the student need at inference time? What is the main thing the student learns from soft maps that binary masks hide?
