# Test-time augmentation

## Several views of one image

Training augmentation changes the examples used to learn weights. Test-time augmentation (TTA) keeps the weights fixed and evaluates several transformed versions of the same image. The intended benefit is reduced sensitivity to transformations that should preserve the label semantics. Improvement is a hypothesis to test.

For input transform $T_k$, model $f$, activation $a$, and inverse spatial transform $T_k^{-1}$:

$$p_{\mathrm{TTA}}(x)=\frac{1}{K}\sum_{k=1}^{K}T_k^{-1}\left(a(f(T_k(x)))\right).$$

The inverse operates on the **prediction map**. Classification usually has a single class vector, but segmentation attaches a prediction to every coordinate.

## The mirror-location example

In a width-8 teaching grid, a defect at column 1 moves to column 6 under a horizontal flip, using zero-based indexing. A flipped-view prediction at column 6 must return to column 1 before averaging. Otherwise the average can create a ghost defect at the mirrored position and weaken the true one.

Horizontal flip on `[B,C,H,W]` uses dimension `-1`; vertical flip uses `-2`. Channel order does not reverse. Flipping twice restores the original map.

```python
@torch.inference_mode()
def predict_tta(model, x):
    model.eval()
    maps = []
    for dims in ((), (-1,), (-2,), (-2, -1)):
        view = x.flip(dims) if dims else x
        p = model(view).float().sigmoid()
        maps.append(p.flip(dims) if dims else p)
    return torch.stack(maps).mean(0)
```

This teaching code stacks four maps for clarity. A production implementation can add one map at a time to an FP32 accumulator. It assumes four independent sigmoid outputs and no spatial resizing inside the model.

## Which transformations are defensible?

Identity plus horizontal flip gives two views. Identity, horizontal, vertical and both flips gives four. Validate whether vertical orientation has physical meaning for your images. Avoid arbitrary 90-degree rotations on a long rectangular strip: they swap axes and alter the input geometry. Scale TTA and tiled inference require additional interpolation, overlap and boundary checks and are optional depth.

A model that is already perfectly flip-equivariant produces identical aligned views, so flip TTA adds no information. The notebook intentionally demonstrates that case before introducing a toy positional bias. Four views cost four forward evaluations per model, although measured latency is not guaranteed to scale exactly fourfold.

## Probabilities versus logits

These are different estimators:

$$\frac{1}{K}\sum_k\sigma(z_k)\ne\sigma\left(\frac{1}{K}\sum_k z_k\right).$$

For logits `[-4, -4, 6]`, mean probability is about **0.3445**, whereas sigmoid of the mean logit is about **0.3392**. Threshold choices can make even small differences matter. Probability averaging bounds each view's contribution. Neither estimator has a universal quality advantage. Choose the rule on held-out data and retune thresholds for that rule. Do not average thresholded masks unless you deliberately want hard voting.

## Failure checks

Use an asymmetric array with an off-center marker, verify exact inversion for every transform, and test that a perfectly equivariant predictor reproduces the identity output. Check width and height independently. A symmetric centered blob can hide a broken inverse.

**Notebook A:** run the TTA cells, inspect the mirrored coordinate and explain why the intentionally un-inverted average fails.
