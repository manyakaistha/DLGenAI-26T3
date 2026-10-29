# 4 · Gradient accumulation with the right denominator

**Core · minutes 28–39, including the second notebook handoff.** Ask: “We can fit two images at once, but want an update based on six. What needs to wait?” The optimizer step waits; backward still happens on each microbatch.

## Microbatch and effective batch

On one device, a full accumulation window has nominal effective batch **B × K**, where B is the microbatch size and K is the number of microbatches per update. This controls how many examples contribute to one update. The activation footprint is mostly governed by B, because each backward frees its graph before the next microbatch.

Accumulation does not combine GPU memory into one pool and does not reduce parameter or optimizer-state storage. It also does not guarantee better throughput: more forward/backward launches may cost time.

<div class="eff-lab" data-eff="accumulation"><p>Example: 13 images, microbatch 2, accumulation 3 → update windows 6, 6, 1.</p></div>

## The uneven-window trap

Let microbatch i contain B_i images, and let its loss be the mean per-image loss L_i. For N images in the window, backward should receive **L_i × B_i/N**. Summing those gradients gives the gradient of the mean loss across those N images.

For five images split into microbatches of 2, 2 and 1, the weights are **2/5, 2/5 and 1/5**. Dividing each microbatch mean by three overweights the final singleton. The rule `loss / K` is correct when all K microbatches have equal size; it is not a universal fix for the final window.

Our runtime first collects up to K microbatches, counts their actual images, and weights each mean accordingly. It holds the batch tensors for one window, but never all of their autograd graphs simultaneously. This is a deliberate readability tradeoff for a small teaching example.

```python
optimizer.zero_grad(set_to_none=True)
N = sum(len(images) for images, targets, ids in window)
for images, targets, ids in window:
    with autocast_context():
        loss = sample_losses(model(images), targets).mean()
        weighted = loss * len(images) / N
    scaler.scale(weighted).backward()
# unscale, clip, optimizer/scaler step, scaler update, then scheduler step
```

## Three clocks

A **microstep** processes one microbatch. An **optimizer update** changes parameters after a complete window. An **epoch** traverses the selected training examples. Log them distinctly. With 13 images, B=2 and K=3: seven microsteps, three optimizer updates, and one epoch. The last update uses one image.

Warmup and cosine scheduling in this lesson run on successful optimizer updates. Advancing the schedule on every microstep would make a three-update schedule finish too early when K>1. If FP16 skips an optimizer update, our schedule also waits.

## When physical and accumulated batches differ

For a sample-separable objective and no batch-dependent layers, accumulated gradients can closely match a physical batch, subject to floating-point order. Our notebook tests exactly that case before clipping or stepping.

BatchNorm uses statistics within each forward microbatch; accumulation does not magically give it physical-large-batch statistics. Dropout and stochastic augmentation can also change the exact comparison. Dice reduced across an entire physical batch is another objective from the per-image Dice used here. Name the reduction before claiming equivalence.

Changing effective batch may change optimization. Do not automatically scale the LR without checking the regime and validation behaviour. Keep B×K fixed when isolating precision effects.

## Handoff 2 · Compare the gradients

[Open the exact notebook instructions →](08_notebooks.md#accumulation-demo)

Run **03 · Accumulation**. It uses up to five real cloud examples, compares a physical batch with microbatches of two, and asserts gradients are close. With the explicit four-image smoke selection only three images enter training, so the check uses sizes 2 and 1. It is a code-path check, not evidence about a representative cloud population.

Before running, ask learners to write the weights for 2+2+1. After running, inspect `train_epoch`: find the gradient reset, the loss weighting and the single clipping/step location.

<details><summary>Exit check: 10 images, B=3, K=2</summary>

Microbatches are 3,3,3,1. Update windows contain 6 and 4 images. Within the final window, weights are 3/4 and 1/4. There are four forward/backward calls and two optimizer updates. Using 1/2 for both final microbatch means would overweight the singleton.

</details>
