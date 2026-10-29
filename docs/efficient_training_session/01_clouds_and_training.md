# 1 · Keep the cloud task fixed

**Core · minutes 00–06.** Ask: “If a training run becomes twice as fast, what evidence would convince you it is still learning the same task?” Collect three answers before naming precision, effective batch and validation protocol.

## Reuse the dataset, inspect the contract again

This session uses **Understanding Clouds from Satellite Images**, the same cloud-pattern dataset used for the bootcamp. The categories Fish, Flower, Gravel and Sugar describe cloud patterns. They are not literal objects and they are not the industrial defect classes used elsewhere in the course.

The CSV has `Image_Label` and `EncodedPixels`. Split the label at its final underscore to obtain the image filename and class. Each image needs four annotation rows, including rows with an empty RLE string. An empty class mask is valid; a missing annotation row is an integrity error.

| Variant | Native image and mask grid | Required pairing |
|---|---|---|
| `resized_384x576` (default) | H384 × W576 | Mirror images **and mirror re-encoded CSV** |
| `original` | H1400 × W2100 | Original images **and original CSV** |

Cloud RLE uses one-based starts and column-major traversal. Decode on the selected native grid, then resize the binary mask with nearest-neighbour interpolation. Resize RGB with bilinear interpolation. Our teaching tensors are H64 × W96, preserving the cloud 3:2 width:height ratio. This is different from the industrial images' 800×128 geometry.

The original competition evaluates submissions at H350 × W525. Our diagnostic values at H64 × W96 are **not official competition scores**. Sources: [original cloud data](https://www.kaggle.com/competitions/understanding_cloud_organization/data), [384×576 mirror](https://www.kaggle.com/datasets/phunghieu/understanding-clouds-from-satellite-images-384x576).

## A model small enough to understand

The notebook uses two downsampling/feature convolutions, an interpolation back to input size and a four-channel output. Parameters start in FP32. The output is raw logits; sigmoid is applied only when probabilities or thresholded predictions are needed. BCEWithLogits takes raw logits and floating binary targets. Soft Dice is calculated per image and class, then combined with BCE.

```python
x.shape       # [B, 3, 64, 96]
y.shape       # [B, 4, 64, 96], float 0/1
logits = model(x)
loss = sample_losses(logits, y).mean()
```

Why no pretrained encoder? It removes network downloads and keeps the lesson focused on the loop you can also use around a fine-tuned model. Why no BatchNorm or dropout? They would complicate our physical-batch versus microbatch comparison. This teaching choice is not a recommendation to remove them from every model.

## Fix the experiment before optimizing it

`prepare` assigns complete image IDs to disjoint train and validation partitions before applying teaching caps. By default we use a seeded image split and cap training at 32 images and validation at 16. A cap of zero means uncapped. These small selections make demonstrations quick; they do not establish reliable cloud generalization.

To reuse an existing bootcamp image split, set `CLOUD_SPLIT_MANIFEST` to its `session1_evidence.json`. Its declared variant must match, and the selected files must exist. The runtime records the selected IDs, CSV fingerprint, variant, resize, seed and scope. No masks from the held-out images enter training.

Neighbouring satellite scenes may share structure. An image split is the teaching contract here; source-group isolation would need provenance and a separate audit for a stronger generalization claim.

## Predict, then inspect

Which change is a fair precision comparison?

1. Use a different random initialization for each precision policy.
2. Keep the same batch, seed, updates and input while changing the execution policy.
3. Double the batch and learning rate while enabling AMP.

<details><summary>Discuss the answer</summary>

Option 2 isolates more of the precision effect. It still needs warmup, synchronization, repeated measurements and quality checks. Option 3 may be a useful later experiment, but it cannot attribute its result to AMP alone. Even a fair tiny-model benchmark may be dominated by Python, kernels or transfers.

</details>

## Name the bottleneck

A run can be limited by data decoding, CPU transforms, transfer, compute or memory. AMP primarily changes eligible operation dtypes; it does not automatically make JPEG decoding faster. Accumulation reduces the activation footprint needed for a larger effective batch, but it does not shrink parameters or optimizer state. A checkpoint recovers progress; it is not a performance optimization.

**Exit check:** explain why mismatching the mirror CSV with original images invalidates the lesson even if training produces a decreasing loss.
