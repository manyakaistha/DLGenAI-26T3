# 3. Augmentations: change the view, preserve the answer

An augmentation creates a plausible variation of a training example. A small brightness change can imitate a different exposure; a flip can present the same shape in a different orientation. These variations do not create new independently collected images, and useful settings depend on the domain.

The rule for segmentation is simple: **when pixels move, their mask labels must move with them**. When only appearance changes, leave the mask values alone.

## Lab 1: move the picture and its labels together

The original is deliberately asymmetric. Dashed outlines show the supplied label boundary. Choose a flip or translation, then deliberately leave the mask behind. This lab shows the union of all foreground labels for readability; real code must transform all four channels.

<div class="teaching-lab" data-session-lab="geometry"><p>A horizontal flip maps image and mask pixel x to W−1−x. Flipping only the image moves the cloud away from its label. Translate both by the same number of pixels; new border pixels have background labels.</p></div>

### Activity: repair a broken pair

1. Select horizontal flip and turn mask synchronization off.
2. Predict whether the image shape and mask shape checks would detect the bug.
3. Turn synchronization back on. Explain what changed even though all shapes stayed the same.

<details>
<summary>Reveal the reasoning</summary>

Shapes do not detect spatial misalignment. Both arrays still have the same dimensions. The label must undergo the exact same geometric operation and parameters as the image. The reported alignment is IoU against the known transformed synthetic shape; it is a debugging measure, not model accuracy.

</details>

## Lab 2: change appearance, keep the label

Move the brightness slider. The cloud location does not change, so the binary mask should not change. Turn on the deliberate bug to multiply the mask values as well. At factor 1 the bug is invisible; try 0.5.

<div class="teaching-lab" data-session-lab="brightness"><p>Multiplying image brightness by 0.5 changes RGB intensities, but a foreground mask value must remain 1. Multiplying that mask by 0.5 corrupts the binary target.</p></div>

### Activity: separate geometry from appearance

For each operation, decide whether to change image only or image and masks: horizontal flip, vertical flip, brightness, translation, contrast, crop. Then name one reason an operation might be unrealistic for a particular camera or subject.

<details>
<summary>Reveal the answer</summary>

Flips, translation and crop must share parameters between image and masks. Brightness and contrast change only the image. A crop can remove every foreground pixel; inspect the result rather than assuming labels remain present. A vertical flip may violate the physical orientation of a scene. Severe brightness changes can remove useful visual evidence even though the target file is unchanged.

</details>

## Lab 3: why masks use nearest-neighbor

A mask is a label map, not a photograph. Nearest-neighbor chooses a label from a source pixel; bilinear interpolation mixes neighboring values. The tiny diagonal below is enlarged to make intermediate values visible. The browser computes the numeric grid directly, rather than relying on browser image smoothing.

<div class="teaching-lab" data-session-lab="interpolation"><p>Nearest-neighbor retains binary values 0 and 1. Bilinear enlargement produces intermediate values along boundaries. Nearest-neighbor preserves the label vocabulary, but downsampling can still remove thin structures.</p></div>

### Activity: inspect a boundary value

Choose bilinear, find a gray boundary pixel, and select its grid position to read its value. Switch to nearest-neighbor. Explain why thresholding a blurred mask afterwards may move its boundary.

<details>
<summary>Reveal the reasoning</summary>

Intermediate mask values do not represent the original binary annotations. Thresholding chooses a new boundary from the mixed values. For this hard-label lesson we use nearest-neighbor. Soft targets can be a deliberate modeling choice, but are not an accidental side effect of resizing.

</details>

## Write the paired transform

Our tensors already have C,H,W axes. Width is the last axis (`-1`); height is the second-last (`-2`). Sample each flip decision once and use that decision for both tensors.

```python
def augment_pair(image, mask):
    if torch.rand(()).item() < 0.5:
        image = torch.flip(image, dims=(-1,))
        mask = torch.flip(mask, dims=(-1,))
    if torch.rand(()).item() < 0.5:
        image = torch.flip(image, dims=(-2,))
        mask = torch.flip(mask, dims=(-2,))
    brightness = 0.8 + 0.4 * torch.rand(()).item()
    image = (image * brightness).clamp(0, 1)
    return image, mask

# Separate Dataset objects keep validation free of random augmentation.
torch.manual_seed(42)
augmented_dataset = CloudDataset(ROOT, transform=augment_pair)
training = Subset(augmented_dataset, train_indices)
validation = Subset(CloudDataset(ROOT), val_indices)
augmented_loader = DataLoader(training, batch_size=4, shuffle=True, num_workers=0,
                             generator=torch.Generator().manual_seed(123))
augmented_batch = next(iter(augmented_loader))
assert set(augmented_batch["mask"].unique().tolist()) <= {0.0, 1.0}
assert torch.equal(validation[0]["image"], validation[0]["image"])
print("Augmented batch:", augmented_batch["image"].shape)

```

Read the transform line by line:

- `torch.rand(())` draws one random scalar in `[0,1)`. Comparing with 0.5 means a 50% chance each time the sample is fetched.
- A single `if` surrounds both flips, so image and masks receive the same decision. Calling two independent random transforms would not guarantee that.
- `torch.flip(..., dims=(-1,))` reverses width across every channel. `(-2,)` reverses height. The trailing comma makes a one-element tuple.
- `0.8 + 0.4 * random` samples brightness from `[0.8,1.2)`. Only the image is multiplied; `clamp(0,1)` keeps intensities in range.
- The return value is a pair; Dataset receives both tensors before returning the sample dictionary.
- `torch.manual_seed(42)` seeds the augmentation RNG for this zero-worker demo. Do not reset the seed inside `__getitem__`, or each request may receive the same random choices.
- Separate Dataset instances give training the transform and leave validation deterministic. Two Subsets of a single randomly augmented Dataset would both be augmented.

This explicit transform uses basic PyTorch so every decision is visible. Libraries can apply paired transforms for you, but image and mask types and interpolation rules must still be correct.

## Lab 4: probability is not a quota

Set p, then draw 20 fetches. Every tile is one decision to flip or leave a sample unchanged. Repeating an image does not add another file to the Dataset.

<div class="teaching-lab" data-session-lab="probability"><p>At p=0.5 each fetch has a 50% probability of a flip. A group of 20 need not contain exactly 10 flips. At p=0 none flip; at p=1 all flip.</p></div>

### Activity: same image, different views

Before drawing, predict whether p=0.5 guarantees exactly half the tiles will flip. Draw again. Set p=0, then p=1, and explain both endpoints. The lab uses a seeded sequence so the demonstration is reproducible after Reset.

## Put the three pieces together

1. Dataset opens one image, decodes labels, resizes and builds tensors.
2. Its optional training transform changes the image–mask pair together.
3. DataLoader groups the returned samples into a batch.

Random augmentation normally happens on demand. It changes views, not `len(dataset)`. Validation still needs deterministic preprocessing such as resizing and tensor conversion, but no random training augmentation in this lesson.

## Final activity: diagnose the bug

```python
if torch.rand(()) < 0.5:
    image = torch.flip(image, (-1,))
if torch.rand(()) < 0.5:
    mask = torch.flip(mask, (-1,))
```

<details>
<summary>Reveal and repair</summary>

The two independent random draws disagree half the time. Put both assignments inside a single `if`. Then visualize several returned pairs; checking shapes alone will not reveal this error.

</details>

Reference: [Torchvision transforms and paired inputs](https://docs.pytorch.org/vision/stable/transforms.html).
