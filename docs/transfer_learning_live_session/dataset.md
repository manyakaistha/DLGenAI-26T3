# Optional MRI application: dataset and patient split

<div class="optional-route">
<strong>Optional MRI application</strong>
<p>This page applies the encoder/decoder input contract to real MRI files. The complete pairing, loading and split explanations are preserved below.</p>
<a href="transfer.md">Continue with pretraining and fair comparisons →</a>
</div>

Our model receives one two-dimensional MRI slice and predicts one binary mask. A patient contributes several slices. **The sample we load and the unit we split are different:** load slices; split patients.

## Read the dataset’s own description first

The [Kaggle dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) describes brain MR images and manually annotated FLAIR-abnormality masks from TCIA / TCGA-LGG, corresponding to 110 patients. Count the folders and valid pairs in your downloaded version rather than hard-coding that number into a loader.

The archive’s `kaggle_3m/README.md` specifies three image channels in this order:

| Array index | MRI sequence | Meaning for this exercise |
| :--- | :--- | :--- |
| `image[...,0]` | Pre-contrast | First supplied sequence, not “red” |
| `image[...,1]` | FLAIR | Sequence available in every case |
| `image[...,2]` | Post-contrast | Third supplied sequence, not “blue” |

FLAIR stands for fluid-attenuated inversion recovery, the name of an MRI acquisition sequence. You do not need its imaging physics to follow the tensor contracts in this lesson.

Missing sequences are replaced with FLAIR so the stored image still has three channels. Three array slots therefore do not guarantee three distinct measurements. Keep the order fixed. Display each sequence with a grayscale colormap rather than interpreting the three-sequence TIFF as a natural-color photograph.

![An annotated real MRI slice showing the three supplied sequences and its binary abnormality mask](figures/mri_sequences.png)

This selected teaching illustration is derived from the named Kaggle dataset; source case/slice, processing and attribution are recorded in the caption and [figure provenance](figures/provenance.json). The browser simulations elsewhere use synthetic arrays.

## Match each image to its mask

Typical extracted paths look like this:

```text
kaggle_3m/
  TCGA_CS_4941_19960909/
    TCGA_CS_4941_19960909_10.tif
    TCGA_CS_4941_19960909_10_mask.tif
```

The case directory is our grouping key. The final number identifies a slice, not a patient. Match paths by the `_mask` suffix; do not independently sort two lists and assume their entries align.

```python
records = []
for mask_path in sorted(root.rglob("*_mask.tif")):
    image_path = mask_path.with_name(mask_path.stem[:-5] + ".tif")
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    records.append({
        "image": image_path,
        "mask": mask_path,
        "patient": mask_path.parent.name,
    })
```

`stem[:-5]` removes the five characters `_mask`. Recursive discovery supports a Kaggle mount containing an extra `kaggle_3m/` directory. The companion notebook additionally checks naming consistency, duplicates, unpaired images, shapes and mask values. It fails visibly if the input is malformed.

## Inspect before transforming

For the real files inspected while preparing these notes, images were unsigned 8-bit arrays `[256,256,3]` and masks were `[256,256]` with values in `{0,255}`. These are observed examples, not a replacement for validation of your own files.

```python
image = np.asarray(Image.open(image_path))
raw_mask = np.asarray(Image.open(mask_path))
assert image.ndim == 3 and image.shape[2] == 3
assert raw_mask.shape == image.shape[:2]
mask = (raw_mask > 0).astype(np.float32)
```

A mask containing only 0 is valid: the selected slice contains no annotated foreground. Preserve those slices. Do not use `convert("L")` on the three-sequence image: that forms an arbitrary color-weighted mixture, not the FLAIR sequence. Select index 1 explicitly when you need FLAIR alone.

After preprocessing, use:

- Image: float32 `[3,H,W]`, with documented intensity scaling and normalization.
- Target: float32 `[1,H,W]`, values 0 or 1.
- Batch: image `[B,3,H,W]`, target `[B,1,H,W]`.
- Model output: raw logits `[B,1,H,W]`; sigmoid is applied inside the Dice term and for evaluation, not before `BCEWithLogitsLoss`.

## Why a random slice split leaks information

Adjacent slices share anatomy, scanner characteristics and acquisition conditions. If one patient occurs on both sides, validation is partly testing familiarity with that patient. It does not answer our intended question: “How does this model behave on a patient unseen during training?”

First permute unique case IDs with a seed; then assign every slice according to its case. The author’s [reference loader](https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/dataset.py) also selects validation cases at the patient level.

```python
patients = sorted({r["patient"] for r in records})
rng = np.random.default_rng(42)
rng.shuffle(patients)
n_val = max(1, int(np.ceil(0.2 * len(patients))))
val_ids = set(patients[:n_val])
train_ids = set(patients[n_val:])
train = [r for r in records if r["patient"] in train_ids]
valid = [r for r in records if r["patient"] in val_ids]
assert train_ids.isdisjoint(val_ids)
assert train_ids | val_ids == set(patients)
```

Require at least two patients before this split. “80/20” refers to case counts; it need not produce exactly 80% of slices because patients contribute different numbers of slices. Save the exact IDs and reuse them in both experiment arms. A future site-held-out experiment would answer a different, harder question; patient separation alone does not prove robustness to new hospitals.

<div class="mri-lab" data-lab="mri" data-ai-help="off" data-mri-lab="split">
<p><strong>Patient split lab.</strong> Four fictional patients have four slices each. With slice splitting, the first three slices of every patient go to training and the fourth to validation: all four patients leak. With patient splitting, three whole patients train and one validates: zero patients overlap. Change the patient count and compare.</p>
</div>

## Keep preprocessing on the correct side

Random paired spatial transforms belong to training only. Validation preprocessing is deterministic. Any statistics learned across the dataset must use training patients only. The notebook uses fixed scaling and fixed channel-normalization constants, so no validation-fitted statistics are required.

The teaching loader retains the full square field of view and resizes image and mask together. It does not use the mask to crop around the abnormality. A target-guided crop at validation would give the model information it would not have at inference.

## Activity: predict the leak

You have 20 slices from patient A and 30 from patient B. You reserve every fifth slice for validation. Have you held out an unseen patient?

<details>
<summary>Reveal the reasoning</summary>

No. Both patients appear in both partitions. The slice counts may look like an 80/20 split, but the patient overlap is two. Hold out all slices of a patient instead. With only two patients this would be a very unstable estimate; use more cases for meaningful evaluation.

</details>

Next: [trace the encoder and decoder](encoder.md).


[Finish this optional application and continue →](transfer.md)
