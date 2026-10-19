# 2. DataLoader: build a batch

Dataset prepares one sample. DataLoader chooses sample indices, asks the Dataset for those samples, and combines them into a batch. An ordinary epoch is one traversal through the chosen samples.

## Step 1: keep images in separate groups

First read [the 80/20 split lesson](split.md) for why we need a validation group and how each line of the splitting code works.

Our notebook prepares 19 training image indices and five validation indices. The sets do not overlap. Splitting four annotation rows independently could place the same photograph in both groups; always split by image identity here.

```python
order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(42)).tolist()
cut = int(0.8 * len(order))
train_indices, val_indices = order[:cut], order[cut:]
assert set(train_indices).isdisjoint(val_indices)
train_data = Subset(dataset, train_indices)
val_data = Subset(dataset, val_indices)
train_loader = DataLoader(
    train_data, batch_size=4, shuffle=True, num_workers=0,
    drop_last=False, generator=torch.Generator().manual_seed(123),
)
val_loader = DataLoader(val_data, batch_size=4, shuffle=False, num_workers=0)
batch = next(iter(train_loader))
print(batch["image"].shape, batch["mask"].shape)
print(batch["filename"])
print("Training images:", len(train_data), "Training batches:", len(train_loader))
print("Validation images:", len(val_data), "Validation batches:", len(val_loader))


```

## Read each argument

- `Subset(dataset, indices)` provides a view of selected image indices; it does not duplicate image files.
- `batch_size=4` requests up to four samples at a time.
- `shuffle=True` rearranges sample order at each training traversal. It does not flip or change any image.
- `generator=...manual_seed(123)` makes the ordinary shuffle sequence repeatable across fresh runs with the same setup. Reusing the same loader advances its random state.
- `num_workers=0` fetches samples in this Python process. This is the easiest setting for notebook debugging.
- `drop_last=False` retains the smaller final batch. Validation should normally keep every sample.
- Validation uses `shuffle=False` so its order stays predictable.

The split generator and loader generator have separate jobs. The tiny random split is a teaching example; audit label support before using a real holdout with rare labels.

## What does collation mean?

Each sample dictionary contains image `[3,H,W]`, mask `[4,H,W]`, and a filename. The default collator stacks tensors with matching keys. A batch has image `[B,3,H,W]`, mask `[B,4,H,W]`, and B filenames.

The two 4s in `[4,4,60,90]` have different meanings: the first is the number of images, the second the number of target channels. A batch size of 2 changes only the first dimension.

`iter(loader)` creates an iterator; `next(...)` requests its next batch. A `for batch in loader` loop continues until the epoch is finished.

## Interactive batch builder

Predict the final batch size, then change the controls. These numbered tiles represent filenames, not individual masks.

<div class="teaching-lab" data-session-lab="batches"><p>19 samples at batch size 4 produce five batches: 4,4,4,4,3. Dropping the last batch keeps 16 samples.</p></div>

## Activity: count before running

For 19 images and batch size 6, predict the number of batches with `drop_last=False`. Then switch on `drop_last` and count how many images are skipped.

<details>
<summary>Reveal the answer</summary>

Four batches: 6,6,6,1. With `drop_last=True`, three batches contain 18 images and one image is skipped. The Dataset still contains 19 images.

</details>

## Step 2: inspect an epoch

```python
seen = []
for batch in train_loader:
    print(batch["image"].shape)
    seen.extend(batch["filename"])
assert len(seen) == len(train_data)
assert len(set(seen)) == len(train_data)
```

These checks apply to this ordinary loader without replacement sampling. DataLoader does not calculate a loss, train a model, or automatically move tensors to a GPU.

## When to add workers

Try more workers only after correct loading works. Workers can fetch samples in parallel, but each consumes memory. `num_workers=2` does not mean batch size 2. Notebook-defined Dataset classes may fail under spawn multiprocessing; use an importable module and a main guard for scripts. Start with zero workers to expose useful errors.

Pinned memory can help CPU-to-CUDA transfers, but it is optional and does not itself move data. Fixed output dimensions are required by default stacking; different sizes require consistent resizing or an explicit padding strategy.

Continue to [augmentations](augmentations.md).

Reference: [PyTorch data loading tutorial](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html).
