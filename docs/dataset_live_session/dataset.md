# 1. Dataset: build one sample

A folder is a collection of files. A PyTorch Dataset gives that collection a Python interface: ask for sample `i`, and receive one image with its labels.

## What counts as one sample?

Our CSV has columns `Image_Label` and `EncodedPixels`. An example key is `demo_001.jpg_Fish`. There are four annotation rows for the same image: Fish, Flower, Gravel and Sugar. One CSV row is an image–class pair; **one Dataset sample is an image with all four masks**.

The mask is not a single class number. It tells us which pixels belong to each cloud pattern. An empty RLE field means an all-zero mask for that channel.

## Step 1: index the annotations once

The companion notebook supplies imports and an RLE decoder. Use this function to collect four annotations under each filename:

```python
def read_index(root):
    frame = pd.read_csv(root / "train.csv", keep_default_na=False)
    parts = frame["Image_Label"].str.rsplit("_", n=1, expand=True)
    frame["filename"], frame["label"] = parts[0], parts[1]
    if not frame["label"].isin(CLASSES).all():
        raise ValueError("Unknown class")
    if frame.duplicated(["filename", "label"]).any():
        raise ValueError("Duplicate image-class pair")
    counts = frame.groupby("filename")["label"].nunique()
    if not counts.eq(len(CLASSES)).all():
        raise ValueError("Every image needs four explicit annotation rows")
    return frame.pivot(index="filename", columns="label", values="EncodedPixels").reindex(columns=CLASSES)

```

Read the function from top to bottom:

- `pd.read_csv` reads the small annotation table; `keep_default_na=False` keeps empty mask fields as empty strings.
- `rsplit("_", n=1)` splits at the last underscore, so underscores inside filenames survive.
- `isin`, `duplicated`, and the count check reject unknown classes, repeated pairs, and missing annotation rows.
- `pivot` makes filenames the rows and labels the columns. This is a table reorganization, not image processing.
- `reindex(columns=CLASSES)` fixes the channel order. Fish must not unexpectedly change from channel 0 to channel 2.

`CLASSES = ("Fish", "Flower", "Gravel", "Sugar")` is shared with every later step. The CSV stays small in memory; the image folder can stay on disk.

## Step 2: implement three methods

```python
class CloudDataset(Dataset):
    def __init__(self, root, size=(60, 90), transform=None):
        self.root = Path(root)
        self.index = read_index(self.root)
        self.size = size
        self.transform = transform

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        filename = self.index.index[i]
        with Image.open(self.root / "train_images" / filename) as source:
            image = source.convert("RGB")
        width, height = image.size
        masks = [decode_rle(self.index.iloc[i][label], height, width) for label in CLASSES]
        out_h, out_w = self.size
        image = image.resize((out_w, out_h), Image.Resampling.BILINEAR)
        masks = [np.asarray(Image.fromarray(m).resize(
            (out_w, out_h), Image.Resampling.NEAREST), dtype=np.float32) for m in masks]
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).contiguous()
        mask_tensor = torch.from_numpy(np.stack(masks))
        if self.transform is not None:
            image_tensor, mask_tensor = self.transform(image_tensor, mask_tensor)
        return {"image": image_tensor, "mask": mask_tensor, "filename": filename}

```

### `__init__`: prepare the address book

Python calls this when you write `CloudDataset(ROOT)`. `self` refers to this particular Dataset object. Store the folder, annotation index, desired output size and optional transform. `transform=None` means “leave out random augmentation.” Reading the index once avoids rereading the CSV for every sample.

### `__len__`: count the images

`len(dataset)` calls this method. We count unique filenames, so the result is 24, not 96. A mask with no foreground is still a valid sample.

### `__getitem__`: build one sample on demand

`dataset[1]` calls `__getitem__(1)`. The index is zero-based, so this requests the second image in the index.

1. Look up the filename and construct its image path using `/` on `Path` objects.
2. `with Image.open(...)` opens the file and closes its handle when the block ends. `convert("RGB")` ensures three channels.
3. Read original `width,height`; decode each RLE on that original canvas. Resizing the canvas before decoding changes pixel locations.
4. Resize the image with bilinear interpolation and every mask with nearest-neighbor. `size` is H,W here; Pillow takes W,H.
5. Convert image bytes into float32 values in `[0,1]`. This is intensity scaling. It does not calculate a mean or standard deviation.
6. `permute(2,0,1)` changes H,W,C into C,H,W. `contiguous()` gives the tensor a contiguous memory layout. `np.stack(masks)` creates the four-channel target.
7. If a transform was supplied, pass **both** tensors to it. The transform returns an image–mask pair.
8. Return a dictionary. The filename is useful metadata for debugging, not a model input.

## Step 3: inspect one sample

```python
dataset = CloudDataset(ROOT)
sample = dataset[1]
print(len(dataset))                   # 24 images
print(sample["image"].shape)          # [3, 60, 90]
print(sample["mask"].shape)           # [4, 60, 90]
print(sample["mask"].unique())        # subset of {0, 1}
```

There is **no batch dimension yet**. The three image channels are RGB; the four target channels are cloud labels. Display the image and masks before trusting the loader. A correctly shaped but horizontally reversed mask is still wrong.

## Activity: be the Dataset

Write the three return values for `dataset[5]`: filename, image shape and mask shape. Then explain which method runs when someone asks for `len(dataset)`.

<details>
<summary>Reveal the reasoning</summary>

`__getitem__` reads the sixth indexed filename, returns image `[3,60,90]` and mask `[4,60,90]`. `__len__` returns the number of indexed images without loading all of them. The exact filename depends on the index order; inspect it rather than assuming a numeric filename.

</details>

## Common mistakes

Appending `.jpg` twice, counting CSV rows, decoding at resized dimensions, and changing the class order all break the sample contract. Print the filename and shapes at the point of failure.

Continue to [the 80/20 split](split.md), then [DataLoader](dataloader.md).

Reference: [PyTorch Dataset tutorial](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html).
