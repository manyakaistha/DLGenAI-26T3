# Dataset, DataLoader & Augmentations

Follow one picture from a file on disk to an augmented training batch. These notes use an original synthetic cloud dataset, with four RLE mask labels per image. No GPU or Kaggle account is needed for the live activities.

## Follow the pipeline

1. **[Dataset: build one sample](dataset.md)** — read a filename, find its four labels, and return correctly shaped tensors.
2. **[The 80/20 split](split.md)** — understand training versus validation, prevent leakage, and split image indices step by step.
3. **[DataLoader: build a batch](dataloader.md)** — group samples, shuffle their order, and understand the final batch.
4. **[Augmentations: make useful variations](augmentations.md)** — change appearance safely while keeping masks aligned. Explore four interactive labs and solve the prediction activities.

## Run the companion notebook

[Download notebook](dataset_tutorial.ipynb) · [Download Python script](dataset_tutorial.py)

Upload the notebook to Colab, select a CPU runtime, and run the cells in order. The first setup cell installs dependencies. The prepared helper cells create the independent dataset and decode RLE; the teaching focus is the topics above.

Local command from the repository root:

```sh
uv run --no-project --with numpy --with pandas --with pillow --with torch --with matplotlib python docs/dataset_live_session/dataset_tutorial.py
```

The generated data lives under `runs/cloud_loader_demo/`. It contains `train_images/`, `test_images/`, `train.csv`, and `sample_submission.csv`. There are 24 training images, four test images, and 96 training annotation rows. Images are RGB, 60 high × 90 wide; masks have four channels.

## Before you begin

You need Python functions, lists, dictionaries, and basic array indexing. A tensor is a multidimensional array. A binary mask uses 1 for a labelled pixel and 0 for its absence. Four masks let different labels coexist in one image.

The browser labs work without Python. Every lab uses small deterministic synthetic arrays and has text explanations if scripts are unavailable. Change a setting only after predicting what it will do.

Created by manaykaistha.
