# Optional: a small code demo

**Follow the browser notes first.** Running code is optional for this session.

The small classroom notebook shows the same ideas as the four lessons: trace encoder features, resize and concatenate a decoder skip, identify the starting weights, and choose which parameters can learn. It uses a clearly labelled dummy input tensor. You do not need to download MRI files or start a training job.

[Download the small notebook](classroom_walkthrough.ipynb) · [Python version](classroom_walkthrough.py)

## If you want to try it

Open the notebook in your existing PyTorch environment or Colab and run its cells in order. It uses PyTorch and torchvision. A GPU is not needed.

1. Trace the named encoder stages and their output sizes, then execute a decoder join: 512 + 256 channels become 768 before convolution.
2. Find the line that chooses random or ImageNet weights.
3. Freeze the encoder, then unfreeze it. Check the trainable parameter count.
4. Read the short experiment plan at the end.

The default `USE_PRETRAINED=False` lets you inspect the structure without downloading weights. Its output explicitly says this is a **random-weight shape demo**. Set it to `True` only when you want to load the actual ImageNet weights; that needs a first-time download or a prepared cache. No masks are predicted, no model is trained, and no performance score is claimed by this small demo.

## What you need to understand afterwards

Which part could be reused? Which part of a segmentation model would still need to learn? What changes when the encoder is frozen? You can answer those questions using the website even if you never run the notebook.

## Optional: the full MRI experiment

The original complete training notebook remains available for independent exploration. It sets up real TIFF data, holds out patients, and trains matched models with random and ImageNet encoder weights. This is an additional practical exercise, with more setup than the classroom demo.

[Full training notebook](transfer_learning.ipynb) · [Full Python script](transfer_learning.py)

<details>
<summary>Full experiment: setup, configuration and troubleshooting</summary>

## Option A: Kaggle

1. Create a notebook and attach [Brain MRI segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) as an input dataset.
2. Enable a GPU for the training demonstration. The code uses CUDA when available and CPU otherwise.
3. Import the supplied notebook. Run its setup cell if the required libraries are missing; restart the kernel if installation replaced an imported binary package.
4. Set `DATA_ROOT` in the configuration cell to the attached dataset path shown in Kaggle’s input panel. The familiar path is `/kaggle/input/lgg-mri-segmentation`, but do not assume every mount uses that spelling. The loader searches below the path recursively.
5. Enable internet for the **first ImageNet-weight download**, or prepare the torchvision weight cache beforehand. The code fails explicitly if the download is unavailable; it never relabels random weights as pretrained.
6. Run the cells in order and inspect the patient-overlap assertion before training.

## Option B: Colab or local files

Download and extract the dataset into your own runtime. On Colab, upload the notebook, choose a GPU runtime and set `DATA_ROOT` to the extracted folder. You may optionally download using Kaggle’s supported tooling in your own runtime after accepting any dataset access requirements. Do not paste API credentials into the teaching notebook or repository.

On this project, MRI data belongs beneath gitignored `data/public/`, separate from industrial competition images. Example:

```sh
DATA_ROOT=data/public/lgg-mri EPOCHS=2 uv run --no-project docs/transfer_learning_live_session/transfer_learning.py
```

The Python script has pinned `uv` dependencies. The notebook setup uses the same versions: torch 2.8.0, torchvision 0.23.0, NumPy 2.2.6, Pillow 12.3.0 and Matplotlib 3.10.9. Follow your hosted runtime’s compatible PyTorch installation instructions if its accelerator requires a different build, and record any changes. Restart after replacing an already imported binary library.

## Read the configuration before running

| Setting | Classroom default | Why it exists |
| :--- | :--- | :--- |
| `IMAGE_SIZE` | 128 | Faster full-field square resize; use 256 in a separate higher-resolution run |
| `EPOCHS` | 2 per arm | A short demonstration, not a converged benchmark |
| `BATCH_SIZE` | 4 | Adjust for memory, equally in both arms |
| `MAX_TRAIN_PATIENTS` | 12 | Caps training cases after the full patient split |
| `MAX_VAL_PATIENTS` | 4 | Caps validation cases after the split |
| `MAX_SLICES_PER_PATIENT` | 12 | Deterministic, evenly spaced subset; no mask-based sampling |
| `INPUT_MODE` | `sequences` | Preserve three supplied sequence channels; alternative `flair_repeat` |
| `ARMS` | `random,imagenet` | Main matched initialization comparison |
| `THRESHOLD` | 0.5 | Fixed for both arms |

Set a patient/slice cap to `0` to use all available entries. A slice cap may miss small labelled regions; inspect subgroup counts and increase the cap if necessary. Patient caps select a seeded subset rather than the first alphabetic case IDs. Splitting happens **before** applying caps, and the saved IDs show both the complete split and selected classroom cases. Neither dataset size nor completion time is guaranteed by the defaults.

For a quick setup check on CPU, use `EPOCHS=1`, `IMAGE_SIZE=64`, a small patient/slice cap and `ARMS=random`. That checks code execution only. It is **not** the transfer-learning comparison. Use both arms and more cases for the intended experiment. The script does not synthesize MRI data or fabricate performance if real files are absent.

## What each cell does

1. **Configuration and imports:** set paths, budgets and seeds; select CPU or CUDA.
2. **Index and split:** match TIFF pairs, validate naming and assign whole patients to training or validation.
3. **Dataset:** preserve the field of view, make binary targets, scale intensities and normalize image channels only.
4. **Model:** expose five ResNet feature maps, concatenate skips and produce one raw-logit channel.
5. **Sanity checks:** verify sample contracts, output shapes and the limited first-convolution weight-sum identity.
6. **Training and evaluation:** compute BCE–Dice, train with the same policy and report patient-aware metrics.
7. **Experiment runner:** reuse decoder/head initialization, train each arm, write artifacts and release model memory before the next arm.

The code is intentionally readable rather than a production training framework. The main comparison uses no random augmentation, no threshold search, no area filtering, no scheduler and no mixed precision. Those are useful future experiments; keeping them fixed or absent makes today’s result easier to interpret.

## Outputs and interpretation

Outputs go to `runs/mri_transfer_session/` (relative to the notebook’s working directory) unless `OUTPUT_DIR` changes. They include `manifest.json`, a run configuration/environment record, a data-inspection panel, per-arm epoch history, metrics, checkpoint, curves and prediction panels, plus `comparison.json`. Re-running with the same output directory replaces those teaching artifacts; use a new directory for an experiment you want to preserve.

A classroom checkpoint is a local artifact, not bundled website content. Download your outputs before ending an ephemeral Kaggle or Colab runtime. Compare the actual `comparison.json` entries; the notes do not supply expected winning scores.

## Troubleshooting

- **No masks found:** verify `DATA_ROOT` points to extracted TIFF folders, not a zip or notebook directory.
- **Unpaired or duplicate cases:** remove duplicate extraction trees; do not silently keep the first occurrence.
- **Unexpected TIFF layout/dtype:** inspect the file and dataset version. The teaching loader rejects unsupported formats instead of guessing intensity scaling.
- **Weight download failure:** enable internet or populate the correct torchvision cache. Do not switch weights to `None` while keeping the pretrained label.
- **Out of memory:** reduce batch size or image size equally for both arms, then start a fresh comparison directory.
- **No positive validation slices:** increase the slice cap or validation case sample; report the issue. Do not move patients based on one model’s score.
- **Changing a cell has no effect:** recreate the dataset/model/optimizer objects that captured the old configuration, or restart and run all cells in order.

Return to the [session overview](README.md).

</details>

[Back to the session overview](README.md)
