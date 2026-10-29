# Notebook setup and exact handoffs

Two clean notebooks contain the full editable runtime. The core notebook supports the hour; the DDP notebook is an optional separate workshop. Dataset download and runtime preparation happen **before** class.

## Obtain the real cloud data

Reuse your existing bootcamp download. If you need it again, obtain the [384×576 mirror](https://www.kaggle.com/datasets/phunghieu/understanding-clouds-from-satellite-images-384x576) under its applicable terms. Original files are available through the [competition](https://www.kaggle.com/competitions/understanding_cloud_organization/data). Do not upload private API credentials into shared notebooks.

Expected layout, for one matching variant:

```text
clouds_384x576/
  train.csv
  train_images/
    <real JPEG image files>
```

The default variant is `resized_384x576`. For original images explicitly set `CLOUD_DATA_VARIANT=original` and point at the original CSV/images. The runtime checks native dimensions, all four annotation rows, RLE bounds, duplicates and selected files. An incomplete download raises an error; it never invents images.

## Setup once in the core notebook

[Download cloud_training.ipynb](cloud_training.ipynb) · [Open core notebook in Colab](https://colab.research.google.com/github/manyakaistha/DLGenAI-26T3/blob/main/docs/efficient_training_session/cloud_training.ipynb#scrollTo=setup)

Run **00 · Setup**: install dependencies if needed, write/import the embedded runtime, configure data paths, and run the data-contract cell. In Colab you must first copy or mount the actual dataset into the runtime. A link cannot supply your local files automatically. Keep the configured notebook tab open when returning from slides.

Before the configuration cell, set paths in Python, replacing the example with your real runtime location:

```python
import os
os.environ['CLOUD_DATA_ROOT'] = '/content/clouds_384x576'
os.environ['CLOUD_DATA_VARIANT'] = 'resized_384x576'
os.environ['CLOUD_OUTPUT_ROOT'] = '/content/cloud_efficiency'
# Optional: reuse the bootcamp split, if this file is available here.
# os.environ['CLOUD_SPLIT_MANIFEST'] = '/content/session1_evidence.json'
```

Run cells in order: installation if needed → embedded runtime → data configuration. If replacing imported binary dependencies, restart before re-importing. Pins used by the editable runtime: PyTorch 2.8.0, NumPy 2.2.6 and Pillow 12.3.0. A hosted runtime may ship a compatible newer stack; report the version actually used and recheck its behaviour. CPU FP32 is sufficient for every core correctness demonstration.

The source defaults are small: B=2, accumulation=2, three epochs, training cap 32, validation cap 16, H64×W96. Edit `cfg` in the notebook to change these. Environment variables configure paths; the notebook's explicit `cfg` controls its displayed lesson settings.

## AMP demo

<a id="amp-demo"></a>
[Open section 01 · Precision](https://colab.research.google.com/github/manyakaistha/DLGenAI-26T3/blob/main/docs/efficient_training_session/cloud_training.ipynb#scrollTo=precision-demo) · [Open section 02 · AMP](https://colab.research.google.com/github/manyakaistha/DLGenAI-26T3/blob/main/docs/efficient_training_session/cloud_training.ipynb#scrollTo=amp-demo)

**Slide handoff 1 · minutes 22–28.** Setup must already be complete. Run the number-cast cell under **01**, then the source inspection/benchmark cell under **02**. Predict the FP16 overflow and the CPU skip before executing. Read the actual device, finite-loss flag, measured steps and peak-memory definition. CPU results demonstrate execution; they do not establish CUDA speedup. Return to the slide immediately after the benchmark discussion.

## Accumulation demo

<a id="accumulation-demo"></a>
[Open section 03 · Accumulation](https://colab.research.google.com/github/manyakaistha/DLGenAI-26T3/blob/main/docs/efficient_training_session/cloud_training.ipynb#scrollTo=accumulation-demo)

**Slide handoff 2 · minutes 34–39.** Run its one code cell. It compares gradients before clipping/update for the same real examples in a physical batch and in microbatches. Expected: allclose assertion passes with a small floating-point difference. Read the actual microbatch sizes; a tiny explicit smoke run can use 2+1 instead of 2+2+1. Find the B_i/N weighting in `train_epoch`. The no-BatchNorm/no-dropout model and per-image loss are essential to interpreting the comparison.

## Resume demo

<a id="resume-demo"></a>
[Open section 04 · Resume](https://colab.research.google.com/github/manyakaistha/DLGenAI-26T3/blob/main/docs/efficient_training_session/cloud_training.ipynb#scrollTo=resume-demo) · [Open section 05 · Validation](https://colab.research.google.com/github/manyakaistha/DLGenAI-26T3/blob/main/docs/efficient_training_session/cloud_training.ipynb#scrollTo=validation-demo)

**Slide handoff 3 · minutes 45–51.** Run the interrupt/resume cell, then the schedule/validation cell. It trains the tiny selected real dataset for one epoch, resumes to the fixed three-epoch plan, and runs an uninterrupted reference. Expected on the CPU route: final weights are close within the stated tolerance. Inspect checkpoint keys and next epoch, then contrast last.pt with best.pt. Pre-run this before class to estimate duration on your machine. If time is tight, discuss the saved local result from your rehearsal with its scope stated; do not present illustrative output as freshly measured.

## DDP demo

<a id="ddp-demo"></a>
[Download cloud_ddp.ipynb](cloud_ddp.ipynb) · [Open D0 · Setup](https://colab.research.google.com/github/manyakaistha/DLGenAI-26T3/blob/main/docs/efficient_training_session/cloud_ddp.ipynb#scrollTo=ddp-setup) · [Open D1 · Shards](https://colab.research.google.com/github/manyakaistha/DLGenAI-26T3/blob/main/docs/efficient_training_session/cloud_ddp.ipynb#scrollTo=ddp-shards) · [Open D2 · Launch](https://colab.research.google.com/github/manyakaistha/DLGenAI-26T3/blob/main/docs/efficient_training_session/cloud_ddp.ipynb#scrollTo=ddp-launch) · [Open D3 · Resume](https://colab.research.google.com/github/manyakaistha/DLGenAI-26T3/blob/main/docs/efficient_training_session/cloud_ddp.ipynb#scrollTo=ddp-resume)

**Optional, outside the hour.** Read the [DDP guide](07_ddp_optional.md). Run D0 first; it writes both Python files and configures the same real cloud data. Default is two CPU/Gloo processes, B=1, accumulation=2, FP32. D1 prints actual sampler shards. D2 launches one epoch through an external script. D3 resumes and verifies replica agreement and unique validation counts. Change `DDP_DEMO_DEVICE` to `cuda` only with at least two visible compatible GPUs; the worker chooses NCCL then. Single-GPU Colab cannot supply a second GPU by changing this variable.

## Local script route

Download [cloud_training.py](cloud_training.py) and, for DDP, [train_cloud_ddp.py](train_cloud_ddp.py) into the same directory.

```bash
CLOUD_DATA_ROOT=/path/to/clouds_384x576 TRAIN_DEVICE=cpu \
uv run cloud_training.py --demo numbers

CLOUD_DATA_ROOT=/path/to/clouds_384x576 TRAIN_DEVICE=cpu \
uv run cloud_training.py --stop-after 1

CLOUD_DATA_ROOT=/path/to/clouds_384x576 TRAIN_DEVICE=cpu \
uv run cloud_training.py --resume runs/cloud_efficiency/last.pt
```

Keep the same output root and configuration across interruption and resume. Use a new output root for a different experiment. `last.pt`, `best.pt`, `manifest.json` and `history.json` are written locally, not uploaded by the lesson.

## Troubleshooting and scope

Missing files: verify the real root and matching image folder. RLE out of bounds: check CSV variant before changing the decoder. Unsupported precision: use FP32 on CPU or select a compatible CUDA policy. Resume mismatch: inspect config/source/split instead of bypassing the guard. DDP no batches: lower BATCH or provide more real images.

Only an explicit `CLOUD_SMOKE_IMAGE_IDS` comma-separated list enables a partial-data code-path check. This must never be described as representative training or performance. Downloads contain no raw data, credentials, fitted weights or claimed hardware speedups.
