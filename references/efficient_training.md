# Efficient training live session · 29 October 2026

Session 4: **60-minute core**, AMP/FP16/BF16, accumulation, update-based warmup/cosine, complete epoch-boundary resume, validation diagnostics. DDP is separate optional depth, excluded from default slide route.

## Sources and routes

- Student source: `docs/efficient_training_session/README.md`, `01_clouds_and_training.md` through `06_validation.md`; optional `07_ddp_optional.md`; exact handoffs/setup `08_notebooks.md`.
- Interactive HTML presentation (existing approved format): `presentation/{index.html,presentation.css,presentation.js,content.js}`. Twenty core + three optional slides; exact Colab cell IDs plus guide handoff anchors on slides 9, 12, 15; optional DDP handoff 23. Tiny seeded footer flowers/butterflies, six palettes, accessible native controls and offline HTML save.
- Event-driven representation/lifecycle/accumulation/shard activities: `activities.js`, `activities.css`. No fabricated benchmark data or animation timelines in teaching activities.
- Runtime: `cloud_training.py`; optional external-process worker `train_cloud_ddp.py`.
- Clean self-contained downloads: `cloud_training.ipynb` (16 cells), `cloud_ddp.ipynb` (12 cells). Notebook generator embeds complete Python sources with stable cell IDs for Colab. Regenerate after editing runtimes.
- Optional instructor speaking script: `presentation/SPEAKER_SCRIPT.md`, local only, excluded from publication.
- Public route `/session-4/`, deck `/session-4/presentation/`, wired through `build_student_pages.py` and `build_student_hub.py`. No existing bootcamp-preview entry points added.

## Dataset contract

Same real **Understanding Clouds from Satellite Images** as bootcamp; independent Fish/Flower/Gravel/Sugar masks. Mirror `phunghieu/understanding-clouds-from-satellite-images-384x576`, W576/H384 matching re-encoded CSV; original W2100/H1400 explicit alternative. CSV Image_Label/EncodedPixels, last-underscore split, four rows/image including empty strings, one-based F-order RLE. Never use industrial 800×128 dimensions/classes in this lesson. Resize to H64/W96, RGB bilinear/mask nearest; diagnostics are not official scores.

Expected `CLOUD_DATA_ROOT/{train.csv,train_images/*.jpg}`. `CLOUD_DATA_VARIANT=resized_384x576|original`. `CLOUD_SPLIT_MANIFEST` can reuse bootcamp session1_evidence.json image partitions. Seeded image split before caps; default caps train32/val16. Optional explicit CLOUD_SMOKE_IMAGE_IDS makes partial real files a code-path check only. No synthetic fallback or raw data published.

## Runtime and scope

PyTorch2.8.0, NumPy2.2.6, Pillow12.3.0 pinned in script metadata. Tiny four-output CNN has no BatchNorm/dropout. Loss is per-image BCE + channel-mean soft Dice, explicitly FP32 reductions. CUDA FP16 uses GradScaler; native CUDA BF16 and CPU BF16 use disabled scaler; CPU FP16 errors explicitly. FP32 CPU route covers core correctness.

Accumulation uses actual B_i/N, resets once/window, unscales/clips once/update; scheduler advances only for successful optimizer steps. Checkpoints atomically save all state, strict config/data identity, progress, best/patience and RNG; fixed full horizon, epoch-boundary same-runtime reproducibility only. last versus best are distinct. Validation tracks image loss, positive-pair Dice, empty-target pairs predicted entirely empty, per-class support; missing denominator null.

DDP one process/device, NCCL CUDA or explicit CPU/Gloo; sampler+loader drop training tails with logged omissions, set_epoch, no_sync covers forward/backward, equal work, unique nonpadding validation shards, sums/counts before division, raw model validation without BatchNorm buffers, collectives outside rank-zero writes, per-rank RNG, fixed-world-size resume. File rendezvous uses `file://` + native path (percent-encoded spaces are incompatible with PyTorch FileStore). GPU, elastic, mid-epoch and multi-node recovery are not claimed tested.

## Build and verification

```bash
uv run --no-project python scripts/build_efficiency_notebooks.py
uv run --no-project scripts/build_efficiency_session.py
uv run --no-project python scripts/build_efficiency_presentation.py
uv run --no-project python scripts/verify_efficiency_session.py
```

Verifier uses installed project Torch runtime and four real mirror audit images; never publishes their bytes. Writes evidence under `runs/efficiency_checks/`. All core notebook code cells executed; physical/accumulated gradient max error 2.794e-8; CPU interrupted/uninterrupted weight max error0. True two-process CPU/Gloo file-rendezvous training + resume to three epochs: replicas identical, one unique validation image counted once, two training images used/one omitted each epoch. Scope is correctness smoke only; CUDA/NCCL, native BF16 GPU throughput, original full-data runs untested locally.

Publication checkout `runs/student_pages_publish` has newer shared navigation than root. Copy new session source/builders explicitly; patch main builder and hub without replacing shared files. Clean full build 131 HTML pages (121 existing +10 session4); all local links/anchors validated. No raw data/checkpoints/instructor notes in publication.
