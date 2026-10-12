# Pixels, Masks & Experiments

## A live lab in industrial segmentation and W&B

A separate teaching companion to Milestone 1 Guides 1–5. Start with the notebook, then follow a strip image through labels, model outputs, RLE, resampling and texture. The original milestone guides and assessment remain the course reference.

**Can this fit in 60–70 minutes?** Yes, as a prepared demonstration of every core topic, with short prediction questions. It cannot include all derivations, independent coding, account setup and detailed exercises in that time. Allow 90–120 minutes for a hands-on workshop. The notes contain the deeper material for later reading.

| Time | Live focus | Student action |
|---|---|---|
| 0–3 | State the task: locate four defect types on 800 × 128 strips | Distinguish an image label from a pixel label |
| 3–20 | W&B notebook: config → training metrics → plots → artifacts | Predict what a high accuracy / low Dice run means |
| 20–28 | Guide 1: long table, clean images, sparsity, cardinality | Count labels on the toy table |
| 28–36 | Guide 2: image co-occurrence vs pixel overlap; logits and heads | Move a logit and compare activations |
| 36–48 | Guide 3: encode/decode, column order, boundaries | Paint a mask and read its RLE |
| 48–57 | Guide 4: aspect ratio, interpolation, sampling phase | Move a thin crack before resizing |
| 57–65 | Guide 5: pooled SD, Cohen's d, texture and contrast | Compare equal-histogram textures |
| 65–70 | Questions / exit check | Explain one data bug and one logging safeguard |

For exactly 60 minutes: stop W&B at minute 17, show one RLE preset rather than student painting, and use the texture example without the numerical exercise. Keep the phase experiment and checkpoint restore; they prevent common misconceptions.

## Before students arrive

1. Download [the notebook](wandb_tutorial.ipynb), run it once, and keep the completed dashboard open as a network fallback.
2. The default is offline. Set `MODE = "online"`, run `wandb.login()`, and use your own project/account for the live dashboard. Credentials never belong in notebook source.
3. Keep a local notebook runtime ready. No competition data or GPU is needed by the notebook. Installation/account setup belongs before the timed lesson.
4. Preview these notes with `uv run --no-project scripts/build_live_session.py`, then `uv run --no-project python -m http.server 8768 --directory runs/live_session_preview`.
5. Open each interactive page once. Figures are pre-generated; students need no Python to read or interact with the notes.

## Follow the session

- [W&B lab — one page, most work in the notebook](wandb.md)
- [1 · Count the right things](01_data.md)
- [2 · Decide what a pixel can mean](02_heads.md)
- [3 · Turn masks into numbers](03_rle.md)
- [4 · Preserve geometry and evidence](04_geometry.md)
- [5 · See beyond average brightness](05_texture.md)
- [Instructor prompts, deeper exercises and coverage map](instructor.md)

## What the notebook actually measures

A three-layer CNN learns bright rectangles on 48 synthetic strips and evaluates 16 held-out strips. It logs batch loss, learning rate, validation loss, foreground Dice, clean-strip accuracy and pixel accuracy; segmentation overlays, a sortable table, probability histogram, pixel confusion matrix, learning curve and threshold plot; and a best-model artifact with configuration and measured history. The default recipe uses 20 epochs and a 193-parameter model. It restores the local best checkpoint and checks its metric. This deliberately easy one-class toy preserves the industrial aspect ratio and teaches tracking. It does not estimate competition performance.

Created by manaykaistha.
