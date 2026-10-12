# W&B: make an experiment inspectable

Download [the executable notebook](wandb_tutorial.ipynb) or [its paired Python cells](wandb_tutorial.py). Use this page as a reference while working through the notebook.

[Open the student notebook in Colab](https://colab.research.google.com/github/manyakaistha/DLGenAI-26T3/blob/main/docs/live_session_materials/wandb_tutorial.ipynb). Install dependencies before the timed session, then choose online mode for the dashboard.

## The experiment record

<div class="flow" role="img" aria-label="Configuration defines the recipe; training produces history and examples; the best checkpoint becomes a versioned artifact."><span>Recipe<br><b>config + seed + split</b></span><span>→ Training<br><b>metric history</b></span><span>→ Inspect<br><b>plots + masks</b></span><span>→ Preserve<br><b>checkpoint + artifact</b></span></div>

A **run** is one execution of a recipe. **Config** records inputs to the experiment: architecture, learning rate, sample counts, seed, threshold. **History** records measured values over time. **Summary** records a final or best result. An **artifact** versions a file and connects it to the producing run.

## Explore your run

1. Set online mode and authenticate securely. Initialize the run; open its URL.
2. Show the tiny synthetic dataset and held-out split. Explain the background label before logging overlays.
3. Run training. Open batch loss and epoch validation Dice. The custom axes keep batch steps and epochs separate.
4. Open the validation table. Sort by Dice; inspect a clean strip and a positive strip. Toggle prediction and ground-truth layers.
5. Open the histogram and threshold curve. A threshold changes a binary decision without retraining weights.
6. Open Artifacts. Inspect `best.pt`, config, history, metadata and producing run. Show the local restore check.

## Four API patterns to remember

```python
run = wandb.init(project="pixels-masks-experiments", config=config)
run.log({"epoch": epoch, "val/foreground_dice": measured_dice})
run.log({"example": wandb.Image(image, masks=mask_layers)})
artifact = wandb.Artifact("tiny-strip-segmenter", type="model")
artifact.add_file("runs/live_session_demo/best.pt")
run.log_artifact(artifact, aliases=["best"])
run.finish()
```

The full notebook adds separate metric axes, a held-out evaluator, rich plots, checkpoint metadata and upload completion checks. A local save and a successful cloud upload are different events. Offline recording must be synced before you can view a dashboard or download an artifact from W&B.

## Read metrics together

| Evidence | What it answers | What it cannot prove |
|---|---|---|
| Train loss | Is optimization progressing on training batches? | Generalization |
| Held-out foreground Dice | Are positive validation masks localized? | Clean-image behavior |
| Empty accuracy | Do clean validation strips remain clean? | Positive defect recall |
| Pixel accuracy | How many pixel decisions are correct? | Success on rare foreground |
| Overlay / table | Where are false positives and misses? | Dataset-wide averages |
| Threshold sweep | How sensitive is the validation result to a decision rule? | A fresh unbiased test score |

This is a toy binary task. Its foreground Dice and empty accuracy are diagnostics, **not the official competition score**. An unweighted average dominated by empty cases can conceal a detector that misses defects.

## Compare two runs fairly

Repeat with learning rate 0.001 and a distinct name. Keep the split, seed, dataset, epochs and metric definitions fixed. In W&B, compare loss and Dice curves, then inspect examples. Do not declare a general winner from a 16-image toy holdout. Record what changed and what remains uncertain.

`best` is a movable alias; `v0`, `v1`, … identify versions. Save the full artifact path when reproducing a result. Restore model architecture and preprocessing as well as weights. Download code and offline-sync commands are at the end of the notebook.

**Check:** If loss improves but Dice stays zero, inspect probabilities, threshold, foreground frequency and overlays before assuming the logger is broken.

## Prepared example and sources

The default recipe trains a **193-parameter CNN**, using 48 training strips and 16 separately generated validation strips for 20 epochs. One measured CPU run reached foreground Dice **0.984**, foreground recall **0.974**, and clean-strip accuracy **1.0**. It initially spent several epochs at **0 Dice** despite **98.14% pixel accuracy**. These are synthetic lesson results; hardware and library versions can affect numerical details.

The notebook also logs false-positive pixel count, a pixel confusion matrix, parameter count and the all-background baseline. Media are from the final epoch; the model artifact records the best measured epoch. With the default recipe those happen to coincide. Plots are also saved locally for later review.

Official API references: [logging metrics and media](https://docs.wandb.ai/guides/track/log/), [tables](https://docs.wandb.ai/guides/track/log/log-tables), [image masks](https://docs.wandb.ai/ref/python/data-types/image/), and [downloading artifacts](https://docs.wandb.ai/models/artifacts/download-and-use-an-artifact).
