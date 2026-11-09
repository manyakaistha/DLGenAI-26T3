# Notebook, exercises and setup

[Download the notebook](inference_lab.ipynb) · [Download the paired Python script](inference_lab.py) · [Open in Colab](https://colab.research.google.com/github/manyakaistha/DLGenAI-26T3/blob/main/docs/inference_session/inference_lab.ipynb)

## Before class

The notebook requires Python 3.10+ and NumPy 2.2.6. Run the optional installation cell in a fresh hosted notebook, then run all cells in order. No GPU, account token, data download or training run is required. Save a personal copy before modifying it.

For a local terminal with uv:

```bash
uv run --no-project inference_lab.py
```

The script's inline dependency metadata installs NumPy. Outputs print to the console. The lesson does not write a competition submission file or fetch any external data. The toy grid is H8 × W50 to preserve the industrial aspect ratio while keeping calculations small.

## Notebook A: TTA at minute 15

Run setup, then **A · TTA alignment**. First predict the mirrored coordinate of the off-center marker. Compare aligned and intentionally unaligned maps. The first predictor is exactly equivariant, so aligned TTA must equal its identity output. The second includes a toy positional bias, so view averaging changes its output. That difference alone is not evidence of better real-model quality.

Exercise: add vertical flip and both flips. Verify each transform's inverse and confirm that the channel axis never changes.

## Notebook B: EMA at minute 28

Run **B · EMA and ensembles**. Write the decay-0.5 scalar updates for values 2, 4, 6 before execution. Change decay to 0.9. Explain why a larger decay follows new weights more slowly. Compare averaging parameters and averaging nonlinear predictions.

Exercise: explain what extra information a real checkpoint needs to reproduce EMA inference, including normalization and BatchNorm-buffer policy. The scalar calculation does not train or claim to improve a neural network.

## Notebook C: decisions at minute 40

Run **C · Threshold and area decisions** and **D · Metrics and RLE**. Read TP, FP and FN for each threshold. Inspect 4- versus 8-connectivity and the removal of a legitimate small component. Confirm the exact RLE string `2 3` and empty sentinel.

Exercise: choose one threshold on a designated selection set and report it unchanged on a separate confirmation set. Explain why optimizing and reporting on the same four pixels exaggerates generalization.

## Troubleshooting

If NumPy is missing, execute the installation cell and rerun setup. If a cell reports an undefined name, restart and run cells in order. Notebook download links work offline after saving, while Colab and reference links require internet. If the classroom runtime fails, use the browser slide activities and the worked examples in the notes.

## Exit ticket

1. Where does the inverse transform belong in segmentation TTA?
2. Why does an EMA checkpoint keep the cost of a single model?
3. Why must thresholds be reconsidered after blending?
4. How can a component filter hurt a rare defect?
5. Which predictions are eligible for an honest held-out ensemble score?

<details><summary>Worked explanations</summary>

1. After predicting each transformed view and before combining it with other views, restore the output map to original coordinates.
2. EMA stores one compatible averaged parameter state. Forward evaluation runs that state once per view.
3. Blending changes the scores and their distribution, so the old decision boundary may no longer fit the objective.
4. A genuine small region can fall below the area cutoff and disappear.
5. Every member contributing to a held-out image's score must have excluded that image from training. OOF and final full-fold test ensembling need distinct interpretation.

</details>
