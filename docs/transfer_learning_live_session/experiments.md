# 4. Evaluate the evidence

You have already used BCE–Dice. Keep it fixed while testing pretraining. This page connects that familiar training objective to the evidence needed for a model comparison.

## Loss: one binary target, one logit channel

Let `z` be raw logits and `y` a binary mask. The numerically stable BCE term is `BCEWithLogitsLoss(z,y)`. For the soft Dice term, first compute probabilities `p=sigmoid(z)`:

$$L_{\mathrm{Dice}}=1-\frac{2\sum py+\epsilon}{\sum p+\sum y+\epsilon}$$

In the binary-mask example below, we sum spatial dimensions per sample, then averages samples. Its fixed objective is `BCE + DiceLoss`, with epsilon `1e-6`. This is one explicit reduction convention; a batch-global Dice loss would weight examples differently. Neither term thresholds probabilities during training.

```python
probability = torch.sigmoid(logits)
intersection = (probability * target).sum((1, 2, 3))
denominator = probability.sum((1, 2, 3)) + target.sum((1, 2, 3))
soft_dice = (2 * intersection + 1e-6) / (denominator + 1e-6)
loss = F.binary_cross_entropy_with_logits(logits, target) + (1-soft_dice).mean()
```

For empty targets, the soft Dice gradient can be weak depending on epsilon and reduction. BCE still penalizes false-positive probabilities. Inspect behavior rather than assuming a combined loss solves every imbalance problem. See [BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html).

## Evaluation: a decision rule changes the mask

At evaluation, threshold probabilities at a fixed value, initially 0.5:

```python
prediction = logits.sigmoid() >= 0.5
```

For binary masks, Dice can be written as:

$$\mathrm{Dice}=\frac{2TP}{2TP+FP+FN}$$

Our lesson assigns Dice 1 when both prediction and target are empty, and 0 when only one is empty. State that convention explicitly. A threshold is part of the prediction pipeline; selecting different thresholds for two arms also changes the comparison. Keep it fixed here, and perform threshold selection as a separate validation experiment later. Never tune on a final test set.

<div class="mri-lab" data-lab="mri" data-ai-help="off" data-mri-lab="threshold">
<p><strong>Threshold and empty-mask lab.</strong> Inspect a fixed synthetic 8 × 8 probability field. Move the threshold and switch between a foreground target and an empty target. Read TP, FP, FN and hard Dice. These are exact calculations on a toy array, not MRI model predictions.</p>
</div>

## Do not let empty targets dominate the story

A mean over samples can look high when many targets are empty. Check foreground examples and false positives on empty targets alongside your main metric. If several samples come from the same source, choose and state whether that source should receive equal weight rather than letting its sample count decide the weighting.

<details class="optional-mri">
<summary>Optional MRI application: patient-level aggregation and empty slices</summary>

## Do not let empty slices dominate the story

A mean over slices gives patients with many slices more influence and can look high if many targets are empty. The notebook reports several distinct views:

- **Mean patient volume Dice:** accumulate TP/FP/FN across selected slices within each patient, compute one Dice per patient, then average patients equally. At the reduced classroom subset this means the selected slices, not a complete clinical volume.
- **Mean foreground-slice Dice:** evaluate only slices whose target contains labelled foreground, while retaining false positives in the patient-level measure.
- **Empty-slice false-positive rate:** fraction of empty-target slices with at least one predicted foreground pixel. This deliberately strict “any pixel” definition must be labelled.
- **Mean validation loss:** the unthresholded training objective, useful for optimization diagnosis.

An undefined subgroup, such as no foreground slices in a tiny validation subset, is reported as unavailable rather than silently treated as 0 or 1. Do not compare these MRI summaries numerically with the industrial competition’s different scoring formula.

</details>

## Inspect the same validation examples

Choose validation examples by a documented rule independent of predictions. For both arms, inspect the same input, reference mask, predicted mask and probability overlay. This helps reveal whether a change comes from fewer false alarms, more complete regions or a boundary change.

<details class="optional-mri">
<summary>Optional MRI application: how the full notebook selects its figures</summary>

The notebook takes the first foreground and first empty slice in sorted validation order where available. For both arms, show the same FLAIR image, reference mask, predicted mask and probability overlay. An overlay helps reveal whether improvement comes from fewer false alarms, more complete regions or a boundary change.

</details>

Do not pick a different easy example for each model. Save curves, per-patient metrics and the configuration, not only the nicest picture.

## Interpret a short run carefully

- Falling training loss and worsening validation behavior can indicate overfitting, but inspect sample size, preprocessing and curves before diagnosing.
- Similar scores after a short budget can be inconclusive. Pretraining can affect convergence even when final performance later becomes similar.
- A better mean from four validation patients is uncertain. Report individual patients and repeat across seeds or patient splits for stronger conclusions.
- A comparison with the same epoch count answers a fixed-data-pass question. Runtime differs; a fixed-wall-clock comparison answers another question.

## Activity: choose the next experiment

You improve Dice after simultaneously changing the encoder, input channels and threshold. What can you conclude?

<details>
<summary>Reveal the reasoning</summary>

The complete pipeline performed differently under that evaluation, but you cannot assign the change to one factor. Return to the saved baseline. Compare initialization first, then input representation, then threshold selection in separate, recorded steps.

</details>

## Your evidence checklist

Save the train/validation split, selected sample manifest, seed, library versions, input settings, weight source, decoder start, loss, optimizer, batch size, training budget and fixed threshold. Include curves, group-level outcomes where appropriate, and common prediction panels. Write one modest conclusion supported by those artifacts.

<div class="lesson-note">
<strong>Optional MRI practice — run the complete application.</strong>
<a href="notebook.md">Read the real-data notebook setup and experiment walkthrough</a>. You can also use the <a href="classroom_walkthrough.ipynb">smaller concept notebook</a> or <a href="README.md">return to the session overview</a>.
</div>
