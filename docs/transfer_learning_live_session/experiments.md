# 4. Compare models and explain the result

A model comparison should answer a specific question. For this session: **does ImageNet encoder initialization help the same segmentation model under a fixed training budget?**

This is a different question from “which complete pipeline gets the highest possible score?” Keeping the question narrow helps us interpret the result.

## Write the experiment before running it

| Setting | Run A | Run B |
| :--- | :--- | :--- |
| Architecture | ResNet18 encoder + a fixed decoder design | Same |
| Encoder initialization | Random | ImageNet |
| Decoder initialization | One saved starting state | Same starting state |
| Patient split and preprocessing | Fixed | Same |
| BCE–Dice loss and training settings | Fixed | Same |
| Parameters allowed to learn | Full model | Full model |
| Evaluation threshold and examples | Fixed | Same |

Initialization is the intended difference. Freezing only the pretrained encoder would also change the training policy. It could be a useful experiment, but its result would not isolate initialization.

For the MRI example, hold out **whole patients**, not individual slices, because slices from one person are related. For other datasets, identify their own source of dependence before choosing a split. The [optional patient-split activity](dataset.md) makes this visible.

<div class="mri-lab" data-lab="mri" data-ai-help="off" data-mri-lab="comparison">
<p>Change an additional setting in Run B. Identify the extra difference and explain why a better score would then be harder to attribute to pretraining.</p>
</div>

## Read three kinds of evidence together

**Validation metric:** compare Dice on the same held-out examples using the same decision rule. Training loss alone cannot establish better performance on unseen data.

**Learning curves:** look at both training and validation. If training loss falls while validation loss rises, overfitting is one possibility. Check the split and preprocessing too. If one model learns faster in the first few epochs, distinguish faster convergence from a better fully trained result.

**Prediction panels:** compare the same images, reference masks and predictions. A score change might come from fewer false alarms, fewer missed regions or better boundaries. Show common examples rather than choosing a different “best” image for each model.

## Work through one metric example

For a binary prediction, hard Dice is:

$$\mathrm{Dice}=\frac{2TP}{2TP+FP+FN}$$

Suppose an illustrative mask comparison has **TP=8, FP=2, FN=2**. Its Dice is `16 / 20 = 0.80`. This is a toy calculation, not a result from either model. A threshold changes which probabilities become predicted foreground, so changing it can change the score without changing model weights.

<details>
<summary>Try thresholds and an empty target</summary>

<div class="mri-lab" data-lab="mri" data-ai-help="off" data-mri-lab="threshold">
<p>The synthetic 8 × 8 probability field is fixed. Change its threshold and inspect TP, FP and FN. For an empty target, any predicted foreground is a false positive. This activity defines Dice as 1 when prediction and target are both empty.</p>
</div>

</details>

## State a conclusion that matches the evidence

A supported statement might be: “On this patient split, the pretrained model reached a higher validation Dice within the fixed budget.” It does not prove that pretraining wins on every split or after every amount of training.

A short run on a small validation sample is uncertain. Record the configuration and per-patient outcomes, then repeat across seeds or splits before making a strong claim. If there are many empty slices, inspect foreground cases and false positives separately so one average does not conceal the failure mode.

## Check your understanding

Run B has a better Dice score, but it also uses a different split and more epochs. What should you do before attributing the result to pretraining?

<details>
<summary>Reveal the reasoning</summary>

Repeat the comparison with the same held-out cases and training budget. The current result compares two systems with several differences. Save the new configuration and use common validation examples to interpret what changed.

</details>

You have completed the classroom path. [Return to the overview](README.md) or [try the short code walkthrough](notebook.md).

<details>
<summary>Go deeper: the full notebook’s metric choices</summary>

The full notebook trains with BCE plus per-sample soft Dice. Evaluation reports patient-level Dice over the selected slices, foreground-slice Dice and the fraction of empty slices with any predicted foreground. Missing subgroups are reported as unavailable. The MRI summaries do not replace the industrial competition’s different scoring rule.

Keep the threshold fixed for the initialization comparison. If you want to tune it, do so as a separate validation experiment and keep a final test set untouched.

</details>
