# 4. Change one thing and compare

**Main idea: a fair comparison helps us understand why a result changed.**

Suppose you train a pretrained model and its Dice score improves. That is encouraging. To attribute the change to pretraining, check what else changed.

## Start with one question

> Does this model benefit from pretrained encoder weights under the same training budget?

Use the **same ResNet-based U-Net** for both runs. Start its encoder randomly in one run and from ImageNet weights in the other.

Keep the split, input preparation, decoder starting weights, BCE–Dice loss, learning settings and training budget the same. Let the same parts learn in both runs. Changing the architecture or freezing only one encoder would introduce another difference.

<div class="mri-lab" data-lab="mri" data-ai-help="off" data-mri-lab="comparison">
<p>Change the second experiment and decide whether you are still testing pretraining alone. The activity checks the setup; it does not invent model scores.</p>
</div>

## Look at evidence you already know

- **Validation Dice:** how well did the predicted regions match the reference masks?
- **Learning curves:** did training improve, and what happened on validation?
- **The same example images:** which regions were missed, and where were false positives added?

Keep the validation examples the same for both models. A short run may tell us about early learning, but more patients and repeated runs are needed for a strong general conclusion.

## Quick check

A student changes the encoder weights **and** trains for more epochs. Can they say that pretraining alone caused the improvement?

<details>
<summary>Check your understanding</summary>

No. Either change could have affected the result. Match the training budget and compare again.

</details>

<div class="lesson-note"><strong>You have completed the classroom path.</strong> You can now explain a pretrained encoder, freezing versus fine-tuning, and a fair comparison. The MRI setup and full experiment are optional follow-up material.</div>

[Return to the session overview](README.md) · [Optional short code demo](notebook.md)

<details>
<summary>Go deeper: thresholds, empty masks and evaluation details</summary>

Keep the threshold fixed when comparing initialization. On empty target slices, a prediction can only add false positives. Decide how empty cases contribute to your metric and state that choice.

The full notebook uses BCE plus per-sample soft Dice for training. At evaluation it reports patient-level Dice over the selected slices, foreground-slice Dice, and the fraction of empty slices with any predicted foreground. These are different views of the result; none should silently replace the industrial competition’s scoring rule.

<div class="mri-lab" data-lab="mri" data-ai-help="off" data-mri-lab="threshold">
<p>This toy 8 × 8 probability field is fixed. Move the threshold to change the prediction. Try an empty target: any predicted foreground is a false positive. Both-empty Dice is defined as 1 in this activity.</p>
</div>

</details>
