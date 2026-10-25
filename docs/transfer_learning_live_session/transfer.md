# 3. Understand what pretraining changes

An architecture specifies the operations and connections. A set of weights specifies the current numerical parameters inside those operations. A training policy specifies which parameters may change and how. Keep these three ideas separate.

## Same architecture, different starting point

```python
from torchvision.models import resnet18, ResNet18_Weights
random_encoder = resnet18(weights=None)
pretrained_encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
```

Both objects have the same layer structure. The second loads weights learned on ImageNet. Neither constructor creates a trained MRI segmentation decoder. In our model, the classifier is discarded and the decoder and one-channel output head begin with new random weights.

A useful mental picture is an apprentice who already knows how to identify edges, textures and patterns but must learn which combinations delineate the labelled MRI abnormality. Some learned features may transfer; some may be poorly matched. Pretraining is an initialization to investigate, not a promise of higher Dice.

## Keep track of the source of knowledge

**ImageNet initialization** comes from a different pretraining task. **Weights already trained on this MRI dataset** may contain information about the cases you intend to validate on. Those are different experiments.

The dataset author provides [MRI-trained U-Net weights](https://github.com/mateuszbuda/brain-segmentation-pytorch). Do not load those weights for our random-versus-ImageNet comparison. This session’s notebook uses torchvision ImageNet weights only, never the author’s MRI model. We need to know which patients a checkpoint has seen before making claims about held-out performance.

<div class="mri-lab" data-lab="mri" data-ai-help="off" data-mri-lab="provenance">
<p><strong>Weight provenance lab.</strong> Switch among random initialization, ImageNet encoder weights and a full MRI-trained checkpoint. Identify what has been learned already, which layers still need training, and whether you know which patients influenced the checkpoint.</p>
</div>

## The question we can answer today

> Under a fixed classroom training budget, does ImageNet encoder initialization change validation behavior relative to random initialization of the same encoder architecture?

This is deliberately narrower than “Which model is best?” A short run can show a difference in convergence. It cannot establish the best possible result after tuning, or demonstrate that the difference is stable across seeds, splits and hospitals.

## Define the two arms

| Setting | Arm A | Arm B |
| :--- | :--- | :--- |
| Architecture | ResNet18 encoder + the same decoder | Identical |
| Encoder start | Random | ImageNet |
| Decoder/head start | One saved initial state | The same saved initial state |
| Data | Saved patient split and selected slices | Identical |
| Image channels and normalization | Fixed | Identical |
| Training policy | Full fine-tuning | Full fine-tuning |
| Loss | BCE + soft Dice | Identical |
| Epochs, batches, optimizer and LR | Fixed | Identical |
| Evaluation | Same threshold and patient aggregation | Identical |

Full fine-tuning means all encoder and decoder parameters can update. Freezing only the pretrained arm would also change the training policy. That is a useful follow-up question, but it does not isolate initialization.

The pretrained encoder’s BatchNorm buffers also come from pretraining; they are part of that initialization. The notebook records the weight identifier and applies the same training-mode policy in both arms.

## A fair comparison requires more than the same seed

Resetting a seed does not make two different models equivalent. Here we explicitly reuse decoder/head initialization, patient IDs, slice selection, preprocessing and loader configuration. We reset the random sources and shuffle generator for each arm. Training augmentation is omitted from the first comparison to reduce moving parts; a paired augmentation policy can be added equally to both arms later.

The notebook requests deterministic operations where available, but reproducibility across all devices, library versions and floating-point kernels is not guaranteed. Save the environment as well as the seed.

<div class="mri-lab" data-lab="mri" data-ai-help="off" data-mri-lab="comparison">
<p><strong>Experiment audit lab.</strong> Change one setting in the second arm and inspect whether you still isolate pretraining. Different splits, thresholds, epochs or losses introduce confounds. A valid comparison here changes only encoder initialization and has no overlapping training/validation patient IDs.</p>
</div>

## Why no universal winner is shown here

Do not fill in imagined validation scores before running the experiment. Measure loss curves, patient-level Dice, foreground-slice Dice, false positives on empty slices, runtime and predictions. A lower training loss alone is not evidence of better generalization.

If the pretrained model is better after two epochs, say “better under this two-epoch budget on this split.” If it is worse, inspect the input range, normalization, output contract and learning rate before concluding that pretraining never helps. If the curves are close, repeat with a larger patient sample and several seeds before making a strong claim.

## Activity: two changes in disguise

A student compares their custom U-Net with a pretrained ResNet U-Net. Both use the same loss and split. Can the result be attributed only to pretraining?

<details>
<summary>Reveal the reasoning</summary>

No. Both the architecture and initialization changed. The result compares two complete systems. To isolate pretraining, also run the ResNet U-Net with random encoder weights, keeping its decoder initialization and other settings fixed.

</details>

Further reading: [PyTorch transfer-learning tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html), which distinguishes fine-tuning from fixed feature extraction; [torchvision ResNet18 weight documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).

Next: [MRI channels and normalization](inputs.md).
