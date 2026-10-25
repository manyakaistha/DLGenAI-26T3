# 2. Understand what pretraining changes

**Pretraining changes a model’s starting state.** It does not change the task we want the final model to perform, and it does not remove the need for training on our own data.

Three terms help separate the decisions:

- **Architecture:** the layers and connections, such as a ResNet18 encoder with a U-Net-style decoder.
- **Initialization:** the values those layers start with, either random or learned previously.
- **Training policy:** which parameters may update, and how we optimize them.

We can change one of these while keeping the others fixed.

## Why features from photographs might help

ImageNet pretraining teaches an encoder to extract features useful for classifying photographs. Some patterns, such as edges and texture combinations, may also be useful in another image task. Starting with them may reduce how much must be learned from scratch.

But MRI scans and industrial surfaces differ from photographs. More task-specific features may need substantial adaptation. Whether pretraining helps, and by how much, must be measured on a valid held-out split. A learned starting point is an opportunity, not a guaranteed improvement.

## Read the two constructors carefully

```python
from torchvision.models import resnet18, ResNet18_Weights

random_model = resnet18(weights=None)
pretrained_model = resnet18(
    weights=ResNet18_Weights.IMAGENET1K_V1
)
```

`weights=None` starts the model with randomly initialized parameters. The weight identifier in the second constructor loads an ImageNet-trained state. The architecture is the same in both cases.

These objects are still classifiers. Their classification pooling/output layers are not a segmentation decoder. In our segmentation model, we keep the spatial feature layers and add a newly initialized decoder and mask-output layer.

## Which parts start with knowledge?

| Component | Random encoder experiment | Pretrained encoder experiment |
| :--- | :--- | :--- |
| Encoder | Random starting weights | ImageNet starting weights |
| Decoder and mask head | New random weights | The same new starting weights |
| Target task | Learn segmentation | Learn the same segmentation task |

Keeping the decoder’s initial state the same makes an initialization comparison easier to interpret. Otherwise different decoder starts introduce another source of variation.

<div class="mri-lab" data-lab="mri" data-ai-help="off" data-mri-lab="provenance">
<p>Inspect the starting state of each component. Compare an ImageNet encoder with a complete MRI-trained checkpoint: what might already have been learned in each case?</p>
</div>

## The MRI example: same task structure, different input meaning

The supplied MRI image has three channels, but they represent pre-contrast, FLAIR and post-contrast sequences, not RGB colors. Its target is one binary abnormality mask. We can use it to illustrate segmentation without learning an entirely new project pipeline during this class.

An ImageNet encoder accepting three channels is shape-compatible with this input. That alone does not make the input distribution appropriate. Channel scaling and normalization still need a deliberate choice; the [optional input notes](inputs.md) explain the implementation.

## Check your understanding

A student replaces a custom U-Net with a pretrained ResNet-based U-Net and gets a different validation score. Have they isolated the effect of pretraining?

<details>
<summary>Reveal the reasoning</summary>

No. The architecture and starting weights both changed. To investigate pretraining, also compare the same ResNet-based U-Net with random encoder weights, keeping the other settings fixed.

</details>

<details>
<summary>Go deeper: checkpoint history and validation</summary>

A complete MRI-trained checkpoint may already have seen patients you intend to hold out. Investigate its training history before treating those patients as unseen. The optional full experiment uses ImageNet encoder weights and a new decoder, not the dataset author’s MRI-trained U-Net checkpoint.

The encoder’s pretrained state also includes BatchNorm running statistics. Record the weight identifier and use the same training-mode policy in a matched comparison.

Reference: [PyTorch transfer-learning tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html).

</details>

Next: [freezing and fine-tuning](finetuning.md).
