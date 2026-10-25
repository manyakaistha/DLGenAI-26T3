# 2. Borrow a trained encoder

**Main idea: pretrained weights give the encoder a learned starting point.**

A model’s **architecture** is its structure: layers and connections. Its **weights** are the numbers learned during training. We can keep the structure the same and change the starting weights.

## A familiar analogy

Someone who can already draw has practised lines, edges and shapes. When asked to sketch a new object, they can reuse those skills while learning what is different about the object.

A pretrained encoder can similarly reuse some patterns learned from other images. It still needs to work with our new data. Whether that helps is something we measure.

## Two starting points for the same encoder

```python
# Same architecture, two different starting points.
random_encoder = resnet18(weights=None)
pretrained_encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
```

Here, **ImageNet** is the earlier image-classification dataset. The second line loads encoder weights learned there. These lines create a ResNet classifier; in a segmentation model, we reuse its feature layers and add a decoder.

## What is reused, and what learns next?

- **Reused:** the encoder’s learned feature weights.
- **New:** the segmentation decoder and output layer.
- **Next:** train the new parts, and decide whether to adapt the encoder too.

<div class="mri-lab" data-lab="mri" data-ai-help="off" data-mri-lab="provenance">
<p>Choose random or ImageNet initialization and see which part of the model starts with learned weights. The MRI-trained checkpoint option is an extra example for later discussion.</p>
</div>

## Quick check

Two models use ResNet18. One starts randomly; the other starts with ImageNet weights. Did we change the architecture?

<details>
<summary>Check your understanding</summary>

No. We changed the starting weights. This lets us ask a focused question about pretraining while keeping the structure the same.

</details>

<details>
<summary>Go deeper: why the source of the weights matters</summary>

ImageNet-trained encoder weights and a complete model already trained on this MRI dataset are different starting points. A model trained on the MRI dataset may already have seen patients you intend to use for validation. Check that history before claiming performance on unseen patients.

The optional full experiment uses torchvision’s ImageNet encoder weights, a new decoder and a patient-level split. It does not use the dataset author’s MRI-trained U-Net weights.

Reference: [PyTorch transfer-learning tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html).

</details>

Next: [choose what should learn](finetuning.md).
