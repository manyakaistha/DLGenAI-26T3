# Optional: input channels and normalization

**This page is for later implementation.** You can understand pretrained encoders without learning another preprocessing pipeline today.

## Two ideas to keep separate

A **channel** is one part of the input tensor. In this dataset, the three channels contain pre-contrast, FLAIR and post-contrast MRI sequences. Their meanings differ from RGB photograph channels.

**Normalization** changes the scale of the numbers the model receives. It applies to the input image; the target mask remains 0 or 1.

## Why it matters when borrowing an encoder

An ImageNet-trained encoder learned from inputs prepared in a particular way. We should prepare our new inputs deliberately and document the choice. Having three channels makes the shape fit; it does not make an MRI scan equivalent to a color photograph.

The optional full notebook keeps the three supplied channels and uses fixed ImageNet normalization constants in both comparison runs. It also supports repeating the FLAIR channel as a separate input experiment.

<details>
<summary>Try the channel activity</summary>

<div class="mri-lab" data-lab="mri" data-ai-help="off" data-mri-lab="channels">
<p>These synthetic intensity values illustrate channel normalization. Choose three sequences or repeated FLAIR. When channel-specific means and scales differ, identical raw values can become different normalized values. The target mask stays unchanged.</p>
</div>

</details>

[Back to the classroom path](README.md)

<details>
<summary>Go deeper: normalization and grayscale adaptation</summary>

## Scaling and normalization do different jobs

The inspected TIFF image samples store 8-bit intensities. The notebook validates that format and converts them to `[0,1]`:

```python
x = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.0
```

Scaling divides a stored numerical range. Channel normalization then applies:

$$z_c = \frac{x_c-\mu_c}{\sigma_c}$$

For torchvision’s ResNet18 ImageNet weights, the documented constants are `mean=(0.485,0.456,0.406)` and `std=(0.229,0.224,0.225)`. The notebook uses them as a fixed starting convention in **both** experiment arms. They do not become MRI population statistics, and using them does not eliminate the domain gap.

For example, `x=0.5` in channel 0 becomes approximately `(0.5−0.485)/0.229 = 0.066`. A normalized value below zero is expected; it does not indicate a corrupted image. Never normalize the target mask with image constants.

## Do not blindly copy the classification transform

The [weight documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) supplies a classification transform that includes resizing and a center crop. A segmentation pipeline must transform image and target geometry together. Cropping only the image would detach its target, and a central crop could remove labelled tissue.

For this session, preserve the entire square field of view, resize image/mask to the same requested size, use bilinear image interpolation and nearest-neighbor mask interpolation, then apply channel normalization to the image only. The default 128 × 128 setting is a classroom speed choice; 256 × 256 retains the inspected files’ native resolution. Hold it fixed in each comparison. Full-field resizing is our teaching pipeline, not a claim to reproduce the author’s volume preprocessing.

## What if the future input has one channel?

This becomes directly relevant again for grayscale industrial strips. There are several strategies:

1. Repeat one channel three times and keep the original input convolution.
2. Replace the first convolution with a one-channel convolution and initialize its weights from the RGB kernel.
3. Learn an adapter from one channel into three channels.

For the **same unnormalized grayscale tensor** `g` repeated in every channel, linearity gives:

$$\sum_{c=1}^{3} W_c * g = \left(\sum_{c=1}^{3}W_c\right)*g$$

So summing the three input-channel kernels reproduces that first convolution, with the same bias and padding. Averaging scales the convolutional contribution by one third; it is not exact equivalence. This statement is about the first convolution, not complete-network performance.

Channel-specific ImageNet normalization complicates the equality: identical raw grayscale channels become different normalized inputs. Do not claim that summing raw weights exactly reproduces the full normalized RGB pipeline. Our small weight-sum check in the notebook deliberately uses unnormalized tensors and says so.

For this session’s main three-sequence experiment, do **not** collapse the input convolution. That would throw away distinctions between sequences and change the experiment.


</details>
