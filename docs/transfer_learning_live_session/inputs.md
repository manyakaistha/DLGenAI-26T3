# 4. Prepare MRI channels for an ImageNet encoder

The encoder’s first convolution expects three channels. Our TIFF also has three channels. The shapes fit, but the meanings differ: **pre-contrast, FLAIR, post-contrast are not red, green and blue**.

## Two valid experimental inputs

**Three supplied sequences:** keep channels 0, 1 and 2 in the documented order. This can expose complementary information, although missing sequences may be duplicated FLAIR. This is the notebook default.

**FLAIR repeated three times:** select `image[...,1]`, then repeat it across three channels. This keeps the network interface unchanged while removing information from the other supplied sequences. Treat it as a separate input ablation, applied to both initialization arms if you want to compare them fairly.

```python
# image is H × W × 3, in the dataset's sequence order.
flair = image[..., 1:2]
flair_three = np.repeat(flair, 3, axis=2)
```

Do not average the three sequences and call the result FLAIR. Do not use a color-space grayscale conversion. These operations mix the sequences with weights unrelated to their acquisition meaning.

## Scaling and normalization do different jobs

The inspected TIFF image samples store 8-bit intensities. The notebook validates that format and converts them to `[0,1]`:

```python
x = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.0
```

Scaling divides a stored numerical range. Channel normalization then applies:

$$z_c = \frac{x_c-\mu_c}{\sigma_c}$$

For torchvision’s ResNet18 ImageNet weights, the documented constants are `mean=(0.485,0.456,0.406)` and `std=(0.229,0.224,0.225)`. The notebook uses them as a fixed starting convention in **both** experiment arms. They do not become MRI population statistics, and using them does not eliminate the domain gap.

For example, `x=0.5` in channel 0 becomes approximately `(0.5−0.485)/0.229 = 0.066`. A normalized value below zero is expected; it does not indicate a corrupted image. Never normalize the target mask with image constants.

<div class="mri-lab" data-lab="mri" data-ai-help="off" data-mri-lab="channels">
<p><strong>Channel and normalization lab.</strong> Select the three supplied MRI sequences or repeated FLAIR. Move synthetic intensity controls and compare raw values to normalized values. Repeated raw intensities become different normalized values when channel means and standard deviations differ. These numbers are illustrative, not measurements of a patient.</p>
</div>

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

## Activity: is “three channels” enough?

A student displays the TIFF as a colored image and decides to apply hue and saturation jitter because the array has three channels. Explain the problem.

<details>
<summary>Reveal the reasoning</summary>

The channels represent different MRI sequences, not visible-light color. Hue/saturation operations mix or reinterpret those sequences. Shape compatibility is not semantic compatibility. Start with conservative, explicitly justified transforms; any spatial transform must still be shared with the mask.

</details>

Next: [freezing and fine-tuning](finetuning.md).
