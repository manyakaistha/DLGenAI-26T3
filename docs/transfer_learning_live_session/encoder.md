# 1. Start with the U-Net you already know

**Main idea: an encoder finds useful features; a decoder uses them to draw a mask.**

Think back to the U-Net you trained. Its left side gradually made the image smaller and built up features. Its right side used those features to produce a prediction at each pixel.

<div class="lesson-flow" aria-label="Segmentation flow">
<span>Image</span><span>Encoder<br><small>Find features</small></span><span>Decoder<br><small>Build the mask</small></span><span>Predicted mask</span>
</div>

## Give the familiar parts their names

The **encoder** learns patterns in the input: edges, textures and more complex combinations. Its feature maps are intermediate representations, not finished masks.

The **decoder** combines those features and increases the spatial size until it can predict a mask. U-Net’s **skip connections** bring earlier, more detailed features across to the decoder.

You have already used all three. Today’s change is where the encoder’s starting weights come from.

## What changes with a pretrained encoder?

We can use the feature-extracting part of a network such as **ResNet18** as the encoder. “Backbone” is another name you may hear for this feature extractor.

The decoder still has work to learn. A network trained to classify photographs does not arrive knowing how to draw our segmentation masks.

<div class="lesson-note"><strong>Keep this picture in mind:</strong> familiar U-Net idea, a different starting point for the encoder.</div>

## Quick check

If we borrow a trained encoder, do we automatically get a trained segmentation model?

<details>
<summary>Check your understanding</summary>

No. We have reused a feature extractor. The decoder and output layer still need to learn the new segmentation task, and the encoder may need to adapt too.

</details>

## Optional: explore the feature sizes

Use the activity if you want to see how the image changes size through the encoder. You do not need to memorise the numbers to follow the lesson.

<details>
<summary>Try the feature-size activity</summary>

<div class="mri-lab" data-lab="mri" data-ai-help="off" data-mri-lab="shapes">
<p>A 256 × 256 input becomes a series of smaller feature maps. At one decoder join, 512 feature channels meet 256 skip channels, giving 768 channels before the next convolution. Try another input size.</p>
</div>

</details>

Next: [borrow a trained encoder](transfer.md).

<details>
<summary>Go deeper: exact tensor shapes and decoder code</summary>

## Trace one 256 × 256 MRI slice

Our companion model uses the stem and four residual stages of torchvision’s ResNet18. For a batch size `B`, its feature shapes are:

| Feature | Operation | Tensor shape |
| :--- | :--- | :--- |
| Input | Three MRI channels | `[B,3,256,256]` |
| `s1` | `conv1 → bn1 → relu`, stride 2 | `[B,64,128,128]` |
| `s2` | `maxpool → layer1` | `[B,64,64,64]` |
| `s3` | `layer2` | `[B,128,32,32]` |
| `s4` | `layer3` | `[B,256,16,16]` |
| `s5` | `layer4` | `[B,512,8,8]` |

The stem feature is captured **before max pooling**. Moving that tap changes its spatial size. The notebook prints these shapes instead of relying on a diagram alone. At classroom resolution 128 × 128, all spatial dimensions in the table halve; channel counts stay the same.

![ResNet encoder feature maps connected to a U-Net-style decoder with labelled spatial scales](figures/encoder_decoder.svg)

## Build one decoder block

To join the deepest feature with the previous encoder stage:

```python
up = F.interpolate(s5, size=s4.shape[-2:], mode="bilinear",
                   align_corners=False)
joined = torch.cat([up, s4], dim=1)
d4 = block(joined)
```

- `size=s4.shape[-2:]` requests the skip tensor’s exact height and width. This avoids assuming that doubling always matches, especially on odd dimensions.
- `torch.cat(..., dim=1)` joins channels: 512 plus 256 gives **768**, not 512.
- The convolution block consumes 768 channels and learns a smaller output representation; our first block emits 256.
- Bilinear interpolation is appropriate for continuous feature maps. It does not justify bilinear resizing of hard target labels.

Repeat with `s3`, `s2` and `s1`. Finally interpolate to the original input size, refine, and apply a one-channel `1×1` convolution. No final sigmoid is included in `forward`; the training loss expects logits.

## Two kinds of skip connection

A **ResNet residual shortcut** usually adds tensors of compatible shape: `output = branch(x) + shortcut(x)`. A projection can align channel counts or stride when necessary. The addition offers a route for information and gradients within the encoder.

A **U-Net skip connection** passes an earlier feature into the decoder, commonly by channel concatenation. It restores access to spatial detail. Concatenation increases the number of channels, while addition requires matching channels. Saying “both have skip connections” does not make their mechanics interchangeable.

## The decoder does not reconstruct the original photograph

Upsampling alone cannot recover every detail lost during downsampling. Fine features from skip connections and learned convolutional refinement help produce useful boundaries. The objective is a target mask, not an exact reconstruction of MRI intensities. A pretrained encoder still needs adaptation to the new input distribution and target.

## Sanity checks before training

```python
model.eval()
with torch.no_grad():
    logits = model(torch.zeros(2, 3, 128, 128))
assert logits.shape == (2, 1, 128, 128)
assert torch.isfinite(logits).all()
```

This checks geometry and finite outputs, not learned quality. Check target shape and visualization separately. A model can pass this test and still receive mismatched masks or a leaking split.


</details>
