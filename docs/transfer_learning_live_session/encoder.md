# 2. Trace the encoder and decoder

You already know the U-Net idea: compress an image into features, then recover a dense prediction using high-resolution information from earlier stages. A pretrained encoder changes the source of those features. It does not remove the need for a decoder.

## Classification and segmentation ask different questions

A ResNet classifier finishes with global average pooling and a linear layer. Pooling summarizes the spatial grid; the linear layer predicts image-level categories. For segmentation we retain spatial feature maps, discard the classification endpoint and construct a decoder that predicts one logit at every output pixel.

The encoder is the visual feature extractor. The decoder combines coarse context with finer spatial detail. A segmentation head is the final convolution mapping decoder features into the number of target channels. These are roles, not particular library names.

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

<div class="mri-lab" data-lab="mri" data-ai-help="off" data-mri-lab="shapes">
<p><strong>Feature-shape lab.</strong> Choose the input dimensions and inspect the named feature stages. At 256 × 256, `s5` is `[B,512,8,8]`; resize it to `s4` at 16 × 16, concatenate to get 768 channels, then convolve to 256. Changing input height/width does not change these channel counts.</p>
</div>

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

## Activity: find the broken convolution

A developer upsamples `[B,512,8,8]`, concatenates `[B,256,16,16]`, then creates `Conv2d(512,256,3)`. What is wrong?

<details>
<summary>Reveal the reasoning</summary>

The concatenated tensor has 768 input channels. The convolution must accept 768, or an explicit projection must reduce the channels first. The height and width match after resizing; the channel contract is the failure.

</details>

The industrial strip is 128 × 800. The same feature-scale reasoning applies, but retain that aspect ratio instead of copying the MRI’s square resize. Reference: [torchvision ResNet implementation](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py).

Next: [what pretraining changes](transfer.md).
