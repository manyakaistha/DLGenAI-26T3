# The inference contract

## A checkpoint is one part of inference

Record the architecture, input channels, normalization, resize/padding policy, output-channel order, activation and checkpoint identity. Reconstruct these exactly before evaluating new techniques. A correct weight file with incorrect preprocessing can produce incorrect predictions without a shape error.

The industrial input is **width 800, height 128**. A PyTorch tensor orders spatial axes as **H, W**, so a batch has spatial shape `[128, 800]`. Preserve the 6.25:1 aspect ratio. The four output channels correspond to `defect_a`, `defect_b`, `defect_c`, `defect_d`.

## Output semantics determine activation

For this lesson's independent binary outputs, logits have shape `[B,4,128,800]`. Use sigmoid independently per channel. A pixel may have multiple defects or none. Four-way softmax would force competition between the four defects and force their probabilities to sum to one.

Some checkpoints use a deliberately different output contract, such as background plus four foreground channels with softmax. Reconstruct the head and activation used to train that checkpoint. Never apply this lesson's sigmoid blindly to an incompatible model. Ensembling requires the same foreground meanings and coordinates, even when the internal heads differ.

## Evaluation mode and autograd are separate

```python
model.load_state_dict(checkpoint_state)
model.eval()
with torch.inference_mode():
    logits = model(images)
    probabilities = logits.float().sigmoid()
```

`eval()` switches modules such as dropout and BatchNorm to evaluation behavior. `inference_mode()` disables autograd bookkeeping and additional tracking. Neither replaces the other. Loading a checkpoint alone does not put a new model into evaluation mode.

## A complete decision pipeline

1. Load and normalize images using the training contract.
2. Predict each validated TTA view with each selected model.
3. Invert every view into the same image coordinates.
4. Aggregate continuous foreground probabilities.
5. Restore the original spatial canvas if the network output differs in size.
6. Apply per-class thresholds and any validated component filter.
7. Encode each binary channel and match the sample submission's row order.

If resizing is involved, define the coordinate convention and interpolation explicitly. Resize continuous maps with an appropriate continuous interpolator before thresholding. Use nearest-neighbor for binary training masks. Flips preserve size and are the only TTA transforms in the core lesson.

## Baseline before improvements

Use one ordinary checkpoint, one view, and a fixed threshold with no component filtering. Cache only **held-out** probabilities if memory allows. This reference makes it possible to isolate an improvement from a pipeline bug. Do not tune on the final test labels or use training predictions as held-out evidence.

**Check yourself:** why can a correctly shaped output still be semantically wrong? Give one activation mistake and one preprocessing mistake.
