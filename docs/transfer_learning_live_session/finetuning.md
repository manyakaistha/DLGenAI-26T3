# 3. Choose what should learn

**Main idea: we can keep the encoder fixed, or let it adapt to the new task.**

The borrowed encoder already has learned weights. The new decoder starts without knowing how to make our masks. We choose which parts can change during training.

## Option 1: freeze the encoder

Keep its learned parameters fixed and train the decoder. This is a way to start by using the encoder as a feature extractor.

```python
for parameter in model.encoder.parameters():
    parameter.requires_grad = False
```

`requires_grad=False` tells PyTorch not to compute gradients for those parameters. In the complete implementation, we also choose the right optimizer parameters and handle the encoder’s training/evaluation mode.

## Option 2: fine-tune the model

Let the encoder adapt as the decoder learns. The starting features are reused, but their weights can now change.

```python
for parameter in model.encoder.parameters():
    parameter.requires_grad = True
```

Neither choice is always best. For now, understand the difference; we can compare their results in a later experiment.

<div class="mri-lab" data-lab="mri" data-ai-help="off" data-mri-lab="freeze">
<p>Choose “train the decoder only” or “fine-tune both.” The decoder learns in both cases; the encoder is fixed in the first and can update in the second.</p>
</div>

## Quick check

If the encoder is frozen, can the model still learn?

<details>
<summary>Check your understanding</summary>

Yes. The decoder can learn to use the features supplied by the fixed encoder. Frozen encoder parameters do not mean the entire model is frozen.

</details>

Next: [make a fair comparison](experiments.md).

<details>
<summary>Go deeper: optimizer settings and BatchNorm</summary>

## Three separate switches

1. **`requires_grad`:** should autograd compute gradients for this parameter?
2. **Optimizer membership and learning rate:** should an optimizer update this parameter, and how strongly?
3. **Module mode:** should layers such as BatchNorm use training behavior or evaluation behavior?

`model.eval()` does not disable autograd. `torch.no_grad()` prevents building an autograd graph in its scope, but does not by itself put BatchNorm into evaluation mode. `requires_grad=False` freezes parameters but does not automatically freeze running-statistic buffers.

## Freeze the encoder parameters

Our model groups the feature extractor as `model.encoder` and the learned reconstruction path as `model.decoder`.

```python
for parameter in model.encoder.parameters():
    parameter.requires_grad_(False)
optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=1e-3)
```

Create a new optimizer for the policy you intend. When reusing a model after previous optimization, clear old gradients and rebuild the optimizer so stale state or parameter membership does not silently affect your lesson.

The encoder still performs a forward pass. Freezing does not make feature computation free, and training must still backpropagate through the decoder. Memory and speed changes depend on the implementation and hardware.

## The BatchNorm trap

BatchNorm has learned affine parameters, commonly called gamma and beta, **and** running estimates of mean and variance. Those running statistics are buffers, not ordinary gradient-updated weights. They can change in training mode even after every encoder parameter has `requires_grad=False`.

For a fixed encoder in this exercise:

```python
model.train()          # decoder trains; this also puts encoder in train mode
model.encoder.eval()   # restore fixed encoder BatchNorm behavior afterwards
```

Do this at the start of **each training epoch**, after the outer `model.train()`. Calling it once before a later `model.train()` is insufficient. The new decoder remains in train mode. For the full fine-tuning main comparison, both arms instead use the same all-train policy.

The [PyTorch BatchNorm documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html) explains the running estimates used in evaluation. For very small batches, how to handle those estimates becomes an experimental choice rather than a universal rule.

## Unfreeze deliberately

```python
for parameter in model.encoder.parameters():
    parameter.requires_grad_(True)
optimizer = torch.optim.AdamW([
    {"params": model.encoder.parameters(), "lr": 1e-4},
    {"params": model.decoder.parameters(), "lr": 1e-3},
])
```

These rates are illustrative starting values, not tuned settings for this dataset. A smaller encoder rate is a way to adapt existing features more cautiously while the new decoder learns faster. Rebuilding the optimizer starts new optimizer state; save that policy in the run configuration. Alternatives can preserve state deliberately, but are beyond the first demonstration.

Check the counts:

```python
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"{trainable:,} / {total:,} trainable parameters")
```

During debugging, inspect `parameter.grad` after backward and compare a copied parameter before/after the step. To confirm a truly fixed encoder, also compare its BatchNorm buffers. A nonzero gradient alone does not prove an update occurred.


</details>
