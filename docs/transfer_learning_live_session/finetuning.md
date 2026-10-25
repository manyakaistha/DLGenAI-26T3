# 3. Freeze or fine-tune the encoder

A pretrained encoder and a new decoder have different starting points. The encoder already contains learned features; the decoder must learn how to use those features for our masks. We choose whether the encoder should stay fixed or adapt too.

## Start by freezing the encoder

Freezing the encoder’s parameters lets us train the new decoder while using a fixed feature extractor:

```python
for parameter in model.encoder.parameters():
    parameter.requires_grad_(False)

optimizer = torch.optim.AdamW(
    model.decoder.parameters(), lr=1e-3
)
```

The loop disables gradients for encoder parameters. The optimizer explicitly contains the decoder parameters. We still run the encoder forward to obtain features; freezing does not eliminate that computation.

Here `1e-3` is an illustrative learning rate, not a tuned recommendation for this dataset. Choose and record learning rates as part of the experiment.

## Fine-tune when you want features to adapt

Fine-tuning allows encoder parameters to update for the new task. For example, we might let a decoder learn first, then unfreeze the encoder:

```python
for parameter in model.encoder.parameters():
    parameter.requires_grad_(True)

optimizer = torch.optim.AdamW([
    {"params": model.encoder.parameters(), "lr": 1e-4},
    {"params": model.decoder.parameters(), "lr": 1e-3},
])
```

This example uses a smaller encoder learning rate so the existing features can change more cautiously. These rates are starting examples; the best policy depends on the data and training budget. Rebuilding the optimizer deliberately starts new optimizer state.

Freezing first is one possible strategy, not a compulsory step. Full fine-tuning from the start is another. Compare policies separately from the question of random versus pretrained initialization.

## Follow one numerical update

For a simple SGD step:

$$w_{\mathrm{new}} = w_{\mathrm{old}} - \eta g$$

If a weight starts at 1.00, its gradient is 0.40 and the learning rate is 0.10, it becomes **0.96**. A frozen parameter stays at 1.00. This arithmetic explains an update; actual AdamW also maintains optimizer state and applies its own update rule.

<div class="mri-lab" data-lab="mri" data-ai-help="off" data-mri-lab="freeze">
<p>Choose a policy, then open the toy update. Explain why the decoder can learn in both cases but the encoder weight changes in only one.</p>
</div>

## A practical detail: BatchNorm mode

An encoder can contain BatchNorm layers with running-statistic buffers. Those buffers can change in training mode even when parameter gradients are disabled. For a fixed encoder in this example, use:

```python
model.train()          # put the decoder in training mode
model.encoder.eval()   # keep encoder BatchNorm statistics fixed
```

Apply this ordering at the start of each training epoch. A later `model.train()` would put the encoder back into training mode too. Also remember that `eval()` changes layer behavior; it does not itself disable gradients.

The complete “fixed encoder” policy therefore combines parameter flags, optimizer membership and module mode. The longer reference below explains how to inspect each one.

## Check your understanding

You unfreeze the encoder by changing `requires_grad`, but leave an optimizer containing only decoder parameters. Will encoder weights update?

<details>
<summary>Reveal the reasoning</summary>

No. Encoder gradients can now be computed, but that optimizer has no encoder parameters to update. Include them in the optimizer or rebuild the optimizer for the new policy. Enabling gradients alone does not perform training.

</details>

Next: [design a controlled comparison](experiments.md).

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
