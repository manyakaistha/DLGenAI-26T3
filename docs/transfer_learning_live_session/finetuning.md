# 3. Freeze, unfreeze and fine-tune

A newly initialized decoder needs to learn how to use encoder features. You can let all model parameters update, or temporarily keep the encoder fixed while training the decoder. The choice changes the training policy and should be recorded.

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

<div class="mri-lab" data-lab="mri" data-ai-help="off" data-mri-lab="freeze">
<p><strong>Gradient and buffer lab.</strong> One synthetic encoder weight starts at 1 and has an illustrative gradient of 0.4. With SGD learning rate 0.1, an enabled optimizer step changes it to 0.96. Freeze the weight, omit it from the optimizer, and independently toggle BatchNorm mode to see why a running-mean buffer can still change.</p>
</div>

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

## Choose the policy for the question

The optional full experiment’s main two arms use **full fine-tuning with the same learning rate**, so initialization is the intended difference. After that comparison, you may run a separate frozen-encoder experiment. Do not silently freeze only one main arm, change its learning rates, and still label the result “effect of pretraining alone.”

## Activity: the encoder changed without gradients

All encoder parameters have `requires_grad=False`, yet a saved encoder state differs after an epoch. What should you inspect first?

<details>
<summary>Reveal the reasoning</summary>

Inspect BatchNorm `running_mean`, `running_var` and `num_batches_tracked`. A later `model.train()` may have re-enabled training behavior. Distinguish changing parameter tensors from changing buffers, then verify optimizer membership and stale gradients if actual weights changed.

</details>

Next: [evaluate the evidence](experiments.md).
