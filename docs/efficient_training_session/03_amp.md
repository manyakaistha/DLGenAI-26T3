# 3 · Automatic mixed precision, one update at a time

**Core · minutes 16–28, including the first notebook handoff.** Ask: “Which tensors do we want to store differently, and which operations can tolerate different arithmetic?” The answer is more selective than converting the model to half precision.

## Autocast is an execution policy

Automatic mixed precision selects dtypes for eligible operations inside a context. Some matrix/convolution operations can use a low-precision format while sensitive operations stay in FP32. The policy depends on device, operation and PyTorch version. Keep the model's trainable parameters in FP32 for this lesson; do not call `model.half()` or manually cast every input.

Our loss explicitly converts logits to FP32 for BCE and Dice reductions. That makes the arithmetic decision visible. It can reduce the performance gain compared with an optimized production loss, and that tradeoff is part of the experiment.

```python
with torch.autocast('cuda', dtype=torch.float16):
    logits = model(images)
    loss = sample_losses(logits, targets).mean()
# Backward is outside the autocast context.
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)
scaler.update()
```

Create `torch.amp.GradScaler('cuda')` once for CUDA FP16. For BF16, choose `dtype=torch.bfloat16` and a disabled scaler in this runtime. FP32 uses a null context and a disabled scaler, preserving one readable control flow.

## Order is part of correctness

<div class="eff-lab" data-eff="lifecycle"><p>Trace: forward → scaled backward → unscale → clip → step → update.</p></div>

Gradients are accumulated across a complete window before unscaling. Unscale once per optimizer update. Clip the actual gradients, not the scaled gradients: otherwise the clipping threshold silently changes with the scale. `scaler.step` checks for nonfinite gradients and may skip the optimizer step. `scaler.update` adjusts the scale afterward.

Our runtime compares scale values before and after the step/update: a decrease means this FP16 step was skipped. Only successful optimizer steps advance the LR schedule. For this lesson there is one optimizer and a standard scaler; more elaborate multi-optimizer arrangements require their own accounting.

A finite number is necessary, but it is not proof of correct training. A decreasing training loss with a broken split, transposed mask or wrong class semantics is still a broken experiment.

## What memory AMP can change

This AdamW example keeps FP32 parameters, FP32 parameter gradients and FP32 moment estimates. Eligible activations can use fewer bytes. A parameter has approximately 4 bytes of weights + 4 bytes of gradients + 8 bytes for two moments: **16 bytes per parameter**, before temporary tensors, allocator overhead and implementation-specific buffers.

Reducing activation bytes therefore does not imply halving the entire GPU footprint. The exact peak depends on saved activations, optimizer implementation, workspaces and allocator behaviour. Measure it rather than promising a fixed multiplier.

## Benchmark without telling a misleading story

Use the same real cloud batch, initial seed, model and update count for every policy. The notebook performs three warmup updates and then ten measured updates. On CUDA it synchronizes before and after the timed interval and reports peak allocated tensor memory. Reserved allocator memory and total device memory are different quantities.

The benchmark is a small compute/update comparison with data already loaded. It does not measure end-to-end JPEG loading or establish full-epoch throughput. A tiny 64×96 model can be overhead-bound. Record hardware, software, batch and policy; rerun meaningful measurements and validate quality before choosing a production policy.

## Handoff 1 · Predict, run, explain

[Open the exact notebook instructions →](08_notebooks.md#amp-demo)

Run **01 · FP16 versus BF16** and **02 · AMP**. Give learners one minute to predict which policy will be skipped on CPU and which number will overflow FP16. Then inspect the runtime's policy function and execute the measured comparison. Ask: “What did we measure, and what did we leave out?”

<details><summary>Repair exercise: backward inside autocast, then clip scaled gradients</summary>

Move backward outside autocast. After all microbatches in the update window, unscale once, clip once, call scaler.step and scaler.update once. Do not zero gradients between microbatches that belong to the same accumulated update.

</details>

Reference: [official AMP examples](https://docs.pytorch.org/docs/2.8/notes/amp_examples.html).
