# Pretrained Backbone Architecture Trade-off
*(Milestone 3 · Question 9 Study Guide)*

## 1. Separate Capacity, Memory and Accuracy

Model selection involves several different quantities: parameter storage, activation memory, training optimizer state, latency and validation quality. A lower parameter count proves a smaller parameter tensor budget at the same dtype; it does not prove lower peak VRAM, faster inference or better segmentation.

For Q9 compare ImageNet-initialized ResNet-34 and EfficientNet-B0 U-Nets under the [same five-epoch protocol](../../MILESTONE_3.md#3-common-training-protocol-for-questions-8-9). Change the encoder architecture, keeping the decoder settings and training/evaluation rules fixed. This compares the two complete recipes, not a universal ranking of backbone families.

## 2. Audit Parameters Yourself

Use Q2's counts for the complete model, including encoder, decoder and segmentation head:
$$R=\frac{P_{ResNet}}{P_{EfficientNet}}.$$

Round only the final ratio. The parameter reduction of model B relative to model A is $100(1-P_B/P_A)\%$; it is a different quantity from the ratio. Neither calculation tells you how much peak GPU memory will change.

### Fictional Practice

If two toy models have 9 million and 3 million parameters, their ratio is 3 and the smaller model has about 66.7% fewer parameters. Float32 parameter storage alone is count × 4 bytes. Gradients, buffers, AdamW moments, activations and temporary workspaces are additional allocations.

## 3. Why Depthwise Convolution Can Save Parameters

A standard $k\times k$ convolution with $C_{in}$ inputs and $C_{out}$ outputs has $k^2C_{in}C_{out}$ kernel weights. A depthwise convolution followed by a pointwise projection has:
$$k^2C_{in}+C_{in}C_{out}.$$

For a toy $3\times3$ layer from 8 to 16 channels, these counts are 1152 and 200. This simplified comparison excludes biases and normalization.

EfficientNet's MBConv blocks additionally include channel expansion, squeeze-and-excitation and projection. Count the complete block to estimate actual savings. Expanded intermediate activations may still be large. ResNet's residual additions aid optimization; they do not by themselves determine the full model's memory footprint.

## 4. Measure GPU Memory under a Defined Workload

For Q9 report **training peak allocated CUDA memory** at batch size 16, native resolution and float32. Run each model in a fresh process on the same GPU. Initialize its optimizer, warm up two full optimizer steps (including AdamW state creation), synchronize, reset peak statistics, and measure one more complete step:

```python
# After warm-up steps, before the measured zero_grad/forward/backward/step:
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
# Run a complete optimizer step here with the fixed batch/protocol.
torch.cuda.synchronize()
peak_mib = torch.cuda.max_memory_allocated() / 2**20
```

Report allocated memory rather than reserved caching-allocator memory, and use the same measurement boundaries for both models. If either model runs out of memory, report that outcome; do not silently reduce its batch size. On a CPU-only runtime report `not_measured` for CUDA memory.

Inference is a separate workload: no gradients or AdamW state, usually a deployment-specific batch size and dtype. A training memory measurement alone is insufficient to certify suitability for an edge device. Measure inference memory and latency on the target device before making that claim.

## 5. Compare Scores without a Preset Winner

Use [Guide 8's official metric](guide_08_scratch_vs_pretrained_convergence.md#3-use-the-official-weighted-metric). Record both epoch-5 scores and their signed difference. For this assessment, “competitive” means the EfficientNet score is no more than 0.01 below the ResNet score; this is a declared comparison tolerance, not an expected result. A higher score also meets that definition.

A smaller model can generalize well, but it can also underfit or need a different training schedule. Five epochs do not establish a model's eventual best performance. Repeat seeds or train longer only as a separately labeled follow-up.

<details>
<summary>Practice: Model A has fewer weights but a larger activation tensor. Which has lower peak VRAM?</summary>

The parameter counts alone are insufficient. Measure the complete workload; activations, temporary workspaces, gradients and optimizer state can change the ranking.

</details>

## 6. Guiding Hints and Assignment Task

Read [Question 9](../../MILESTONE_3.md#question-9-pretrained-backbone-architecture-trade-off-resnet-34-vs-efficientnet-b0). Reuse your Q2 counts, measure both training memory peaks when CUDA is available, and compare epoch-5 scores with the stated tolerance. Explain the relevant convolution mechanism and one limitation of using parameter count as a proxy for deployment cost.

Report the ratio, memory measurements, observed score comparison and architectural explanation. Use `tie`, `not_measured` or `oom` where appropriate; do not infer an unmeasured GPU result.
