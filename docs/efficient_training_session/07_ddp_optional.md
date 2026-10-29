# Optional · DDP, from one update to a complete distributed run

**Outside the 60-minute core.** Allow a separate 60–90-minute workshop plus reading time. Use the same Clouds CSV, native masks, four-channel network, loss and data manifest as the core lesson. Start with CPU/Gloo mechanics; move to CUDA/NCCL only when two compatible GPUs are available.

[Download the complete DDP notebook](cloud_ddp.ipynb) · [Setup and exact starting sections](08_notebooks.md#ddp-demo)

## 1 · The first mental model: replicated work

Imagine two processes. Each owns a full model replica and optimizer. Each receives different training images, computes a local forward pass and local backward pass, then participates in gradient synchronization. Once gradients agree, each optimizer independently applies the same update to its local replica.

DDP is data parallelism. It does not divide one model across GPUs and does not pool device memory. If a full model plus gradients and optimizer state cannot fit on one GPU, ordinary DDP alone will not solve that problem. Activation accumulation, checkpointing activations, model sharding and model parallelism address different constraints.

**Rank** is a process identifier in a process group. **World size** is the number of participating processes. **Local rank** identifies the process's device position on its machine. On one machine with two GPUs, global rank and local rank happen to be 0 and 1; across machines they no longer mean the same thing.

A **process group** defines the participants in collective communication. NCCL is the usual CUDA backend; Gloo supports our CPU demonstration. A **rendezvous** lets the processes discover one another before training.

## 2 · A numerical update you can calculate

Suppose two equal-size local batches produce scalar gradients 2 and 6. DDP's ordinary synchronized gradient is their average, 4. Both replicas then apply an optimizer update using 4. DDP implements communication efficiently in gradient buckets; the conceptual result is averaged parameter gradients, not averaged model predictions or an automatic global loss value.

If each rank computes a local mean over the same number of examples, averaging rank gradients matches the global example mean for a sample-separable loss. With unequal local counts 1 and 3, averaging means gives each rank half the weight, which is not the same as giving every example equal weight. This worker deliberately uses equal training work per rank to avoid silently changing that objective.

**Try:** gradients 2 from one image and 6 from three images. Equal-rank average is 4; image-weighted average is (2×1+6×3)/4=5. Different denominators answer different questions.

<div class="eff-lab" data-eff="shards"><p>Use the shard workbench to compare dropping and padding an odd number of training images.</p></div>

## 3 · Launch a script, not a notebook closure

The notebook embeds the complete core runtime and the DDP worker, writes them beside itself and starts an external process with torchrun. This avoids trying to pickle functions defined in notebook cells. There is no project installation or pretrained download required.

For two CPUs demonstrating synchronization:

```bash
BATCH=1 ACCUMULATION=2 EPOCHS=3 PRECISION=fp32 \
CLOUD_DATA_ROOT=/path/to/clouds_384x576 \
uv run --with torch==2.8.0 --with numpy==2.2.6 --with Pillow==12.3.0 \
python -m torch.distributed.run --standalone --nproc_per_node=2 \
train_cloud_ddp.py --cpu --stop-after 1
```

Run this from the directory containing both downloaded Python files. For two visible CUDA GPUs, remove `--cpu`, choose `PRECISION=bf16` only when native BF16 is supported, and retain one process per GPU. `torchrun` sets RANK, WORLD_SIZE and LOCAL_RANK. The worker calls `torch.cuda.set_device(local_rank)` before constructing the model and uses `DDP(model, device_ids=[local_rank])`.

A CPU/Gloo run tests distributed control flow. It is not a substitute for a CUDA benchmark. Ordinary single-GPU Colab cannot demonstrate actual two-GPU NCCL execution; use CPU mechanics there or a properly provisioned two-GPU environment.

### Local file rendezvous alternative

If socket rendezvous fails on a local macOS CPU setup, launch two terminals with the same configured data/output variables and an **initially nonexistent shared file path**. In the first run `uv run train_cloud_ddp.py --cpu --init-file /tmp/cloud_ddp_unique --rank 0 --world-size 2`; in the second run the same command with `--rank 1`. Both processes must run concurrently and use the same fresh path. Use a new rendezvous path for the next launch. This is a local mechanics alternative, not a multi-node setup. The repository verifier starts and bounds both processes automatically.

## 4 · Reuse the data without duplicating training work

Every rank reconstructs the same manifest, then a `DistributedSampler` chooses its training shard. The DataLoader uses the sampler and therefore does not also request shuffle. Call `sampler.set_epoch(epoch)` before each epoch, so all ranks agree on a new shared shuffle before slicing it into shards.

The worker sets **sampler drop_last=True** and **DataLoader drop_last=True**. These are two distinct drops. The sampler makes the number of samples divisible by world size without padding duplicates. The loader makes each rank's microbatches full. The worker logs the actual image IDs used and the total omitted count each epoch.

For five training images and two ranks, sampler dropping uses four and omits one. Padding instead would make six positions and duplicate an image. Both choices have consequences. With larger batches the loader can omit additional examples. Never describe the selected dataset size as the number actually trained on without checking this.

The small four-image smoke selection yields three training images and one held-out image. With two ranks, use BATCH=1: two training images are used that epoch and one is omitted. This is only a code-path demonstration. The normal capped teaching run uses more images; neither setup establishes leaderboard quality.

## 5 · Accumulation and communication

For full windows the nominal effective batch is **microbatch × accumulation × world_size**. With B=2, K=3 and W=2, that is 12 images per update. A final short window has a smaller actual effective batch. Each local loss still needs its actual local window denominator.

Without special handling DDP can synchronize after every backward, even when the optimizer waits. `model.no_sync()` suppresses synchronization on intermediate microsteps; the final backward in the window synchronizes the accumulated gradient.

```python
for i, batch in enumerate(window):
    sync_context = nullcontext() if i == len(window)-1 else model.no_sync()
    with sync_context:             # includes FORWARD and BACKWARD
        with autocast_context():
            loss = per_image_loss(model(x), y).mean() * len(x) / N
        scaler.scale(loss).backward()
# unscale once, clip once, step/update once; schedule successful updates
```

The forward belongs inside no_sync as well as backward. All ranks must use the same window/synchronization pattern. This worker achieves that through equal-length full-batch loaders, not by assuming arbitrary uneven inputs will work.

AMP remains local execution policy on each rank. Synchronized gradients give replicas the same nonfinite situation under this matched setup. Each rank clips and steps after synchronization. The worker checks every parameter replica for exact agreement after the epoch and aborts if they diverge.

## 6 · Validation: reduce totals before dividing

Training samplers that pad are often unsuitable for exact validation because duplicate images bias totals. The worker instead gives rank r `validation_ids[r::world_size]`. These shards are disjoint and collectively cover each selected validation image once. A rank can legitimately have zero validation examples in a tiny smoke run.

Validation uses the underlying raw model in eval/no-grad mode, avoiding DDP forward collectives when ranks have different numbers of validation batches. Our model has no BatchNorm buffers; a model with mutable or unsynchronized buffers requires additional design. Do not copy this assumption blindly to another architecture.

Each rank computes sufficient totals: summed image loss, image count, summed foreground-pair Dice, foreground-pair count, correctly empty pairs, empty-pair count and per-class support. An all_reduce SUM combines those totals, then the code divides once.

If one rank has mean loss 1 over one image and another has mean loss 3 over three images, averaging rank means gives 2. The correct global image mean is (1+9)/4=2.5. The same principle applies to conditional metric denominators. Averaging local Dice means can be wrong when positive pair counts differ.

The worker asserts global validation image count equals the manifest's unique held-out count. That invariant is more informative than a plausible-looking metric alone.

## 7 · Rank-zero I/O is not rank-zero communication

Only rank zero writes shared last.pt, best.pt and history files. Every rank still participates in collectives in the same order. The worker gathers rank-specific RNG state **before** entering the rank-zero save branch and waits at a barrier after saving.

```python
all_rng = [None] * world_size
dist.all_gather_object(all_rng, local_rng)  # every rank calls it
if rank == 0:
    atomic_save(checkpoint, output / 'last.pt')
dist.barrier()                             # every rank calls it
```

Putting all_gather inside `if rank == 0` hangs because the other participants never arrive. A barrier does not repair a missing collective or a wrong collective order. Errors need to be diagnosed in every rank's logs.

Small object gathers are used for teaching checkpoint metadata here. Keep large tensors out of object collectives in real training, particularly with NCCL; use appropriate tensor collectives or scalable checkpointing instead.

## 8 · Distributed resume

The checkpoint stores the raw model state (avoiding the wrapper's `module.` prefix), optimizer, scheduler, scaler, complete progress/selection state, manifest, world size and per-rank RNG state. Every rank loads the same shared checkpoint from an accessible filesystem and restores its own RNG entry.

Resume requires the same configuration, data/split and world size. This is deliberate: changing world size changes partitioning and potentially effective batch and update count. Elastic recovery is a separate design. We demonstrate epoch-boundary interruption with the full schedule horizon fixed from the beginning, just as in the core notebook.

The notebook launches one epoch, resumes to the planned third epoch and checks the logged replica agreement and unique validation count. A replica check confirms synchronization consistency; it does not prove the objective, split or architecture is correct. Compare a distributed run to a single-process reference only after matching the examples actually used, their global update grouping and loss reductions. The dropping policy makes a naive comparison to the full single-process dataset unfair.

## 9 · Debug a hang systematically

First read the earliest exception from all ranks: one rank can fail while another waits in a collective. Check visible GPU count and local rank mapping, process-group initialization, equal training batch counts, no_sync boundaries and collective order. Then inspect data access on every process.

Use `TORCH_DISTRIBUTED_DEBUG=DETAIL` for diagnosis; NCCL-specific debugging is appropriate only for CUDA/NCCL. Keep the tiny CPU/Gloo check as a structural test, then run a separate hardware-specific acceptance check before relying on GPUs in class. Always destroy the process group in a finally block, as this worker does.

Do not “fix” a hang by removing synchronization or letting a rank skip backward when it dislikes a sample. That can silently diverge the training protocol. Uneven-input techniques such as join contexts exist, but require careful treatment of loss weighting and additional collectives; they are outside this worker's equal-work contract.

## 10 · Scaling is a measurement question

Two GPUs do not guarantee twice the speed. Communication, input throughput, small workloads and imbalance limit scaling. Measure synchronized end-to-end updates, compare a matched workload and report scaling efficiency only with real hardware measurements. CPU mechanics numbers do not establish GPU scaling.

Increasing world size with fixed local batch increases global batch. Keeping global batch fixed instead reduces per-rank work and may reduce compute efficiency. State which experiment you are running. Do not mix a strong-scaling claim with a different training budget.

## Practice ladder

1. Draw two replicas and trace one average-gradient update by hand.
2. Run **D1 · Inspect shards**. Find used and omitted real IDs; change the epoch.
3. Run **D2 · Launch** with CPU/Gloo. Read both rank logs.
4. Run **D3 · Resume**. Inspect unique validation count and replica agreement.
5. On verified two-GPU hardware, repeat with NCCL/FP32, then BF16 or FP16.
6. Add one controlled experiment: compare communication with and without no_sync, keeping effective batch and data fixed. Measure rather than assume a benefit.

<details><summary>Exit questions and expected reasoning</summary>

Why does DDP not fit a model that is too large for one GPU? Each rank still holds a complete replica and optimizer state.

Why not average validation means? Ranks can have unequal example or foreground-pair counts; sum numerators and denominators first.

Why call set_epoch? It changes the shared shuffle before deterministic rank sharding each epoch.

Why is a rank-zero-only all_reduce wrong? A collective requires all its participants in compatible order.

Why are training omissions reported? Dropping maintains equal work but changes which examples contribute that epoch.

</details>

References: [PyTorch DDP contract](https://docs.pytorch.org/docs/2.8/generated/torch.nn.parallel.DistributedDataParallel.html), [DistributedSampler and loaders](https://docs.pytorch.org/docs/2.8/data.html), [AMP with accumulation and multiple GPUs](https://docs.pytorch.org/docs/2.8/notes/amp_examples.html).
