# 5 · Save the training process, not only the weights

**Core · minutes 39–51, including the third notebook handoff.** Ask: “If I reload only a model after epoch one, have I resumed training?” Let learners name missing state before revealing the checkpoint contents.

## Weights answer an inference question

For inference you often need model configuration, learned weights and preprocessing/threshold metadata. To continue the same training process, you also need optimizer moments, the LR scheduler position, precision scaler state, progress counters and randomness state. A reset optimizer can make the next update different even when the starting weights are identical.

| Saved item | What would reset or become ambiguous without it? |
|---|---|
| Model state | Learned parameters and registered buffers |
| Optimizer state | AdamW moments and update counts |
| Scheduler state | Warmup/decay position and next LR |
| GradScaler state | FP16 scale and its growth tracking |
| Next epoch, best value, bad-epoch count | Progress, model selection and early-stopping history |
| Config and data manifest | Whether this is the same experiment and split |
| Torch, Python, CUDA RNG state | Stochastic continuation within the stated reproducibility scope |

Our files use `state_dict`s and primitive/tensor state, loaded with `weights_only=True`. Load checkpoints you trust; that option is not a reason to ignore the source of a downloaded file. The runtime compares the entire configuration and manifest before continuing, so changing the schedule horizon, output root or precision policy intentionally starts a different experiment rather than silently calling it exact resume.

## The LR schedule has its own clock

We use linear warmup followed by cosine decay to a fraction of the base LR. The schedule horizon is calculated from the planned epochs and update windows **before** training begins. Stopping after epoch one does not change a three-epoch plan into a one-epoch plan.

A scheduler is advanced after a successful optimizer step. This is neither every batch nor merely every epoch. PyTorch's scheduler initialization establishes the first LR; log the rate actually used for an update separately from the next rate recorded after stepping.

## Last and best have different jobs

`last.pt` captures the latest completed epoch and is the natural recovery point. `best.pt` captures the state at the lowest held-out loss in this lesson. The latest weights need not be the selected weights.

Early stopping watches validation loss with a configured patience. It is a compute-management choice, not evidence that the model has converged globally. Never select checkpoints on test-set performance. Our runtime records the monitor, best value and patience state, so recovery cannot accidentally restart patience.

## What exact resume means here

This lesson checkpoints at epoch boundaries, uses deterministic epoch-specific shuffle seeds, no augmentation and `num_workers=0`. Within the same CPU/runtime setup it compares a one-epoch interruption plus resume with an uninterrupted three-epoch run.

It does **not** promise exact arbitrary mid-epoch recovery. That would require the current sampler position and any partially accumulated gradients, plus worker/augmentation state. Cross-hardware, cross-version and nondeterministic CUDA operations can also prevent bitwise replay. The optional DDP worker adds per-rank randomness and a fixed world-size requirement.

Checkpoints are written to a temporary file then atomically replaced locally, reducing the chance of leaving a half-written last file after interruption. Remote storage and pre-emption require their own durability design.

## Handoff 3 · Prove continuation

[Open the exact notebook instructions →](08_notebooks.md#resume-demo)

Run **04 · Interrupt, resume**. Inspect the saved keys. Stop after the first completed epoch, reload last.pt under the same three-epoch configuration, then compare final model tensors with an uninterrupted reference. On CPU the notebook asserts closeness.

Then run **05 · Inspect schedule and validation**. Separate training loss, validation loss, next LR and checkpoint selection in the record. Ask a learner to find the line that keeps the scheduler still when FP16 skips an update.

<details><summary>Repair exercise: resume with epochs=1, then extend to epochs=3</summary>

Those runs used different schedule horizons from the beginning, so they are not an exact interruption test. Plan epochs=3 for both, use `stop_after=1` only to interrupt the first run, and resume with the unchanged configuration. Extending a completed schedule can be a valid new experiment, but document it as such.

</details>

**Exit check:** which file would you use after a crash, and which would you use for the selected validation result? Explain why the answers can differ.
