# Inference quick reference

## Recipe checklist

- Reconstruct the checkpoint's preprocessing, output meanings and activation.
- Select raw or EMA weights using the same held-out protocol.
- Set evaluation mode and disable gradient tracking.
- Predict validated views and invert spatial outputs.
- Blend aligned probabilities using recorded nonnegative normalized weights.
- Restore the native H128 × W800 canvas before decisions.
- Apply recorded per-class thresholds, then validated area filtering.
- Encode F-order one-based RLE and match the sample submission exactly.
- Freeze the selected recipe before final confirmation or test inference.

## Frequent errors

| Symptom | Likely error | First check |
| --- | --- | --- |
| Mirrored ghosts | Missing TTA inverse | Off-center marker |
| One class suppresses another | Wrong activation/head contract | Independent sigmoid versus checkpoint semantics |
| EMA looks unexpectedly poor | Wrong checkpoint or buffer policy | Saved EMA state and BatchNorm handling |
| Blend erases a clear defect | Dilution by a weak member | Per-image aligned maps and member weights |
| Small defects disappear | Threshold/area cutoff too aggressive | Per-class size bins and native resolution |
| Great local fold blend, weak confirmation | Training exposure or tuning overfit | Membership of every contributing model |
| Correct-looking mask, wrong submission | RLE/order/empty sentinel | Round trip and sample row identity |

## Official documentation and background

- [PyTorch evaluation mode](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.eval)
- [PyTorch inference mode](https://pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html)
- [PyTorch AveragedModel, SWA and EMA](https://pytorch.org/docs/stable/generated/torch.optim.swa_utils.AveragedModel.html)
- [PyTorch BatchNorm refresh](https://pytorch.org/docs/stable/generated/torch.optim.swa_utils.update_bn.html)
- [SWA paper: Averaging Weights Leads to Wider Optima and Better Generalization](https://arxiv.org/abs/1803.05407)
- [Competition evaluation description](https://www.kaggle.com/competitions/industrial-surface-defect-detection/overview/evaluation)
- [NumPy flatten order](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html)

These references explain APIs and underlying methods. They do not establish a guaranteed performance gain for this dataset. The session's synthetic examples, diagnostic metrics and evaluation counts are explicitly educational.
