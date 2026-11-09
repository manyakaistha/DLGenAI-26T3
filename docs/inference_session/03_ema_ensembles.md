# EMA and prediction ensembles

## What is being averaged?

| Method | Averaged object | When created | Inference evaluations per view |
| --- | --- | --- | --- |
| EMA | Parameters along one training trajectory | During training | 1 |
| SWA | Selected compatible parameter snapshots | Usually late training | 1 |
| Checkpoint prediction ensemble | Output probabilities | During inference | Number of checkpoints |
| Model/fold prediction ensemble | Aligned outputs of separate models | During inference | Number of models |
| TTA | Aligned outputs of transformed inputs | During inference | Number of views per model |

EMA stands for **exponential moving average**. It is built during training and selected as a checkpoint for inference. It does not require several forward passes at inference. Keeping the ordinary and EMA models resident during training does use additional storage.

## EMA arithmetic

Initialize an EMA copy from the starting model, then update after each successful optimizer step:

$$\theta_t^{\mathrm{EMA}}=\beta\theta_{t-1}^{\mathrm{EMA}}+(1-\beta)\theta_t.$$

For an EMA scalar starting at `0`, new parameters `2, 4, 6`, and decay `0.5`, the successive EMA values are `1, 2.5, 4.25`. Larger decay means slower response. A rough memory scale is $1/(1-\beta)$ updates: decay 0.99 gives about 100 updates, and 0.999 gives about 1,000. This is an intuition, not a hard window or a default suitable for every training run.

```python
# ema_model was deep-copied from model before training.
# Call after a successful optimizer step, including AMP skip handling.
@torch.no_grad()
def update_ema_parameters(ema_model, model, beta):
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.mul_(beta).add_(p, alpha=1 - beta)
```

This fragment updates **parameters only**. A complete implementation must define its buffer policy, assert matching model structures, and save the EMA state. BatchNorm running means/variances are buffers; integer counters must not be averaged as floating parameters. Options include copying appropriate buffers from the current model or recomputing BatchNorm statistics on representative training data before evaluation. Document and validate the chosen policy. PyTorch's `AveragedModel` exposes buffer handling and `update_bn` supports refreshing BatchNorm statistics.

Evaluate raw and EMA checkpoints on the same held-out data. Smooth parameter trajectories do not guarantee better validation performance. Decay, warmup, optimizer-step frequency and training length all affect the result. A saved final checkpoint alone cannot reconstruct an EMA history that was never recorded.

## EMA versus SWA

SWA usually takes an arithmetic mean over selected late-training parameter states, often with a training schedule designed to explore useful solutions. EMA gives recent states exponentially greater weight. Both require compatible parameter identities. Averaging unrelated architectures is undefined, and arbitrary independent networks can have incompatible neuron permutations even when tensor shapes match.

For nonlinear networks, predictions from average weights generally differ from the average of their predictions:

$$f\left(x;\frac{\theta_1+\theta_2}{2}\right)\ne\frac{f(x;\theta_1)+f(x;\theta_2)}{2}.$$

## Probability ensembling

After TTA alignment for each model, combine compatible foreground maps:

$$p(x)=\sum_{m=1}^{M}w_m p_m(x),\quad w_m\ge0,\quad\sum_m w_m=1.$$

A toy pixel with predictions 0.2 and 0.8 and weights 0.25 and 0.75 produces 0.65. Threshold **after** blending. Equal weights form a useful baseline. A weaker model can dilute correct detections, and strongly correlated models may add little. Different architectures do not automatically yield useful diversity.

## Held-out evidence and folds

For a common holdout, every member used to score the blend must have excluded those image IDs during training. In K-fold OOF prediction, each image is predicted by the model that excluded its fold. Averaging all K models on every training image leaks training exposure into the evaluation. OOF predictions from independently cross-fitted model families can support a blend comparison, but they are not identical to the final K-model test-time ensemble. Keep an untouched confirmation set or outer split when possible.

**Notebook B:** calculate the EMA updates before execution, then compare prediction blending with parameter averaging in a nonlinear toy function.
