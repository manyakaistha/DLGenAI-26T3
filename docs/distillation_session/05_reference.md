# Quick reference & further reading

## Technique comparison

| Technique | Main goal | Extra training data? | Inference cost | Main pitfall |
| --- | --- | --- | --- | --- |
| **Ensemble** | Higher accuracy & lower variance | No | $M \times$ slower | Heavy to deploy |
| **Distillation** | Compress ensemble into 1 fast student | No (uses OOF soft maps) | $1 \times$ (single model) | In-fold teacher leakage or overconfident teacher |
| **Pseudo-labeling** | Learn from unlabeled / target-domain images | Yes (unlabeled pool) | $1 \times$ (single model) | Confirmation bias & validation leakage |
| **Copy-paste** | Boost rare-class recall | Synthetic combinations | $1 \times$ (single model) | Visible paste seams / edge-halo shortcuts |
| **Calibration (Platt / $T$)** | Honest probabilities for blending & gating | No (fit on OOF) | Negligible | Fitting on training data instead of OOF |

## Key formulas at a glance

- **Temperature-scaled probability**:
  $$p_T = \sigma\!\left(\frac{z}{T}\right)$$
- **Distillation loss**:
  $$\mathcal{L} = \alpha \, \mathcal{L}_{\text{hard}}(y, p_s) + (1 - \alpha) \, T^2 \, \mathcal{L}_{\text{soft}}(p_{t,T}, p_{s,T})$$
- **Platt logit calibration**:
  $$p_{\text{cal}} = \sigma(a \cdot z + b)$$

## Practical starting defaults

- **Distillation**: $\alpha = 0.4$, $T \in [1.5, 3.0]$, OOF ensemble teacher.
- **Pseudo-labeling**: soft targets (or high-confidence gate), 20–30% unlabeled mix per batch, stronger student augmentations.
- **Calibration**: fit per-class $T$ or $(a, b)$ on OOF logits before blending or distilling.

## Further reading

- Hinton, Vinyals & Dean (2015), *Distilling the Knowledge in a Neural Network* — the original soft-targets & temperature-scaling paper.
- Xie et al. (2020), *Self-training with Noisy Student improves ImageNet classification* — why stronger student noise/augmentation makes self-training work.
- Ghiasi et al. (2021), *Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation* — clean analysis of copy-paste for rare objects.
- Guo et al. (2017), *On Calibration of Modern Neural Networks* — why modern nets are overconfident and why temperature scaling works so well.
