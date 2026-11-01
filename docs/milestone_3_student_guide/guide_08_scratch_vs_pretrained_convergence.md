# Scratch Baseline vs. Pretrained Transfer Learning
*(Milestone 3 · Question 8 Study Guide)*

## 1. Ask a Question the Experiment Can Answer

Pretraining can supply useful visual features, but a domain shift, different normalization or a short training budget can change the outcome. Faster convergence, a smaller generalization gap and a higher score are **hypotheses to measure**. There is no predetermined winner or score-improvement band.

For Q8, compare the **same SMP ResNet-34 U-Net architecture** with random versus ImageNet encoder initialization. This isolates encoder initialization; comparing a different basic U-Net from Milestone 2 would also change architecture and would not isolate pretraining.

A useful experiment matrix has one row per run and columns for initialization, backbone, split, seed, loss, training settings and evaluation rules. Mark the one column allowed to change before training. This prevents accidental changes such as freezing only one model or applying a threshold sweep to only one result.

## 2. Fix the Protocol Before Looking at Results

Use the [assessment protocol](../../MILESTONE_3.md#3-common-training-protocol-for-questions-8-9): the same saved split, image preprocessing, augmentation policy, sample order, batch size, optimizer, epochs and metric. The assessment fixes native single-channel [0,1] input, no augmentations, full precision, all parameters trainable, normal training-mode BatchNorm, and threshold 0.5 with no area filter. Q2's frozen encoder is a parameter-count exercise, not the training configuration for Q8.

```python
import torch
import segmentation_models_pytorch as smp

def make_model(weights):
    torch.manual_seed(42)
    return smp.Unet(
        encoder_name="resnet34", encoder_weights=weights,
        encoder_depth=5, decoder_channels=(256, 128, 64, 32, 16),
        decoder_use_norm="batchnorm", decoder_interpolation="nearest",
        in_channels=1, classes=4, activation=None,
    )

scratch = make_model(None)
pretrained = make_model("imagenet")
# Match decoder/head initialization explicitly to remove another source of variation.
pretrained.decoder.load_state_dict(scratch.decoder.state_dict())
pretrained.segmentation_head.load_state_dict(scratch.segmentation_head.state_dict())
```

Train one model at a time with a fresh optimizer. Reset the data-order generator and RNGs before each run. Do not reuse optimizer state or carry forward the Q2 freeze. Record library versions and the saved split; GPU kernels can still introduce small run-to-run differences.

## 3. Use the Official Weighted Metric

Follow [Milestone 1 Q8's definitions](../../MILESTONE_1.md#question-8-official-competition-weighted-metric-toy-batch-calculation):
$$Score=0.70\,ForegroundDice+0.20\,FalsePositiveScore+0.10\,EmptyAccuracy.$$

- **ForegroundDice:** mean binary Dice over image-class pairs with a nonempty target.
- **FalsePositiveScore:** one minus mean predicted-positive pixel fraction over ground-truth empty image-class pairs.
- **EmptyAccuracy:** fraction of ground-truth empty pairs predicted completely empty.

Accumulate these components over the **whole validation set**, not equally over uneven batches. Check that the split contains positive and empty pairs; report a missing denominator rather than inventing a score. Do not substitute average Dice over all pairs or Soft Dice training loss for this weighted metric.

Use sigmoid followed by `>= 0.5` on every channel, with no connected-component filtering for this comparison. Compute the official score and foreground Dice from those binary masks.

## 4. Read Learning Curves without Predicting Their Outcome

At epochs 1 and 5 record foreground Dice, official score and validation loss. A higher score and lower loss need not select the same model: they measure different properties. Report the two winners separately, including ties.

For the generalization comparison, evaluate **both the training split and validation split in eval mode** after the epoch, without augmentation. Average loss using the same fixed batch size and sample-weighted batch aggregation on both splits. This avoids comparing online training losses (collected while weights change and BatchNorm uses batch statistics) directly with an eval-mode loss. The Dice term is batch-coupled, so record the batching as well.

Compute:
$$G=|L_{val,eval}-L_{train,eval}|.$$

A smaller gap alone does not prove a better model: an underfit model can be equally poor on both sets. Read the gap alongside validation quality.

## 5. Worked Practice on Fictional Logs

Suppose a toy experiment gives model R score 0.41 and model S score 0.40. Then $\Delta Score=0.01$ for R minus S. If R's train/validation eval losses are 0.30/0.34, its gap is 0.04. If S's are 0.33/0.35, its gap is 0.02. R has the better score but S has the tighter gap. These invented logs do not predict the assessment's outcome.

## 6. Guiding Hints and Assignment Task

Read [Question 8](../../MILESTONE_3.md#question-8-scratch-baseline-vs-pretrained-transfer-learning-empirical-training-convergence). Keep a small per-epoch table containing model, seed, split ID, eval losses, foreground Dice and all three score components. At epoch 5 compute pretrained score minus scratch score, then choose the exhaustive band listed in the question.

Report the measured loss winner, score winner, epoch-1 foreground-Dice winner, gap comparison and signed score difference. Unexpected results are valid when your setup and measurements are documented. Do not select an answer based on a theoretical expectation.
