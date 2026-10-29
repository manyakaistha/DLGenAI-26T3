# 6 · Efficiency needs a quality check

**Core · minutes 51–57.** Put this claim on screen: “AMP saved memory, therefore the experiment succeeded.” Ask learners what evidence is still missing.

## Keep three kinds of evidence together

Record resource use, numerical health and held-out quality. Resource use includes measured time and peak allocated CUDA tensor memory under a declared workload. Numerical health includes finite loss/gradients and skipped updates. Held-out quality includes a fixed validation protocol and sufficient class support.

A fast run with every FP16 step skipped is not useful progress. A finite training loss with all-background predictions can still fail segmentation. A promising validation value based on one image cannot support the same conclusion as a representative holdout.

## Read our diagnostic definitions

The notebook evaluates raw logits after sigmoid and a fixed threshold of 0.5 on the teaching grid H64×W96. It uses `eval()` and no gradients. It does not tune the threshold against the test set.

**Validation loss** is a mean over images of our per-image BCE plus soft Dice objective. It is the checkpoint-selection monitor in this lesson.

**Foreground-pair Dice** averages hard Dice only over image–class pairs with at least one positive target pixel. For each such pair: 2TP/(2TP+FP+FN). This makes performance on annotated foreground visible instead of hiding it behind empty pairs.

**Empty-pair specificity** here means the fraction of empty-target image–class pairs for which the entire predicted mask is also empty. It is an image–class diagnostic, not pixel-level specificity TN/(TN+FP). State that definition when reporting it.

**Class support** counts positive image–class pairs per category. A class with no positive targets cannot establish positive-region quality. An absent denominator is represented as `null`, never as a perfect score.

## Why empty masks need an explicit policy

A target may be empty for Fish while Flower is present in the same image. Predicting no Fish pixels is appropriate for that pair, but says nothing about Flower localization. Counting all empty pairs as Dice=1 can dominate a pooled mean when foreground is rare. Conversely, dropping empty targets hides false-positive behaviour. Separate diagnostics expose both questions.

Consider three target-positive pairs with Dice .2, .4 and .6, plus seven empty pairs. Foreground mean is .4. If six empty predictions are correct, empty-pair specificity is 6/7. Giving those six a perfect Dice and the false positive a zero would produce an overall (1.2+6)/10=.72. That number answers a different question and looks much more optimistic. These are invented teaching numbers, not cloud results.

## Comparing experiments fairly

Hold fixed the same real image IDs, preprocessing, model start, effective batch, training budget, threshold and reduction. For a stronger study, repeat across seeds and use sufficiently representative validation. A timing comparison can be local and useful while a quality comparison remains inconclusive.

Write a compact experiment record:

```text
Question: does BF16 reduce peak allocated CUDA memory here?
Fixed: image IDs, model seed, B×K, updates, resize, loss, validation
Changed: precision policy only
Measured: hardware + runtime + warmup + timed steps + allocated peak
Quality: held-out loss, foreground Dice, empty-pair specificity, support
Decision: adopt / investigate / insufficient evidence, with reason
```

## Interactive discussion

A student reports foreground Dice improved from .40 to .44, but changed the split and doubled effective batch. What can you conclude?

<details><summary>Suggested response</summary>

The new configuration produced a different observed value. You cannot attribute that change to precision or any single training choice. Repeat with fixed evaluation and one controlled change. Inspect positive-class support and uncertainty before claiming a reliable improvement.

</details>

## The last three minutes

Ask each learner to finish these sentences:

1. FP16 has finer spacing near one, while BF16 has ____.
2. A five-image window split 2+2+1 needs loss weights ____.
3. Loading only model weights resets ____.
4. My next Milestone 4 experiment changes ____ and holds ____ fixed.

DDP is available as a separate extension. It adds communication and data-sharding obligations; it is not a prerequisite for today's single-device experiment.
