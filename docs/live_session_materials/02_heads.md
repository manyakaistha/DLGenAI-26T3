# 2 · Decide what a pixel can mean

Examples, masks, statistics and sensor dimensions on this page are synthetic teaching examples.
**Guide 2.** Multiple labels in an image and multiple labels at a pixel are separate questions.

## Co-occurrence is not overlap

A strip can contain A on the left and C on the right. Its image label is `[1,0,1,0]`, yet every pixel can still have at most one defect class. Stack four decoded binary masks and sum along the class axis: values above one reveal pixel overlap.

<div class="lab" data-lab="overlap"></div>

```python
# masks: [4, H, W], binary and registered to the same image
occupancy = masks.sum(axis=0)
overlap_pixels = (occupancy > 1).sum()
```

The class masks in a dataset may be **disjoint** (each pixel belongs to at most one defect class) or **overlapping** (a pixel belongs to two or more classes). The presence of A and C in one image does not tell you which case applies. **Conduct your own annotation audit:** decode the masks at their original size, align them to the same image, and count pixels whose occupancy is greater than one. Check every image with multiple positive class rows, and report both the number of overlapping pixels and the number of images containing overlap. Single-class images cannot have cross-class overlap. Use your findings to decide whether a single class label per pixel can represent the annotations.

## Two masks through the overlap audit

![Exclusive masks, overlapping masks, occupancy sums and legal target maps](figures/overlap_targets.png)

For two binary masks A and C, `A & C` identifies overlap. For all four classes, `masks.sum(axis=0) > 1` identifies pixels with at least two labels. Count **overlapping pixels** separately from **images containing any overlap**. Report how many multi-label images were examined; a zero overlap result on ten examples is weaker evidence than an exhaustive annotation audit.

For exclusive masks, the background channel is `1 - union_of_defects`. A clean pixel becomes integer label 0, A becomes 1, B becomes 2, C becomes 3, D becomes 4. These are category identifiers: class 4 is not four times as defective as class 1.

```python
# Four native binary masks, with classes in a fixed A/B/C/D order.
assert ((masks == 0) | (masks == 1)).all()
assert (masks.sum(axis=0) <= 1).all()
integer_target = np.zeros(masks.shape[1:], dtype=np.int64)
for class_id, mask in enumerate(masks, start=1):
    integer_target[mask.astype(bool)] = class_id
```

## From logits to probabilities

A segmentation model produces a score for each class at each pixel. These raw scores are called **logits**. A logit can be negative, zero or positive; it is not yet a probability. An activation function converts the scores into numbers between 0 and 1. Sigmoid and softmax do this in different ways, so they express different assumptions about the labels.

### Sigmoid: a separate yes/no question for each class

For each defect class $k$, sigmoid asks, “Does this pixel belong to this class?” It converts that class’s logit $z_k$ into a probability-like score $p_k$:

$$
p_k = \sigma(z_k) = \frac{1}{1 + e^{-z_k}}
$$

Here $e$ is the base of the exponential function. A logit of $0$ becomes $0.5$; a logit of $2$ becomes approximately $0.881$; a logit of $-2$ becomes approximately $0.119$. Larger logits therefore mean stronger predicted presence of that class.

Each conversion uses **only that class’s logit**. If you change A’s logit while holding the other logits fixed, only A’s sigmoid score changes. The four scores do not need to add up to 1. For example, A can score $0.881$ and B can score $0.731$ at the same pixel. With a threshold of $0.5$, both are selected. If all four scores are below their thresholds, no defect is selected and the pixel is treated as background. The threshold is a decision rule to evaluate on validation data, rather than a fact established by the activation function.

### Softmax: choose among background and the four defects

Softmax asks, “Which one of these categories does this pixel belong to?” In this example the choices are background, A, B, C and D. It reads all five logits together and divides each exponentiated score by their shared total:

$$
p_k = \frac{e^{z_k}}{\sum_{j=0}^{4} e^{z_j}},
\qquad \sum_{k=0}^{4} p_k = 1
$$

The index $k$ identifies one category, and $j$ runs over all five categories in the denominator. The exponential makes every contribution positive; dividing by the total makes the probabilities add up to 1. You can think of the categories as sharing a fixed probability budget. Increase A’s logit while holding the others fixed, and A takes a larger share while every other share decreases.

To turn these probabilities into a class label, **argmax** selects the category with the largest probability. For logits $[0,2,1,-1,-2]$ in background/A/B/C/D order, the probabilities are approximately $[0.086,0.636,0.234,0.032,0.012]$. A has the largest share, so this pixel is assigned A alone. Background needs its own category so the model can also choose “no defect.”

### What to try in the chart

The chart below represents **one pixel**, not a whole image. Each slider sets one raw logit. The left panel applies four independent sigmoids to A–D; the right panel applies softmax to all five categories, including background.

1. Raise only A’s slider. On the left, only A’s bar changes. On the right, A’s bar increases while the other bars decrease.
2. Raise both A and B above zero. Both sigmoid scores exceed $0.5$, so both classes can be selected. Softmax still assigns a single class when you take argmax.
3. Raise the background slider. The sigmoid panel stays unchanged because it has no background channel. In the softmax panel, background takes probability away from the defects and can become the chosen label.

Watch the totals beneath the chart: softmax always sums to 1, while the sum of the sigmoid scores can vary. Both methods produce model scores; neither automatically guarantees calibrated confidence. For example, a score of $0.8$ does not by itself establish that 80% of similarly scored predictions will be correct.

<div class="lab" data-lab="heads"></div>

## Tensor and target choices

| Design | Output logits | Training target | Typical loss | Binary decision |
|---|---|---|---|---|
| Four sigmoids | B × 4 × H × W | Four binary masks, same shape | BCEWithLogitsLoss | Independent class thresholds |
| Five-way softmax | B × 5 × H × W | B × H × W integer labels 0–4 | CrossEntropyLoss | Background-inclusive argmax |

The head, target format and loss have to agree:

- **Four sigmoid channels + `BCEWithLogitsLoss`:** use this when each class is an independent yes/no label and pixels may carry more than one class. The model returns logits shaped `[B, 4, H, W]`; the target is a floating point stack of four binary masks with the same shape. The loss compares each class decision independently.
- **Five softmax channels + `CrossEntropyLoss`:** use this when each pixel has exactly one category: background, A, B, C or D. The model returns logits shaped `[B, 5, H, W]`; the target is an integer map shaped `[B, H, W]` whose values are class IDs 0–4. The loss trains the categories to compete for that one label.

In both cases, give the loss the **raw logits**. `BCEWithLogitsLoss` and `CrossEntropyLoss` apply numerically stable sigmoid and softmax calculations internally. Applying those activations first would be the wrong input and can make training less stable. For softmax targets, first confirm masks do not overlap: converting overlapping masks to one integer ID (often with `argmax`) silently drops one of the labels.

A sigmoid head can represent disjoint labels too, but its independent outputs can select several classes at one pixel. You then need a clear decision rule, such as validated per-class thresholds or a documented tie-breaking policy. A softmax head already enforces one winner, including the option of background.

## What the constraint does and does not buy

A softmax head followed by argmax produces **exactly one category per pixel**. This is a useful constraint when the audited annotations assign each pixel to either background or one defect class: the output has the same structure as the target. The constraint only controls which labels can be represented. A model can still choose the wrong class, place a boundary incorrectly, or miss a rare defect. Measure those errors on held-out images.

If the annotations allow A and C at the same pixel, one integer class ID cannot preserve both labels. A standard softmax/argmax design over background and the four individual defects would discard one of them. Independent sigmoid outputs with separate binary masks can represent that overlap, because each class has its own yes/no decision. They can also be used on disjoint data, but they may predict competing labels at one pixel; decide how to handle those predictions and validate that rule.

Choose the output representation from what the **pixel annotations** permit, then compare performance on validation data. Seeing two defect types somewhere in the same photograph is insufficient to make that choice.

**Check:** Why does a mutually exclusive head need five channels here? Background must be a learnable category; four defect-only softmax channels would force even pristine pixels into a defect category.

## One pixel, two probability systems

Take A logit=2 and B logit=1. Their independent sigmoid probabilities are approximately 0.881 and 0.731. Both pass a 0.5 threshold. With five logits `[0,2,1,-1,-2]`, the softmax vector is approximately `[0.086,0.637,0.234,0.032,0.012]`; its entries sum to 1, and A wins argmax. Raising B’s logit increases its softmax probability and decreases the other probabilities through the shared denominator.

![Output tensor shapes and corresponding targets](figures/head_tensors.png)

### Compare the two designs

After your overlap audit, train two small models with the **same data split, backbone, preprocessing, random seed and training budget**. Change the output head, target encoding and matching loss together:

1. Train four output channels with sigmoid-style binary masks and `BCEWithLogitsLoss`.
2. Train five output channels with background-inclusive integer labels and `CrossEntropyLoss`.
3. Use the same held-out images for both. Compare per-class Dice and IoU, foreground recall, and inference cost. Report per-class results as well as an average so rare classes do not disappear inside a high pixel-accuracy score.
4. For sigmoid, choose each class threshold using validation data only. For softmax, use background-inclusive argmax. State these decision rules when reporting results.

If your audit confirms disjoint masks, this is a direct comparison of two valid ways to model the same labels. If it finds overlap, the standard softmax target cannot preserve those pixels. Keep the overlap for the sigmoid experiment and explain the limitation; only compare softmax after stating how you handled overlapping pixels, since that choice changes the target task. Do not erase labels silently to make the comparison run.

**Implementation check:** BCE targets are floating binary masks with the same shape as logits. Cross-entropy targets are integer class IDs and omit the channel dimension. Apply `sigmoid` or `softmax` when inspecting predictions, while feeding raw logits into the corresponding stable loss. A one-channel softmax is always 1; it cannot represent a foreground/background choice.

<noscript>Interactive controls require JavaScript. The numbered figures, worked examples and code on this page provide the same concepts for static reading.</noscript>
