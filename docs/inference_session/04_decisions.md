# Thresholds and connected components

## A threshold is a decision rule

A sigmoid score is a number between zero and one. It is not proof that the score is statistically calibrated. **Threshold tuning** chooses a decision boundary for an objective. **Probability calibration** studies agreement between predicted confidence and observed frequencies. These are related but different tasks.

For class $c$, predict foreground when $p_c\ge\tau_c$. State whether equality counts. This lesson uses `>=`. Lowering a threshold tends to add pixels and raise recall while risking more false positives. Raising it tends to remove pixels while risking missing weak or small defects.

A default of 0.5 is a reference, not a guaranteed optimum. Tune per-class thresholds only with enough held-out support. A large search on rare classes can overfit a few examples.

## A small worked example

Four pixels have probabilities `[0.8, 0.6, 0.4, 0.2]` and target `[1,1,0,0]`.

| Threshold | Predicted mask | TP / FP / FN | Dice |
| --- | --- | --- | --- |
| 0.3 | 1110 | 2 / 1 / 0 | 0.8 |
| 0.5 | 1100 | 2 / 0 / 0 | 1.0 |
| 0.7 | 1000 | 1 / 0 / 1 | 2/3 |

These values describe one synthetic positive mask, not a threshold recommendation for the competition. Thresholds must be tuned **after** the final aggregation rule is chosen because TTA and blending change the probability distribution.

## Connected components and area

Threshold first. Then label spatially connected foreground regions separately for each image and class. Under 4-connectivity, pixels connect by edges. Under 8-connectivity, diagonal neighbors also connect. Connectivity affects component sizes and therefore the filter's result.

Keep components with `area >= A_min` in this lesson. If `A_min=3`, a two-pixel component disappears while a five-pixel component remains. This may remove noise, but it will also erase a genuine two-pixel defect. Component filtering cannot know which small regions are real.

Whole-mask area gating is a different rule: it measures the total predicted area and either keeps or removes the entire image–class mask. Name and validate the actual rule implemented.

## Resolution matters

An area threshold is in pixels on a specified grid. Apply it on the intended native canvas after restoring probability-map size. When both linear dimensions are halved, the same physical area contains roughly one quarter of the pixels. Blindly reusing the same area cutoff changes the decision rule.

## A manageable search

Start with fixed thresholds and no filtering. Try a small threshold grid on validation caches, then a small area grid. Inspect positive-pair Dice, empty-target false alarms and class support together. Joint tuning can help, but a larger search requires stronger confirmation. Keep the confirmation set untouched while selecting the recipe.

**Notebook C:** identify the threshold that fits the four-pixel example, then inspect what disappears at area 3. Explain why a cleaner-looking mask can have worse recall.
