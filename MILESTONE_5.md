# Milestone 5 · Post-Processing and Ensembling

Read the linked study guides before answering. The guides explain the methods and provide separate practice examples; they do not contain solutions to these questions.

## Suggested readings

- [Probability maps to binary masks](docs/milestone_5_student_guide/topics/01_thresholding_and_probability_calibration.md): sigmoid probabilities, strict $p>T$ versus $p\ge T$, threshold choice and the score terms.
- [Connected components and area filtering](docs/milestone_5_student_guide/topics/02_connected_components_and_minimum_area_filtering.md): 4/8-connectivity, OpenCV component areas, filtering and pair-level recall ceilings.
- [Test-time augmentation](docs/milestone_5_student_guide/topics/03_test_time_augmentation_mechanics_and_inversion.md): tensor flip dimensions, inverse transforms and probability averaging.
- [Ensembling](docs/milestone_5_student_guide/topics/04_ensembling_models_and_checkpoints.md): soft averages, hard votes, weights and logit/probability averaging.
- [Inference budgets](docs/milestone_5_student_guide/topics/05_scalable_inference_architecture_and_memory.md): map memory, streaming accumulators, forward-pass counts and timing.
- [Submission contract](docs/milestone_5_student_guide/topics/06_submission_integrity_and_the_competition_contract.md): RLE and CSV preflight.
- [Inference ablations](docs/milestone_5_student_guide/topics/07_inference_experimentation_discipline_and_ablation_ladders.md): validation discipline and final pipeline ordering.

Use each question's stated threshold convention and requested precision. Unless a question says otherwise, the competition score is

$$S=0.70\,\mathrm{ForegroundDice}+0.20\,\mathrm{FalsePositiveScore}+0.10\,\mathrm{EmptyAccuracy}.$$

For the explicitly defined empty-pair convention in Question 4, $\mathrm{FalsePositiveScore}=1-\operatorname{mean}(|P|/102{,}400)$ over empty pairs, and EmptyAccuracy is the fraction of empty pairs whose prediction contains no pixels.

---

## Question 1 · Logits, Probability Thresholds, and the Strict-Inequality Trap

A model's final layer outputs raw logits (no sigmoid applied yet) for six pixels of one image-class pair:

| Pixel | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| Logit | −2.0 | −0.9 | −0.5 | 0.0 | 0.8 | 2.2 |
| Truth | 0 | 0 | 1 | 1 | 1 | 0 |

1. Which logit corresponds exactly to probability threshold $T=0.3$? Give 4 decimal places.
2. At $T=0.5$, how many pixels are predicted positive using strict $p>T$, and how many using $p\ge T$?
3. What are Dice at $T=0.3$ (strict) and $T=0.5$ (strict)? Give 4 decimal places.

**Answer format:** `logit_T0.3, n_strict, n_geq, Dice0.3, Dice0.5`.

---

## Question 2 · Per-Class Threshold and Minimum-Area Selection from a Sweep

For `defect_b`, a student sweeps three post-processing configurations on the holdout and records:

| Config | $T$ | $N$ (minimum area) | FgDice | FalsePositiveScore | EmptyAccuracy |
|---|---:|---:|---:|---:|---:|
| X | 0.5 | 0 | 0.62 | 0.9950 | 0.40 |
| Y | 0.7 | 100 | 0.58 | 0.9990 | 0.85 |
| Z | 0.9 | 300 | 0.45 | 0.9998 | 0.97 |

Compute the official score to 4 decimal places for each configuration. Which configuration should the student choose, and which two score components explain why it beats the configuration with the highest FgDice?

**Answer format:** `Score_X, Score_Y, Score_Z, best_config, components`.

---

## Question 3 · Connected Components, Connectivity, and Area Filtering

A predicted binary mask on a $6\times8$ canvas (1 = predicted defect) is:

```text
1 1 0 0 0 0 0 1
1 1 0 0 0 0 0 0
0 0 1 0 0 0 0 0
0 0 0 1 0 0 0 0
0 0 0 0 0 1 1 0
1 0 0 0 0 1 1 1
```

1. How many connected components are there under 8-connectivity? How many under 4-connectivity?
2. Using 8-connectivity, discard every component with area $<5$. How many foreground pixels remain?
3. Using 8-connectivity, discard every component with area $<6$. How many foreground pixels remain?

**Answer format:** `n_8conn, n_4conn, pixels_N5, pixels_N6`.

---

## Question 4 · Does Post-Processing Improve the Official Score?

A validation set has 100 image-class pairs: 20 positive and 80 empty ($|Y|=0$). The canvas is $128\times800=102{,}400$ pixels. Before post-processing:

- Positive pairs: 12 pairs have Dice 0.70 and 8 pairs have Dice 0.45.
- Empty pairs: 50 are predicted completely empty, 20 contain one 20-pixel speckle each, and 10 contain one 2,000-pixel blob each.

Apply a minimum-area filter with $N=50$ (discard components with area $<50$). It removes the 20 speckles and keeps the 2,000-pixel blobs. It also deletes the only predicted component (30 pixels) of one positive pair whose Dice was 0.45, so that pair's Dice becomes 0.

Use the score formula above. After filtering, compute ForegroundDice, FalsePositiveScore, and EmptyAccuracy. Then give the official score before and after filtering, to 4 decimal places.

**Answer format:** `FgDice_after, FP_after, EmptyAcc_after, Score_before, Score_after`.

---

## Question 5 · Recall Ceiling of Minimum-Area Filtering at the Pair Level

Decode every non-empty mask in `train.csv` and run 8-connected component analysis on each image-class pair. A positive pair is “wiped out” by a filter threshold $N$ if every one of its connected components has area $<N$; after filtering, that pair's prediction would be empty even if its pre-filter prediction matched those ground-truth components perfectly.

1. With $N=100$, how many positive image-class pairs (across all four classes) are wiped out?
2. Which defect class loses the most pairs?

**Answer format:** `integer, defect name`.

---

## Question 6 · Test-Time Augmentation: Flip Axes and the “Forgot to Un-flip” Bug

Input tensors have shape $(B,C,H,W)=(B,1,128,800)$. For one $2\times2$ patch of a single class, the model produced:

- Original pass: `[[0.9, 0.6], [0.4, 0.2]]`
- Horizontal-flip pass (raw output, in the flipped frame): `[[0.5, 0.8], [0.3, 0.4]]`
- Vertical-flip pass (raw output, in the flipped frame): `[[0.1, 0.5], [0.7, 0.9]]`

1. Which `dims` list does `torch.flip` need for a horizontal flip and for a vertical flip on this tensor layout?
2. After correctly inverse-flipping each output and averaging the three passes, what is the averaged probability at the top-right pixel? Give 4 decimal places.
3. At $T=0.5$ (strict $>$), how many pixels in the $2\times2$ patch are positive with the correct pipeline?
4. How many are positive if the student forgets to inverse-flip the two augmented outputs before averaging?

**Answer format:** `hflip_dim, vflip_dim, top-right_value, n_correct, n_buggy`.

---

## Question 7 · Ensembling: Soft Averaging, Hard Voting, and Where to Average

Three models predict probabilities for five pixels of one image-class pair. Ground truth is positive at $p1,p3,p5$ and negative at $p2,p4$.

| Pixel | Model A | Model B | Model C |
|---|---:|---:|---:|
| p1 | 0.90 | 0.40 | 0.45 |
| p2 | 0.55 | 0.55 | 0.20 |
| p3 | 0.80 | 0.70 | 0.90 |
| p4 | 0.10 | 0.20 | 0.30 |
| p5 | 0.60 | 0.45 | 0.55 |

Use $T=0.5$ (strict $>$) throughout.

1. What is Dice of the soft ensemble (mean probabilities, then threshold)?
2. What is Dice of the hard-vote ensemble (threshold each model, then majority of 3)?
3. For weights $(A,B,C)=(0.5,0.3,0.2)$, what is the ensemble probability at $p1$?
4. Two snapshots output logits +1 and −3 for the same pixel. What are the final probabilities when averaging in probability space versus averaging logits first (4 d.p.)? At $T=0.30$, which method calls the pixel positive?

**Answer format:** `Dice_soft, Dice_vote, weighted_p1, prob_avg, logit_avg, method`.

---

## Question 8 · Inference Budget: Memory and Time Across Two Ensemble Configurations

The hidden test set has 5,000 images, each $128\times800$, with 4 output classes. One forward pass of one model on one image takes 12 ms. Use decimal units ($1\,\mathrm{GB}=10^9$ bytes). Compare:

- **Config A:** 3 models, 3-view TTA (original, horizontal flip, vertical flip).
- **Config B:** 2 models, 2-view TTA (original, horizontal flip).

1. How many GB does one full test-set set of probability maps (5,000 images × 4 classes) take as float32, float16, and uint8?
2. If every model/view prediction set is kept in memory as float32, how many GB do Config A and Config B require? Does either fit in a 16 GB RAM budget (ignore all other RAM use)? What if only a single running float32 accumulator is kept?
3. How many total forward passes and how many minutes of pure model time does each configuration need?
4. The notebook limit is 30 minutes, and data loading, RLE encoding, and I/O take 8 minutes. What is the maximum number of whole models that can be ensembled with 3-view TTA? With 2-view TTA?

**Answer format:** `GB_f32, GB_f16, GB_u8, GB_A_naive, GB_B_naive, fits (A/B/both/neither), accumulator_GB, passes_A, passes_B, min_A, min_B, max_models_3view, max_models_2view`.

---

## Question 9 · Designing the Final Inference Notebook

You are ready to submit an ensemble of two or more checkpoints with flip TTA, per-class thresholds, and per-class minimum-area filters. Use descriptive bullet points.

1. List the correct order of pipeline steps from raw image to RLE string, and explain why swapping any two adjacent steps could be a bug (for example, thresholding before averaging).
2. Name two ways to avoid overfitting your thresholds and minimum-area values to a single holdout. Explain how the all-empty baseline from Milestone 1 Question 9 relates to this risk.
3. How should an entirely empty final mask be written in the submission file? Name two things to check before pressing “Submit” to avoid a failed run under time or memory limits.

**Answer format:** Bullet points covering pipeline order, justification for the order, and a safety checklist.
