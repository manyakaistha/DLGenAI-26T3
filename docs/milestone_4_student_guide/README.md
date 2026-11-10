# Milestone 4 Student Study Guides
## Training Improvements & Model Optimization

Milestone 4 starts from the segmentation pipeline built in Milestone 2 and the model/loss alternatives studied in Milestone 3. It asks how to make training more deliberate: measure imbalance at the right level, control optimization schedules, save the checkpoint that matches the task metric, fit work within a memory budget, and diagnose validation errors before changing the model.

The industrial input remains a native `128 × 800` grayscale strip. Targets are four independent binary channels in `defect_a` through `defect_d` order; pixels may have multiple or no labels. Do not resize to square images or infer class prevalence from examples in these guides.

## Assessment map

The [Milestone 4 assessment](../../MILESTONE_4.md) defines the exact data subset, split, training settings, and requested fields. Read it before coding. These guides explain the methods and use separate practice data; they do not provide assessed counts, trace outcomes, or measured run scores.

| Assessment | Read first | Method to learn |
|---|---|---|
| Q1 · image/class counts | [Guide 01](topics/01_multilevel_class_imbalance.md) | Count clean images, per-class positives, and exact label combinations with distinct denominators. |
| Q2 · pixel prevalence | [Guide 01](topics/01_multilevel_class_imbalance.md) | Decode a class mask, divide by all image-canvas pixels, and compute positive-image mean area separately. |
| Q3 · bounded sampling | [Guide 01](topics/01_multilevel_class_imbalance.md) | Apply combination-group weights per image, then normalize group mass as `n_g × w_g`. |
| Q4 · warmup/cosine | [Guide 02](topics/02_learning_rate_schedulers_and_warmup.md) | Evaluate a step-indexed multiplier and match scheduler cadence to optimizer updates. |
| Q5 · early stopping | [Guide 03](topics/03_early_stopping_and_checkpoint_governance.md) | Apply strict `min_delta`, patience, and best-checkpoint restoration epoch by epoch. |
| Q6 · AMP and accumulation | [Guide 04](topics/04_mixed_precision_amp_and_gpu_memory.md) | Separate persistent optimizer state from activation memory; count optimizer updates, not microbatches. |
| Q7 · empty-pair metric | [Guide 05](topics/05_validation_dice_and_overfitting_diagnostics.md) | Keep foreground Dice, empty-pair accuracy, and false-positive pixel score distinct. |
| Q8 · curves and checkpoint | [Guides 02, 03, 05](topics/02_learning_rate_schedulers_and_warmup.md) | Log each epoch and select against the declared official validation score. |
| Q9 · size diagnostics | [Guide 06](topics/06_validation_error_analysis_and_spatial_diagnostics.md) | Bin positive image-class pairs by mask area; distinguish pair counts from pixel totals. |
| Q10 · OFAT | [Guide 07](topics/07_controlled_experimentation_and_iteration.md) | Change one factor at a time, estimate the stated seed-noise band, and inspect clean-pair false alarms. |

## Study guide sequence

1. **[Multi-level class imbalance](topics/01_multilevel_class_imbalance.md)** — image-level clean/positive status, overlapping per-class image counts, pixel occupancy, combination groups, sampling versus loss weighting, and the unchanged-validation rule.
2. **[Learning-rate schedules and warmup](topics/02_learning_rate_schedulers_and_warmup.md)** — what a learning rate changes, warmup/cosine and other schedules, update cadence, and safe logging.
3. **[Early stopping and checkpoint governance](topics/03_early_stopping_and_checkpoint_governance.md)** — monitor choice, strict minimum change, patience counting, rollback, and limits of repeated holdout tuning.
4. **[Mixed precision and GPU memory](topics/04_mixed_precision_amp_and_gpu_memory.md)** — FP32 versus autocast activation storage, gradient scaling, persistent AdamW state, accumulation, and optimizer-step accounting.
5. **[Validation Dice and overfitting diagnostics](topics/05_validation_dice_and_overfitting_diagnostics.md)** — empty-mask conventions, the naive-Dice failure, the official metric components, and learning-curve reading.
6. **[Validation error analysis](topics/06_validation_error_analysis_and_spatial_diagnostics.md)** — aligned image/mask/probability/prediction views, spatial error patterns, and size-stratified diagnostics.
7. **[Controlled experimentation](topics/07_controlled_experimentation_and_iteration.md)** — invariant baselines, OFAT comparisons, seed noise, ablations, and evidence logs.

## Practice resources

All supplied practice inputs are synthetic and are not measurements from the competition dataset. The [walkthrough notebook and paired script](notebooks/README.md) are optional CPU-oriented demonstrations. Stand-alone practice includes a [group sampler](examples/toy_imbalance_sampler.py), [schedule visualizer](examples/toy_scheduler_visualizer.py), [checkpoint tracker](examples/toy_early_stopping.py), [AMP lifecycle](examples/toy_amp_lifecycle.py), and [error analyzer](examples/toy_error_analyzer.py). Their configuration is illustrative; use the assessment's explicit definitions for assessed runs.

## Working boundaries

- Discover actual class counts and model behavior from `data/public/`; no dataset statistics or benchmark outcomes are supplied here.
- Keep all sampling, oversampling, and augmentation decisions inside the training partition. Validation must retain its natural image-level distribution.
- A toy example illustrates arithmetic or a failure mode; it does not predict a winning intervention.
- Record software, device, seed, split, and configuration with every run. Repeated selection against one validation set can overfit human decisions even when gradients never use validation examples.
