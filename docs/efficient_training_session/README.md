# Train efficiently. Keep the evidence.

**Live Session 4 · 29 October 2026 · 60 minutes**

You have already trained and fine-tuned a model. Today you will make its training loop more efficient, recoverable and explainable. We reuse **Understanding Clouds from Satellite Images**, with Fish, Flower, Gravel and Sugar as four independent mask channels.

[Open the interactive presentation →](presentation/index.html) · [Notebook setup and exact handoffs →](08_notebooks.md)

## The hour at a glance

| Clock | Teach and do | Evidence learners should produce |
|---|---|---|
| 00–06 | Recall the cloud task; name a bottleneck | A correct input/target contract |
| 06–16 | FP32, FP16 and BF16; predict numerical casts | Explain range versus spacing |
| 16–28 | AMP lifecycle and first notebook handoff | A finite run and an honest timing interpretation |
| 28–39 | Gradient accumulation; second handoff | Correct uneven-window gradient weighting |
| 39–51 | Schedule, save, interrupt and resume; third handoff | Resumed versus uninterrupted weight comparison |
| 51–57 | Validation diagnostics; choose one experiment | A quality check alongside resource measurements |
| 57–60 | Exit questions and next steps | A defensible next milestone experiment |

**DDP is optional self-study.** It is excluded from the presentation's default route and from this hour. [The separate guide](07_ddp_optional.md) and [DDP notebook](cloud_ddp.ipynb) cover the complete two-process example.

## How to use these materials

Read the short core explanations in order. Every chapter includes a prediction or repair exercise; discuss your answer before opening its explanation. The interactive workbenches use deliberately small, labelled examples. The notebooks use real downloaded cloud images and never silently substitute generated data.

For teaching, open the presentation and complete notebook **00 · Setup** before the session. Each handoff slide has a Colab link to the named section and a link to its run instructions. Browser links select a section; they do not execute prerequisite cells. Keep one already configured notebook tab open and use its section headings when returning.

## What we will implement

A deliberately small convolutional segmentation network takes RGB tensors shaped `[B,3,64,96]` and produces four raw logit maps `[B,4,64,96]`. A per-image BCE plus soft Dice objective gives a transparent gradient-accumulation example. It has no BatchNorm or dropout, so the physical-batch comparison does not conceal their differences.

We implement autocast, FP16 gradient scaling, correct accumulation windows, clipping, update-based warmup/cosine scheduling, complete epoch-boundary checkpoints, and held-out diagnostics. This is a training-mechanics lesson, not a tuned cloud solution.

## Is this the right bridge to Milestone 4?

Yes: efficiency and reproducibility are useful before making longer model comparisons. Start Milestone 4 only after learners can identify the held-out set, run a trustworthy baseline, match logits to the loss, and resume without resetting training state. Otherwise use the final exercise to repair those foundations first.

After today, compare **one** model or training change while fixing the split, preprocessing and evaluation protocol. Avoid changing precision, effective batch, architecture and threshold together. More experiments only help if you can explain what changed.

## Preparation checklist

- Download one coherent cloud variant and verify its CSV matches its images.
- Run notebook setup and one warmup before class; installation and data download are outside the hour.
- Have a CPU FP32 route ready. A CUDA AMP demonstration requires compatible hardware; multi-GPU hardware is not needed for the core lesson.
- Keep measured outputs private and record hardware, runtime and dataset scope before comparing timing.

[Begin with the data contract →](01_clouds_and_training.md)
