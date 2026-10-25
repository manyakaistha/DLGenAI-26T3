# Give your U-Net a head start

**26 October 2026 · Milestone 3 · Transfer learning**

You have already trained a U-Net. Today, we will keep that familiar idea and ask: **what if the encoder started with some useful features already learned?**

<div class="lesson-note">
<strong>For this class:</strong> follow four focused lessons and try the browser activities. Your existing segmentation project is the starting point. The MRI pictures give us an example to look at together; downloading the dataset and training another model are optional.
</div>

## Four connected lessons

1. **[Recognise the U-Net parts](encoder.md).** The encoder finds features; the decoder turns them into a mask.
2. **[Borrow a trained encoder](transfer.md).** Learn what “pretrained” means and which part of the model we reuse.
3. **[Choose what should learn](finetuning.md).** Compare a frozen encoder with fine-tuning.
4. **[Make a fair comparison](experiments.md).** Change one thing and check the evidence.

Each page explains the mechanism, works through a code or numerical example, and includes a reasoning question. Sections labelled **“Go deeper”** can wait until you want the details.

## Why are we looking at MRI pictures?

The task is familiar: **give the model a picture and teach it to mark a region**. For your project, the regions are surface defects. In the MRI example, the reference mask marks an annotated abnormal region.

We only need to recognise the image and its mask today. MRI terminology, file handling and the full training pipeline are in the optional pages below. You can learn the transfer-learning idea using the pictures and activities on this website alone.

## Three ideas to take away

- **Pretrained** means some model weights were learned on an earlier task.
- **Freeze** means keep a set of parameters fixed; **fine-tune** means let them adapt.
- To test pretraining fairly, keep the model structure and experiment settings the same, then change the encoder’s starting weights.

## Optional: watch or try a short code demo

[Small classroom notebook](classroom_walkthrough.ipynb) · [Python version](classroom_walkthrough.py) · [How to use it](notebook.md)

The classroom demo traces feature shapes, executes one decoder join, and switches which parameters can learn. It needs no MRI files, GPU or training run. Its default uses random weights for a shape demonstration; an instructor can enable ImageNet weights to show the pretrained version.

## Explore later

- [About the MRI example](dataset.md): the supplied pictures, masks and a simple patient-split activity.
- [Input details](inputs.md): channels, scaling and normalization.
- [The full experiment](notebook.md#optional-the-full-mri-experiment): the original complete training notebook, available when you want to try it.

## Where the example comes from

MRI example: [Mateusz Buda’s Brain MRI segmentation dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation), from TCIA / TCGA-LGG. The reference masks mark FLAIR abnormalities. Dataset metadata lists **CC BY-NC-SA 4.0**; the [teaching figure’s provenance](figures/provenance.json) records its source and changes. This is a machine-learning example, not a diagnostic tool.

Further reading: [author’s implementation](https://github.com/mateuszbuda/brain-segmentation-pytorch), [associated paper](https://doi.org/10.1016/j.compbiomed.2019.05.002), [dataset license](https://creativecommons.org/licenses/by-nc-sa/4.0/).
