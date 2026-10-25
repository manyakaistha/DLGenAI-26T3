# Industrial Surface Anomaly Segmentation — Student Guides

Public milestone guides and live session resources for DLGenAI 26T3.

- **[Open the student resource hub](https://manyakaistha.github.io/DLGenAI-26T3/)**
- [Milestone Guides](https://manyakaistha.github.io/DLGenAI-26T3/milestones/) — choose a milestone for its guides, questions and starter resources.
- [Live Session Resources](https://manyakaistha.github.io/DLGenAI-26T3/live-sessions/) — find session notes, interactive activities and notebooks.
- [Milestone 1 questions](MILESTONE_1.md)
- [Milestone 1 study guide and nine learning modules](docs/milestone_1_student_guide_v2/README.md)
- [Milestone 2 questions](MILESTONE_2.md)
- [Milestone 2 study guides and eight concept modules](https://manyakaistha.github.io/DLGenAI-26T3/milestones/milestone-2/)
- [Q11–Q12 starter notebook in Google Colab](https://colab.research.google.com/drive/1Jf4f7jKddRSMpiZ1D8oVqmR8LDK-t86Q?usp=sharing)
- [Colab setup instructions and notebook/Python downloads](docs/starter_code_q11_q12/README.md)

The guides include theory, illustrations, and worked practice exercises. The revised assessment contains questions and response formats without computed answers.

## Using the starter notebook

Open the Colab link, save a personal copy in Drive, choose a GPU runtime, and follow the [starter setup instructions](docs/starter_code_q11_q12/README.md). Set the competition data path, complete the Q11 TODOs, train for 5–10 epochs, and run the Q12 threshold sweep. Save the output files before ending the Colab session.

## Build the website

```bash
uv run --no-project scripts/build_student_pages.py --output _site
uv run --no-project python -m http.server 8765 --directory _site
```

The build checks all local links and anchors. GitHub Actions publishes the site to GitHub Pages when `main` changes. Enable GitHub Actions as the Pages publishing source in repository settings. Math rendering uses MathJax from jsDelivr.

This repository contains only student learning materials and their website build files. Download competition data separately into `data/public/` in your own working environment.

## Live session: Pixels, Masks & Experiments

A separate interactive teaching companion to Guides 1–3, with a compact W&B demo, synthetic image and mask examples, and numbered RLE traces.

- [Student session notes](https://manyakaistha.github.io/DLGenAI-26T3/session/)
- [Notebook](docs/live_session_materials/wandb_tutorial.ipynb)

The notebook uses a small synthetic dataset and needs no competition data or GPU.


## Live session: Dataset, DataLoader & Augmentations

A beginner-friendly continuation with an 80/20 split lesson, explained PyTorch code, a batch builder and four interactive augmentation labs.

- [Student session notes](https://manyakaistha.github.io/DLGenAI-26T3/session-2/)
- [Notebook](docs/dataset_live_session/dataset_tutorial.ipynb)

The notebook creates original synthetic cloud-like data locally and runs on CPU.


## Live session: From a Baseline U-Net to a Pretrained Encoder · 26 October 2026

Detailed study notes explain encoder–decoder mechanics, pretrained initialization, freezing and fine-tuning, BCE–Dice, and controlled evaluation. Full MRI application pages appear between the relevant concept pages and are clearly optional, with skip/continue links back to the theory.

- [Student session notes](https://manyakaistha.github.io/DLGenAI-26T3/session-3/)
- [Concept walkthrough notebook](docs/transfer_learning_live_session/classroom_walkthrough.ipynb)
- [Optional complete MRI experiment](docs/transfer_learning_live_session/notebook.md)

Original technical depth, code examples and interactive controls are preserved. Dataset preparation and full training remain optional; dataset attribution and license are documented in the application notes.


## Interactive live presentation · 26 October

[Open the 16:9 presentation](https://manyakaistha.github.io/DLGenAI-26T3/session-3/presentation/) alongside the unchanged detailed study guides. It contains 21 core slides, three optional MRI stops, step-through diagrams and numerical activities. Use arrows, the slide index and fullscreen; download the standalone HTML from the index to present offline.

Sources and controls: [presentation README](docs/transfer_learning_presentation/README.md). Build independently with `uv run --no-project python scripts/build_transfer_presentation.py`, or use the website build above.
