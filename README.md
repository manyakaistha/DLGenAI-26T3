# Industrial Surface Anomaly Segmentation — Student Guides

Public milestone guides and live session resources for DLGenAI 26T3.

- **[Open the student resource hub](https://manyakaistha.github.io/DLGenAI-26T3/)**
- [Milestone Guides](https://manyakaistha.github.io/DLGenAI-26T3/milestones/) — choose a milestone for its guides, questions and starter resources.
- [Live Session Resources](https://manyakaistha.github.io/DLGenAI-26T3/live-sessions/) — find session notes, interactive activities and notebooks.
- [Milestone 1 questions](MILESTONE_1.md)
- [Milestone 1 study guide and nine learning modules](docs/milestone_1_student_guide_v2/README.md)
- [Milestone 2 questions](MILESTONE_2.md)
- [Milestone 2 study guides and eight concept modules](https://manyakaistha.github.io/DLGenAI-26T3/milestones/milestone-2/)
- [Milestone 3 questions](MILESTONE_3.md)
- [Milestone 3 study guides and nine concept modules](docs/milestone_3_student_guide/README.md)
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


## Live session: From a Baseline U-Net to a Pretrained Encoder

Notes on encoder-decoder architectures, ImageNet transfer learning, fine-tuning, Batch Normalization drift, and the 4-Sigmoid vs 5-Softmax comparison, with optional MRI examples.

- [Student session notes](https://manyakaistha.github.io/DLGenAI-26T3/session-3/)
- [Notebook](docs/transfer_learning_live_session/transfer_learning.ipynb)

The notebook trains a toy encoder-decoder on synthetic textures.


## Live session: Train Efficiently, Keep the Evidence

Notes, activities, and an end-to-end cloud training walkthrough on automatic mixed precision (AMP), gradient accumulation, cosine schedules, and checkpoint governance.

- [Student session notes](https://manyakaistha.github.io/DLGenAI-26T3/session-4/)
- [Notebook](docs/efficient_training_session/cloud_training.ipynb)

The notebook demonstrates mixed-precision training and checkpoint validation on GPU.
