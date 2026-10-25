# Industrial Surface Anomaly Segmentation — Student Guides

Public milestone guides and live session resources for DLGenAI 26T3.

- **[Open the student resource hub](https://manyakaistha.github.io/DLGenAI-26T3/)**
- [Milestone Guides](https://manyakaistha.github.io/DLGenAI-26T3/milestones/) — choose a milestone for its guides, questions and starter resources.
- [Live Session Resources](https://manyakaistha.github.io/DLGenAI-26T3/live-sessions/) — find session notes, interactive activities and notebooks.
- [Revised milestone questions](MILESTONE_1.md)
- [Study guide index and nine learning modules](docs/milestone_1_student_guide_v2/README.md)
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
