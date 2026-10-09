# Industrial Surface Anomaly Segmentation — Student Guides

Public learning resources for DLGenAI 26T3, Milestone 1.

- **[Read the full guide online](https://manyakaistha.github.io/DLGenAI-26T3/)**
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
