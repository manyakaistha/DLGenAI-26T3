# Starter Code: Questions 11 and 12

**[Open the starter notebook in Google Colab](https://colab.research.google.com/drive/1Jf4f7jKddRSMpiZ1D8oVqmR8LDK-t86Q?usp=sharing)**

You can also download [starter.ipynb](starter.ipynb) or use [starter.py](starter.py) in an editor that supports `# %%` cells.

The starter supplies image-level combination stratification, native-resolution image/mask loading, a ResNet-34 U-Net with four sigmoid channels, BCE plus foreground Dice loss, 5–10 epoch training, and the threshold sweep `{0.3, 0.5, 0.7, 0.9}`. It applies no minimum-area filtering. Complete the Question 11 calculation TODOs and write the Question 12 conclusions from your own run.

## Run in Google Colab

1. Open the Colab link above and select **File → Save a copy in Drive**. Work in your own copy.
2. Select **Runtime → Change runtime type → GPU**, then connect to the runtime. GPU availability depends on Colab capacity.
3. Add a setup cell **before the notebook's imports** and run:

```python
!curl -LsSf https://astral.sh/uv/install.sh | sh
import sys
!~/.local/bin/uv pip install --python {sys.executable} torch torchvision segmentation-models-pytorch numpy pandas pillow scikit-learn matplotlib tqdm
```

4. Download the competition data using your own competition access. Upload the dataset archive to the Colab Files panel and extract it into a local runtime folder. For example, after uploading an archive named `public.zip`:

```python
!unzip -q /content/public.zip -d /content/data
from pathlib import Path
print(list(Path("/content/data").rglob("train.csv")))
```

5. Set `DATA_ROOT` in the notebook's configuration cell to the folder containing both `train.csv` and `train/`. Archive layouts vary; use the printed path to locate the right folder. For example, if the CSV is `/content/data/public/train.csv`, set:

```python
DATA_ROOT = Path("/content/data/public")
OUTPUT_DIR = Path("/content/milestone_1_outputs")
```

6. Run cells from top to bottom. Complete the Q11 TODOs, inspect the image/mask plot, train for **5–10 epochs**, and run the four-threshold sweep. Reduce `BATCH_SIZE` if the runtime runs out of GPU memory. Keep the same seed, split and selected checkpoint throughout the sweep.
7. Save your notebook and download the output folder before ending your session. With the output path above, add a final cell:

```python
import shutil
from google.colab import files
archive = shutil.make_archive("/content/milestone_1_outputs", "zip", OUTPUT_DIR)
files.download(archive)
```

Colab runtime files are temporary. Saving your notebook to Drive does not also save its dataset, checkpoint or CSV files. See the [Colab FAQ](https://research.google.com/colaboratory/faq.html) for runtime storage and GPU details, and the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) for the installer used above.

## Run locally or in Kaggle

Run from a working directory containing `data/public/train.csv` and `data/public/train/`, or edit `DATA_ROOT` in the first code cell. In Kaggle, point it at the competition's mounted `public` folder. The notebook has no imports from this project's `src/`, fold files, trained models, or reports.

For a standalone environment with `uv` installed:

```bash
uv venv --python 3.11
uv pip install --python .venv/bin/python torch torchvision segmentation-models-pytorch numpy pandas pillow scikit-learn matplotlib tqdm jupyterlab
uv run --no-project --python .venv/bin/python python -m jupyterlab
```

Select the environment's Python kernel, then open the notebook. For this project's existing environment, run from the project root:

```bash
uv run --with jupyterlab jupyter lab docs/starter_code_q11_q12/starter.ipynb
```

The model's ImageNet weights download on first use. A GPU is recommended for the full run; reduce `BATCH_SIZE` if necessary. CUDA, Apple MPS, and CPU are detected automatically. CPU execution is supported but slower. Results depend on hardware, training, and the chosen seed.

## Student work

1. Inspect the image-level tables and complete the two Q11 calculations. Distinguish the ideal fractional allocation from actual integer holdout counts.
2. Inspect the plotted image and masks before training. Keep all four annotation rows for an image in the same partition.
3. Train for 5–10 epochs. Keep one fixed split and the same selected checkpoint for every threshold.
4. Use the generated table and plots to write the Q12 bullet points. Measure rather than assume where the scores peak.

The starter reports an error if a combination occurs only once, because such a stratum cannot appear in both partitions. If this occurs with a changed dataset, document an explicit allocation policy.

Outputs go to `runs/milestone_1_starter/`: `holdout.csv`, `best.pt`, `training_history.csv`, `threshold_sweep.csv`, and `threshold_sweep.png`. These are local experiment files; keep them out of Git.

`assessment_score` implements the component definitions explicitly given in Question 8. It is the score to use for the Q12 exercise; it does not claim to reproduce an unpublished leaderboard implementation. `fp_pixel_fraction` measures false-alarm volume; `fp_pair_rate` measures how often an empty pair receives any false alarm.
