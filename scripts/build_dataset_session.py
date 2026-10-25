# /// script
# requires-python = ">=3.10"
# dependencies = ["Markdown==3.8.2", "pymdown-extensions==10.16.1"]
# ///
"""Build the independent Dataset live lab using the established session theme."""
import argparse
import shutil
from pathlib import Path
import build_live_session as session

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/dataset_live_session"


def build(out):
    # Reuse the existing checked renderer without changing the previous session.
    # Only student-facing sources enter the temporary staging tree.
    stage = ROOT / "runs/dataset_session_source"
    stage.mkdir(parents=True, exist_ok=True)
    for name in ("README.md", "notes.md", "dataset.md", "split.md", "dataloader.md", "augmentations.md", "dataset_tutorial.ipynb", "dataset_tutorial.py"):
        shutil.copy2(SOURCE / name, stage / name)
    overview = (stage / "README.md").read_text(encoding="utf-8", errors="ignore")
    overview = overview.replace("- [Instructor timing and answers](instructor.md)\n", "")
    (stage / "README.md").write_text(overview, encoding="utf-8")
    session.SOURCE = stage
    session.MODULES = ["README", "dataset", "split", "dataloader", "augmentations", "notes"]
    session.TITLES = ["Session overview", "1 · Dataset", "2 · 80/20 split", "3 · DataLoader", "4 · Augmentations", "Topic index"]
    session.URLS = ["index.html", "dataset.html", "split.html", "dataloader.html", "augmentations.html", "notes.html"]
    # The original renderer expects two download names; stage aliases only.
    shutil.copy2(stage / "dataset_tutorial.ipynb", stage / "wandb_tutorial.ipynb")
    shutil.copy2(stage / "dataset_tutorial.py", stage / "wandb_tutorial.py")
    session.notebook = lambda: None
    out.mkdir(parents=True, exist_ok=True)
    for name in ("dataset_tutorial.ipynb", "dataset_tutorial.py"):
        shutil.copy2(stage / name, out / name)
    session.build(out)
    for asset in ("labs.js", "labs.css"):
        shutil.copy2(SOURCE / asset, out / asset)
    replacements = {
        "Pixels, Masks &amp; Experiments": "From CSV rows to batches",
        "Pixels, Masks & Experiments": "From CSV rows to batches",
        "Pixels, Masks<br>& Experiments": "Dataset, DataLoader<br>&amp; Augmentations",
        "Explore segmentation data, masks and experiment tracking.": "Load a sample. Build a batch. Transform a pair.",
        "From pixels<br>to experiments.": "From CSV rows<br>to batches.",
        "A live lab in industrial segmentation.": "A live lab with synthetic cloud images.",
        "Industrial surface segmentation": "PyTorch Dataset &amp; DataLoader",
        "wandb_tutorial.ipynb": "dataset_tutorial.ipynb",
        "</head>": '<link rel="stylesheet" href="labs.css"><script defer src="labs.js"></script></head>',
        "Observe. Represent. Measure.": "Read. Decode. Stack.",
    }
    for page in out.glob("*.html"):
        text = page.read_text(encoding="utf-8", errors="ignore")
        for old, new in replacements.items():
            text = text.replace(old, new)
        page.write_text(text, encoding="utf-8")
    for alias in ("wandb_tutorial.ipynb", "wandb_tutorial.py"):
        (out / alias).unlink(missing_ok=True)
    print("Independent Dataset session ready:", out / "index.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "runs/dataset_session_preview")
    build(parser.parse_args().output.resolve())
