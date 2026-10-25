"""Generate the unexecuted MRI session notebook from its percent-cell script."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/transfer_learning_live_session"


def build_one(stem):
    text = (SOURCE / (stem + ".py")).read_text(encoding="utf-8", errors="ignore")
    cells, kind, lines = [], None, []

    def flush():
        if kind is None or not any(line.strip() for line in lines):
            return
        source = "\n".join(lines).strip() + "\n"
        cell = dict(cell_type=kind, metadata={}, source=source.splitlines(keepends=True),
                    id=f"mri-session-{len(cells):02d}")
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    for line in text.splitlines():
        if line.startswith("# %%"):
            flush(); lines = []
            kind = "markdown" if "[markdown]" in line else "code"
        elif kind:
            lines.append((line[2:] if line.startswith("# ") else "" if line == "#" else line) if kind == "markdown" else line)
    flush()
    setup = dict(cell_type="code", id="mri-setup", metadata={}, execution_count=None, outputs=[],
                 source=["# Run only if this runtime needs the pinned teaching dependencies.\n",
                         "# Restart the kernel after replacing already-imported binary packages.\n",
                         "%pip install torch==2.8.0 torchvision==0.23.0 numpy==2.2.6 Pillow==12.3.0 matplotlib==3.10.9\n"])
    if stem == "transfer_learning":
        cells.insert(1, setup)
    notebook = dict(cells=cells, metadata=dict(
        kernelspec=dict(display_name="Python 3", language="python", name="python3"),
        language_info=dict(name="python", version="3.11")), nbformat=4, nbformat_minor=5)
    out = SOURCE / (stem + ".ipynb")
    out.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(f"Generated {len(cells)} unexecuted cells → {out}")


def build():
    build_one("transfer_learning")
    build_one("classroom_walkthrough")


if __name__ == "__main__":
    build()
