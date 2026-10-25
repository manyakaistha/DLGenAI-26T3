# /// script
# requires-python = ">=3.10"
# dependencies = ["Markdown==3.8.2", "pymdown-extensions==10.16.1"]
# ///
"""Build the 26 October MRI transfer-learning live session."""
import argparse
from pathlib import Path
from build_live_session import build as render_session
from build_transfer_learning_notebook import build as build_notebook
from build_student_hub import validate_site

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'docs/transfer_learning_live_session'
MODULES = ['README', 'dataset', 'encoder', 'transfer', 'inputs', 'finetuning', 'experiments', 'notebook']
TITLES = ['Session overview', '1 · MRI data & patient split', '2 · Encoder & decoder',
          '3 · Pretraining & fair comparisons', '4 · MRI channels & normalization',
          '5 · Freeze & fine-tune', '6 · Evaluate the evidence', '7 · Companion notebook']
URLS = ['index.html', *[name + '.html' for name in MODULES[1:]]]


def build(out):
    build_notebook()
    render_session(out, source_dir=SOURCE, modules=MODULES, titles=TITLES, urls=URLS,
                   downloads=('transfer_learning.ipynb', 'transfer_learning.py'),
                   extra_files=('labs.css', 'labs.js', 'figures/provenance.json'),
                   identity=dict(title='MRI Transfer Learning · 26 October 2026',
                                 brand='MRI Transfer<br>Learning',
                                 description='Learn pretrained encoders, patient-level splitting, MRI channels, fine-tuning and fair segmentation experiments.',
                                 intro='26 October 2026 · From your baseline U-Net to a pretrained encoder.',
                                 topline='26 October · Milestone 3 live session',
                                 kicker='Inspect. Reuse. Compare.', hero='A different<br>starting point.',
                                 caption='Transfer learning, tested fairly.'))
    validate_site(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/transfer_session_preview')
    build(parser.parse_args().output.resolve())
