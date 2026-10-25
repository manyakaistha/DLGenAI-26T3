# /// script
# requires-python = ">=3.10"
# dependencies = ["Markdown==3.8.2", "pymdown-extensions==10.16.1"]
# ///
"""Build the 26 October MRI transfer-learning live session."""
import argparse
from pathlib import Path
from build_live_session import build as render_session
from build_transfer_learning_notebook import build as build_notebook
from build_transfer_presentation import build as build_presentation
from build_student_hub import validate_site

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'docs/transfer_learning_live_session'
MODULES = ['README', 'encoder', 'dataset', 'transfer', 'inputs', 'finetuning', 'experiments', 'notebook']
TITLES = ['Session overview', '1 · Encoder & decoder', 'Optional · MRI data & split',
          '2 · Pretraining & fair comparisons', 'Optional · MRI channels & normalization',
          '3 · Freezing & fine-tuning', '4 · Losses, metrics & evidence', 'Optional · Full MRI notebook']
URLS = ['index.html', *[name + '.html' for name in MODULES[1:]]]


def build(out):
    build_notebook()
    render_session(out, source_dir=SOURCE, modules=MODULES, titles=TITLES, urls=URLS,
                   downloads=('classroom_walkthrough.ipynb', 'classroom_walkthrough.py',
                              'transfer_learning.ipynb', 'transfer_learning.py'),
                   nav_groups={1: 'Concepts & optional MRI applications'},
                   optional_urls=('dataset.html', 'inputs.html', 'notebook.html'),
                   page_routes={
                       'index.html': (None, 'encoder.html'),
                       'encoder.html': ('index.html', 'transfer.html'),
                       'dataset.html': ('encoder.html', 'transfer.html'),
                       'transfer.html': ('encoder.html', 'finetuning.html'),
                       'inputs.html': ('transfer.html', 'finetuning.html'),
                       'finetuning.html': ('transfer.html', 'experiments.html'),
                       'experiments.html': ('finetuning.html', 'index.html'),
                       'notebook.html': ('experiments.html', 'index.html'),
                   },
                   extra_files=('labs.css', 'labs.js', 'figures/provenance.json'),
                   identity=dict(title='From a Baseline U-Net to a Pretrained Encoder · 26 October 2026',
                                 brand='Transfer Learning<br>Study Notes',
                                 description='Detailed encoder-decoder, pretraining, fine-tuning, loss and evaluation notes, with optional MRI applications between the concept pages.',
                                 intro='26 October 2026 · Detailed concepts, worked examples and optional MRI applications.',
                                 topline='26 October · Milestone 3 live session',
                                 kicker='Inspect. Reuse. Compare.', hero='From features<br>to evidence.',
                                 caption='Understand the mechanism. Test the idea.', download_label='Classroom notebook ↓'))
    build_presentation(out / 'presentation')
    validate_site(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/transfer_session_preview')
    build(parser.parse_args().output.resolve())
