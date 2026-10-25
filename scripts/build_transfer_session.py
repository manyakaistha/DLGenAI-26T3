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
MODULES = ['README', 'encoder', 'transfer', 'finetuning', 'experiments', 'dataset', 'inputs', 'notebook']
TITLES = ['Session overview', '1 · The U-Net you know', '2 · Borrow an encoder',
          '3 · Freeze or fine-tune', '4 · Compare fairly', 'About the MRI example',
          'Input details', 'Notebooks & optional practice']
URLS = ['index.html', *[name + '.html' for name in MODULES[1:]]]


def build(out):
    build_notebook()
    render_session(out, source_dir=SOURCE, modules=MODULES, titles=TITLES, urls=URLS,
                   downloads=('classroom_walkthrough.ipynb', 'classroom_walkthrough.py',
                              'transfer_learning.ipynb', 'transfer_learning.py'),
                   nav_groups={1: 'Follow in class', 5: 'Explore later · optional'},
                   extra_files=('labs.css', 'labs.js', 'figures/provenance.json'),
                   identity=dict(title='Give Your U-Net a Head Start · 26 October 2026',
                                 brand='Your U-Net,<br>with a head start',
                                 description='Four short lessons on pretrained encoders, freezing, fine-tuning and fair comparisons. MRI setup and full training are optional.',
                                 intro='26 October 2026 · Build on the U-Net you already know.',
                                 topline='26 October · Milestone 3 live session',
                                 kicker='A familiar model. One new idea.', hero='Start with what<br>you know.',
                                 caption='Borrow features. Keep learning.', download_label='Classroom notebook ↓'))
    validate_site(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/transfer_session_preview')
    build(parser.parse_args().output.resolve())
