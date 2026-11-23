# /// script
# requires-python = ">=3.10"
# dependencies = ["Markdown==3.8.2", "pymdown-extensions==10.16.1"]
# ///
"""Build Session 6 reading material and its self-contained HTML slides."""
import argparse
from pathlib import Path
from build_live_session import build as render_session
from build_student_hub import validate_site

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'docs/distillation_session'
MODULES = ['README', '01_distillation', '02_pseudo_labeling', '03_data_tricks', '04_calibration_and_safety', '05_reference']
TITLES = [
    'Session 6 overview',
    '1 · Ensemble distillation',
    '2 · Pseudo-labeling',
    '3 · Data-centric tricks',
    '4 · Calibration & safety',
    '5 · Quick reference',
]


def build(out):
    out = Path(out)
    p = SOURCE / 'presentation'
    target = out / 'presentation'
    target.mkdir(parents=True, exist_ok=True)
    js = (p / 'presentation.js').read_text(encoding='utf-8').replace(
        '/* SESSION_CONTENT */', (p / 'content.js').read_text(encoding='utf-8')
    )
    page = (
        (p / 'index.html')
        .read_text(encoding='utf-8')
        .replace('/* PRESENTATION_CSS */', (p / 'presentation.css').read_text(encoding='utf-8'))
        .replace('/* PRESENTATION_JS */', js)
        .replace('/* ACTIVITIES_JS */', (SOURCE / 'activities.js').read_text(encoding='utf-8'))
    )
    (target / 'index.html').write_text(page, encoding='utf-8')
    render_session(
        out,
        source_dir=SOURCE,
        modules=MODULES,
        titles=TITLES,
        urls=['index.html', *[m + '.html' for m in MODULES[1:]]],
        downloads=(),
        identity=dict(
            title='Session 6 · Distillation, Pseudo-Labeling & Data Tricks · 23 November 2026',
            brand='Teach smaller.<br>Label smarter.',
            description='Ensemble distillation, pseudo-labeling, domain shift, resolution, copy-paste augmentation and probability calibration.',
            intro='23 November 2026 · 40 minutes · interactive animated slides.',
            topline='Live Session 6 · 23 November 2026',
            kicker='Distill. Pseudo. Clean. Calibrate.',
            hero='Teach smaller.<br>Label smarter.',
            caption='A focused 40-minute tour of distillation, pseudo-labeling, and data-centric habits.',
            download_label='',
        ),
    )
    validate_site(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/distillation_preview')
    build(parser.parse_args().output.resolve())
