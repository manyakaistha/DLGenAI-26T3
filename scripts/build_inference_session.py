# /// script
# requires-python = ">=3.10"
# dependencies = ["Markdown==3.8.2", "pymdown-extensions==10.16.1"]
# ///
"""Build Session 5 reading material and its self-contained HTML slides."""
import argparse
from pathlib import Path
from build_live_session import build as render_session
from build_student_hub import validate_site
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'docs/inference_session'
MODULES=['README','01_pipeline','02_tta','03_ema_ensembles','04_decisions','05_validation_submission','06_practical','07_reference']
TITLES=['Session 5 overview','1 · Inference contract','2 · Test-time augmentation','3 · EMA and ensembles','4 · Thresholds and components','5 · Validation and submission','6 · Notebook and exercises','7 · Quick reference']
def build(out):
    out=Path(out)
    p=SOURCE/'presentation'
    target=out/'presentation';target.mkdir(parents=True,exist_ok=True)
    js=(p/'presentation.js').read_text().replace('/* SESSION_CONTENT */',(p/'content.js').read_text())
    page=(p/'index.html').read_text().replace('/* PRESENTATION_CSS */',(p/'presentation.css').read_text()).replace('/* PRESENTATION_JS */',js).replace('/* ACTIVITIES_JS */',(SOURCE/'activities.js').read_text())
    (target/'index.html').write_text(page)
    render_session(out,source_dir=SOURCE,modules=MODULES,titles=TITLES,
        urls=['index.html',*[m+'.html' for m in MODULES[1:]]],downloads=('inference_lab.ipynb','inference_lab.py'),
        identity=dict(title='Session 5 · Inference Techniques · 9 November 2026',brand='Inference you<br>can explain.',description='TTA, EMA, SWA, probability ensembling, thresholds, component filtering and validated segmentation inference.',intro='9 November 2026 · 60 minutes · synthetic CPU notebook.',topline='Live Session 5 · 9 November 2026',kicker='Predict. Align. Decide.',hero='Inference you<br>can explain.',caption='A careful hour with TTA, EMA and ensembles.',download_label='Inference notebook ↓'))
    validate_site(out)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'runs/inference_preview');build(parser.parse_args().output.resolve())
