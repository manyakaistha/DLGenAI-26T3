# /// script
# requires-python = ">=3.10"
# dependencies = ["Markdown==3.8.2", "pymdown-extensions==10.16.1"]
# ///
"""Build the 60-minute cloud training session and separate optional DDP extension."""
import argparse
from pathlib import Path
from build_live_session import build as render_session
from build_efficiency_presentation import build as build_presentation
from build_student_hub import validate_site
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'docs/efficient_training_session'
MODULES=['README','01_clouds_and_training','02_fp16_bf16','03_amp','04_accumulation','05_resume_and_schedule','06_validation','07_ddp_optional','08_notebooks']
TITLES=['Session 4 overview','1 · Cloud training contract','2 · FP16 versus BF16','3 · Automatic mixed precision','4 · Gradient accumulation','5 · Schedule, checkpoint and resume','6 · Validation and next experiment','Optional · DDP in depth','Notebooks · Setup and handoffs']
URLS=['index.html',*[m+'.html' for m in MODULES[1:]]]
def build(out):
    out=Path(out);build_presentation(out/'presentation')
    render_session(out,source_dir=SOURCE,modules=MODULES,titles=TITLES,urls=URLS,
        downloads=('cloud_training.ipynb','cloud_training.py','cloud_ddp.ipynb','train_cloud_ddp.py'),
        extra_files=('activities.css','activities.js'),optional_urls=('07_ddp_optional.html',),
        page_routes={'06_validation.html':('05_resume_and_schedule.html','08_notebooks.html'),'08_notebooks.html':('06_validation.html',None),'07_ddp_optional.html':('06_validation.html','08_notebooks.html')},
        nav_groups={1:'60-minute core',7:'Outside the hour',8:'Prepare before class'},
        identity=dict(title='Session 4 · Efficient Training · 29 October 2026',brand='Train efficiently.<br>Keep the evidence.',description='AMP, FP16 versus BF16, gradient accumulation, scheduling, complete resume and validation on real Clouds data; optional in-depth DDP.',intro='29 October 2026 · 60-minute core with exact notebook handoffs. DDP is a separate optional extension.',topline='Live Session 4 · 29 October 2026',kicker='Measure. Recover. Explain.',hero='Train smarter.<br>Keep the evidence.',caption='One real cloud dataset. One careful hour.',download_label='Core notebook ↓'))
    validate_site(out)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'runs/efficiency_preview');build(parser.parse_args().output.resolve())
