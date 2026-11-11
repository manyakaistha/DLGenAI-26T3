# /// script
# requires-python = ">=3.10"
# dependencies = ["Markdown==3.8.2", "pymdown-extensions==10.16.1"]
# ///
"""Build the Hugging Face deployment guide and its downloadable starter folder."""
import argparse
import zipfile
from pathlib import Path

from build_live_session import build as render_session
from build_student_hub import validate_site

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'docs/deployment_guide'
MODULES = ['README', '01_prepare', '02_publish']
TITLES = ['Overview', '1 · Prepare the files', '2 · Publish on Hugging Face']
URLS = ['index.html', '01_prepare.html', '02_publish.html']
ZIP = 'surface_defect_space.zip'


def make_zip():
    """Deterministic archive of docs/deployment_guide/space_template/."""
    template = SOURCE / 'space_template'
    with zipfile.ZipFile(SOURCE / ZIP, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(template.iterdir()):
            info = zipfile.ZipInfo(f'surface_defect_space/{path.name}', (2026, 11, 11, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())


def build(out):
    out = Path(out)
    make_zip()
    render_session(
        out, source_dir=SOURCE, modules=MODULES, titles=TITLES, urls=URLS, downloads=(ZIP,),
        identity=dict(
            title='Deploy Your Model on Hugging Face', brand='Deploy your<br>model.',
            description='A minimal Gradio app on a free ZeroGPU Hugging Face Space: upload an image and see the predicted masks overlaid.',
            intro='A 20-minute hands-on guide. Five small files, one free Space.',
            topline='Hands-on guide · Deployment', kicker='Prepare. Publish. Share.',
            hero='From weights<br>to a web page.', caption='Upload an image. See the masks.',
            download_label='Starter folder ↓'))
    validate_site(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/deployment_preview')
    build(parser.parse_args().output.resolve())
