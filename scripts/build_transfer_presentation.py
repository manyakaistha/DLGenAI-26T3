"""Build a standalone, offline-capable HTML classroom presentation (stdlib only)."""
import argparse
import base64
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'docs/transfer_learning_presentation'


def build(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    page = (SOURCE / 'index.html').read_text()
    css = (SOURCE / 'presentation.css').read_text()
    js = (SOURCE / 'presentation.js').read_text()
    figure = ROOT / 'docs/transfer_learning_live_session/figures/mri_sequences.png'
    data_url = 'data:image/png;base64,' + base64.b64encode(figure.read_bytes()).decode()
    js = js.replace('__MRI_IMAGE__', data_url)
    page = page.replace('/* PRESENTATION_CSS */', css).replace('/* PRESENTATION_JS */', js)
    target = out / 'index.html'
    target.write_text(page)
    print(f'Built self-contained presentation → {target}')
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/transfer_presentation_preview')
    build(parser.parse_args().output.resolve())
