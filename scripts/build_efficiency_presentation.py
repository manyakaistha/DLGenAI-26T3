"""Build the standalone 29 October interactive teaching presentation."""
import argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'docs/efficient_training_session'
def build(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);p=SOURCE/'presentation'
    js=(p/'presentation.js').read_text().replace('/* SESSION_CONTENT */',(p/'content.js').read_text())
    page=(p/'index.html').read_text().replace('/* PRESENTATION_CSS */',(p/'presentation.css').read_text()+'\n'+(SOURCE/'activities.css').read_text()).replace('/* PRESENTATION_JS */',js).replace('/* ACTIVITIES_JS */',(SOURCE/'activities.js').read_text())
    (out/'index.html').write_text(page);print(f'Built efficient-training presentation → {out}')
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'runs/efficiency_preview/presentation');build(parser.parse_args().output)
