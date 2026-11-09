"""Generate an output-free notebook from the paired inference lab script."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def build():
    source = ROOT / 'docs/inference_session/inference_lab.py'
    chunks = source.read_text().split('# %%')[1:]
    cells = [dict(cell_type='code', id='optional-install', metadata={}, execution_count=None, outputs=[], source=['# Optional in a fresh hosted runtime:\n', '%pip install numpy==2.2.6'])]
    for i, chunk in enumerate(chunks):
        is_md = chunk.startswith(' [markdown]')
        text = chunk.split('\n', 1)[1].strip('\n') + '\n'
        if is_md:
            text = '\n'.join(line[2:] if line.startswith('# ') else line[1:] if line == '#' else line for line in text.splitlines()) + '\n'
        cell = dict(cell_type='markdown' if is_md else 'code', id=f'inference-{i:02d}', metadata={}, source=text.splitlines(keepends=True))
        if not is_md:
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    notebook = dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'), language_info=dict(name='python', version='3.11')), nbformat=4, nbformat_minor=5)
    target = source.with_suffix('.ipynb')
    target.write_text(json.dumps(notebook, indent=1) + '\n')
    print(f'Built {len(cells)}-cell inference notebook')

if __name__ == '__main__':
    build()
