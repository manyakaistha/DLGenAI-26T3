# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy==2.2.6", "matplotlib==3.10.9"]
# ///
"""Deterministic synthetic figures for the deployment guide; no competition data."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
OUT = Path(__file__).resolve().parents[1] / 'docs/deployment_guide/figures'
OUT.mkdir(parents=True, exist_ok=True)
PAPER, INK, TEAL, RED, PURPLE, MUTED = '#fffefa', '#172d3a', '#087f8c', '#df653c', '#73419d', '#5d6c73'
COLORS = np.array([[230, 57, 70], [42, 157, 143], [69, 123, 255], [244, 162, 97]], float)
NAMES = ['defect_a', 'defect_b', 'defect_c', 'defect_d']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.facecolor': PAPER, 'text.color': INK})


def flow():
    steps = [('1  Upload', 'any image\n(PNG / JPG)', TEAL), ('2  Prepare', 'grayscale\nresize 800×128\nnormalise', TEAL),
             ('3  Model', 'your weights\n→ 4 logit maps', PURPLE), ('4  Decide', 'sigmoid\n> threshold', PURPLE),
             ('5  Overlay', 'colour each\nmask on image', RED)]
    fig, ax = plt.subplots(figsize=(13.4, 3.1)); ax.set_xlim(0, 13.4); ax.set_ylim(0, 3.1); ax.axis('off')
    w, gap = 2.15, 0.56
    for i, (t, sub, c) in enumerate(steps):
        x = 0.15 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 0.5), w, 2.0, boxstyle='round,pad=0.02,rounding_size=0.15', fc='white', ec=c, lw=2.2))
        ax.add_patch(FancyBboxPatch((x, 2.0), w, 0.5, boxstyle='round,pad=0.02,rounding_size=0.15', fc=c, ec=c))
        ax.text(x + w / 2, 2.25, t, ha='center', va='center', color='white', fontsize=13, fontweight='bold')
        ax.text(x + w / 2, 1.2, sub, ha='center', va='center', fontsize=11.5, linespacing=1.5)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.05, 1.5), (x + w + gap - 0.05, 1.5), arrowstyle='-|>', mutation_scale=18, color=MUTED, lw=2))
    ax.text(6.7, 0.12, 'Gradio runs steps 1 and 5 for you — you only write steps 2 to 4.', ha='center', fontsize=12, color=MUTED, style='italic')
    fig.savefig(OUT / 'flow.png', dpi=160, bbox_inches='tight'); plt.close(fig)


def overlay():
    rng = np.random.default_rng(7)
    H, W = 128, 800
    yy, xx = np.mgrid[0:H, 0:W]
    img = 120 + 18 * np.sin(xx / 37.0) + 10 * np.sin(yy / 9.0) + rng.normal(0, 6, (H, W))
    masks = np.zeros((4, H, W), bool)
    masks[0] = ((xx - 190) / 70) ** 2 + ((yy - 52) / 26) ** 2 < 1
    masks[2] = (abs(xx - 470) < 60) & (abs(yy - 70) < 9)
    masks[3] = ((xx - 665) / 45) ** 2 + ((yy - 40) / 20) ** 2 < 1
    for c, m in enumerate(masks): img[m] -= [30, 0, 22, 14][c] if c != 1 else 0
    gray = np.clip(img, 0, 255)
    canvas = np.stack([gray] * 3, -1)
    for c in range(4):
        canvas[masks[c]] = 0.5 * canvas[masks[c]] + 0.5 * COLORS[c]
    fig, axs = plt.subplots(2, 1, figsize=(11, 4.6))
    for ax, im, t in zip(axs, [np.stack([gray] * 3, -1), canvas], ['Input: any surface image', 'Output: predicted masks drawn on top']):
        ax.imshow(im.astype(np.uint8)); ax.set_title(t, loc='left', fontsize=12.5, fontweight='bold'); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_color('#c7d0d3')
    fig.legend(handles=[Patch(fc=COLORS[i] / 255, label=NAMES[i]) for i in range(4)], loc='lower center', ncol=4, frameon=False, fontsize=11.5)
    fig.text(0.99, 0.01, 'synthetic illustration', ha='right', fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(OUT / 'overlay_example.png', dpi=160, bbox_inches='tight'); plt.close(fig)


def files():
    items = [('app.py', 'the web app: preprocess → model → overlay', TEAL), ('model.py', 'your model class (copied from training)', PURPLE),
             ('weights.pt', 'your trained weights (state_dict)', PURPLE), ('requirements.txt', 'libraries the Space installs', MUTED),
             ('README.md', 'Space settings (a short header)', MUTED)]
    fig, ax = plt.subplots(figsize=(10.5, 3.3)); ax.set_xlim(0, 10.5); ax.set_ylim(0, 3.3); ax.axis('off')
    for i, (n, d, c) in enumerate(items):
        y = 3.0 - i * 0.6
        ax.add_patch(FancyBboxPatch((0.1, y - 0.22), 2.5, 0.44, boxstyle='round,pad=0.02,rounding_size=0.1', fc='white', ec=c, lw=2))
        ax.text(1.35, y, n, ha='center', va='center', fontsize=12.5, family='DejaVu Sans Mono', fontweight='bold', color=c)
        ax.text(2.85, y, d, va='center', fontsize=12.5)
    fig.savefig(OUT / 'space_files.png', dpi=160, bbox_inches='tight'); plt.close(fig)


if __name__ == '__main__':
    flow(); overlay(); files(); print('wrote', *sorted(p.name for p in OUT.glob('*.png')))
