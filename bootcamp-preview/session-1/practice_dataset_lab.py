# Generated from the matching notebook by scripts/build_bootcamp_notebooks.py.

# %% [markdown] 1
# # Session 1 practice · Explore real cloud masks
# 
# **30–31 October 2026 · Guided practice · ~45–60 minutes**
# 
# This notebook uses Kaggle's **Understanding Clouds from Satellite Images** training data: Fish, Flower, Gravel and Sugar masks. It is a separate public dataset for learning the same four-channel, RLE and validation ideas; the objects, image geometry and scoring rules differ from any industrial task. We inspect labels and a few actual pictures, then compare a tiny **spatial pixel-prior baseline** on held-out images. It is deliberately not a segmentation training pipeline or competition solution.
# 
# **Before Run All:** choose one coherent source. Default `CLOUD_DATA_VARIANT=resized_384x576` uses the [Kaggle 384×576 resized mirror](https://www.kaggle.com/datasets/phunghieu/understanding-clouds-from-satellite-images-384x576), with its **own** `train.csv` and `train_images/`. `CLOUD_DATA_VARIANT=original` uses the [official competition files](https://www.kaggle.com/competitions/understanding_cloud_organization/data), whose images and masks are 2100×1400; Kaggle may ask you to accept its rules before downloading. Set `CLOUD_DATA_ROOT` to the chosen dataset folder or edit `DATA_ROOT` below. **Never mix** CSV labels from one size with images from the other. We check both image dimensions and the CSV's RLE coordinate range. The notebook reads real data only; missing files cause an actionable error. The data's usage terms are governed by its source, and no data is redistributed here.
# 
# Local Jupyter: `uv run --no-project --with numpy==2.2.6 --with pandas==2.3.3 --with matplotlib==3.10.9 --with pillow==12.3.0 --with jupyterlab==4.4.9 jupyter lab`. For the paired script, use the same four pinned packages with `python docs/bootcamp_session_1/practice_dataset_lab.py`. Kaggle's Python notebook environment normally has these packages.
# 
# **Safety for your evidence:** results are a small illustrative holdout, not a leaderboard score. Do not tune on hidden test data.

# %% 2
from pathlib import Path
import os
import json
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

CLASSES = ('Fish', 'Flower', 'Gravel', 'Sugar')
DATA_VARIANT = os.environ.get('CLOUD_DATA_VARIANT', 'resized_384x576')
CONTRACTS = {
    'resized_384x576': {'size': (576, 384), 'default_root': 'data/bootcamp/clouds_384x576'},
    'original': {'size': (2100, 1400), 'default_root': 'data/bootcamp/understanding_cloud_organization'},
}
if DATA_VARIANT not in CONTRACTS:
    raise ValueError(f'Unknown CLOUD_DATA_VARIANT={DATA_VARIANT!r}; choose one of {tuple(CONTRACTS)}')
EXPECTED_SIZE = CONTRACTS[DATA_VARIANT]['size']  # (width, height), for both image and CSV mask
DATA_ROOT = Path(os.environ.get('CLOUD_DATA_ROOT', CONTRACTS[DATA_VARIANT]['default_root'])).expanduser()
SMOKE_IDS = tuple(x.strip() for x in os.environ.get('CLOUD_SMOKE_IMAGE_IDS', '').split(',') if x.strip())
OUTPUT_ROOT = Path(os.environ.get('CLOUD_OUTPUT_ROOT', 'runs/bootcamp_session1_practice')).expanduser()
CSV = DATA_ROOT / 'train.csv'
IMAGES = DATA_ROOT / 'train_images'
if not CSV.is_file() or not IMAGES.is_dir():
    raise FileNotFoundError(
        f'Real cloud data missing under {DATA_ROOT.resolve()}. Set CLOUD_DATA_ROOT to the folder '
        f'containing train.csv and train_images/ for CLOUD_DATA_VARIANT={DATA_VARIANT}. '
        'Download the selected Kaggle source yourself, then rerun. No synthetic fallback is used.'
    )
print('Read-only dataset root:', DATA_ROOT.resolve(), '| variant:', DATA_VARIANT, '| expected size:', EXPECTED_SIZE)
if SMOKE_IDS:
    print('SMOKE MODE: using only explicitly listed image IDs; results are a code-path check, not dataset evidence')

# %% [markdown] 3
# ## 1 · Treat one picture as the unit
# 
# `Image_Label` joins a filename and category with the **last underscore**. Its four rows belong to one picture. Empty `EncodedPixels` means no labelled pixels for that image/category. Read strings faithfully so missing values do not become accidental foreground.
# 
# **Predict:** Is the number of CSV rows the number of images? What would go wrong if we randomly split rows?

# %% 4
frame = pd.read_csv(CSV, keep_default_na=False, usecols=['Image_Label', 'EncodedPixels'])
parts = frame['Image_Label'].str.rsplit('_', n=1, expand=True)
if parts.shape[1] != 2:
    raise ValueError('Unexpected Image_Label format; inspect the dataset version')
frame['filename'], frame['label'] = parts[0], parts[1]
if not frame.label.isin(CLASSES).all():
    raise ValueError('Unexpected cloud label; check the CSV')
if frame.duplicated(['filename', 'label']).any():
    raise ValueError('Duplicate image/category rows')
index = frame.pivot(index='filename', columns='label', values='EncodedPixels').reindex(columns=CLASSES)
if index.isna().any().any():
    raise ValueError('At least one image lacks one of its four category rows')
if len(frame) != 4 * len(index):
    raise ValueError('Expected exactly four class rows per image')
if SMOKE_IDS:
    unknown = sorted(set(SMOKE_IDS) - set(index.index))
    if unknown:
        raise ValueError(f'Smoke image IDs absent from CSV: {unknown}')
    index = index.loc[list(dict.fromkeys(SMOKE_IDS))]
if len(index) < 4:
    raise ValueError('The lab needs at least four distinct image IDs for a small holdout')
# An explicit variant must match the RLE coordinate system. The original full CSV
# must extend beyond the mirror canvas; a mirror CSV cannot pass as original.
nonempty = frame.loc[frame.EncodedPixels.astype(str).str.strip().ne(''), 'EncodedPixels']
max_end = 0
for encoded in nonempty:
    try:
        values = np.asarray([int(piece) for piece in str(encoded).split()], dtype=np.int64)
    except ValueError as exc:
        raise ValueError('RLE contains a non-integer token in the CSV') from exc
    if values.size % 2 or not values.size:
        raise ValueError('Malformed RLE in the CSV')
    ends = values[::2] + values[1::2] - 1
    if np.any(values[::2] < 1) or np.any(values[1::2] < 1):
        raise ValueError('RLE starts and lengths must be positive')
    max_end = max(max_end, int(ends.max()))
if max_end > EXPECTED_SIZE[0] * EXPECTED_SIZE[1]:
    raise ValueError(f'RLE reaches {max_end}, beyond selected {DATA_VARIANT} mask canvas')
if DATA_VARIANT == 'original' and max_end <= 576 * 384:
    raise ValueError('CSV RLE never reaches beyond the resized mirror canvas; check that this is the original CSV')
print('CSV maximum encoded pixel index:', max_end)
missing_files = [name for name in index.index if not (IMAGES / name).is_file()]
if missing_files:
    raise FileNotFoundError(f'{len(missing_files)} referenced images are missing; first: {missing_files[0]}')
print(f'{len(index):,} selected images / {4*len(index):,} selected image-category rows')
if SMOKE_IDS: print(f'Full CSV has {len(frame):,} rows; subsequent summaries use the selected smoke images only')
print('first filenames:', index.index[:3].tolist())
assert len(index.columns) == 4

# %% [markdown] 5
# ## 2 · Presence is not area
# 
# First count images with each category using only CSV labels. This runs without decoding every huge mask. “Two categories in one image” means co-occurrence; it does not establish that the masks share pixels.
# 
# **Predict:** Which category is rarest by image count? Does that guarantee it has the fewest labelled pixels?

# %% 6
presence = index.apply(lambda column: column.astype(str).str.strip().ne(''))
counts = presence.sum().sort_values()
summary = pd.DataFrame({'images_present': counts, 'fraction_of_images': counts / len(index)})
print(summary.to_string())
print('fully label-empty images:', int((presence.sum(axis=1) == 0).sum()))
print('images with 2+ labels:', int((presence.sum(axis=1) >= 2).sum()))
print('presence cardinality:', presence.sum(axis=1).value_counts().sort_index().to_dict())
fig, ax = plt.subplots(1, 2, figsize=(10, 3.3))
summary.images_present.plot.bar(ax=ax[0], title='Images with label')
presence.sum(axis=1).value_counts().sort_index().plot.bar(ax=ax[1], title='Number of labels per image')
ax[0].set_ylabel('image count'); ax[1].set_xlabel('labels present')
plt.tight_layout(); plt.show()

# %% [markdown] 7
# ## 3 · Verify geometry and RLE against actual pictures
# 
# The chosen variant has its own dimensions for **both image and RLE mask**. Read the file and verify the selected contract before decoding. The original training grid is 2100×1400 (width×height); the mirror's CSV was re-encoded at 576×384. Neither is the official competition's downscaled 525×350 scoring grid. The decoder below assumes the dataset's **1-based, column-major (`F`)** convention; the round-trip and overlay are required checks. The [official Kaggle overview](https://www.kaggle.com/c/understanding_cloud_organization/overview) describes the traversal order; a [winning solution’s label resize utility](https://github.com/pudae/kaggle-understanding-clouds/blob/master/tools/resize_labels.py) shows the original 1400×2100 label geometry and nearest-neighbor mask resize. A round-trip alone can hide a matched encoder/decoder bug, so inspect the overlay visually too.
# 
# **Predict:** For an image with width `W` and height `H`, what shape must one mask have?

# %% 8
def decode_rle(text, shape):
    h, w = shape
    flat = np.zeros(h*w, dtype=np.uint8)
    if pd.isna(text) or not str(text).strip():
        return flat.reshape(shape, order='F')
    values = np.asarray(str(text).split(), dtype=np.int64)
    if values.size % 2:
        raise ValueError('RLE must contain start/length pairs')
    previous_end = 0
    for start, length in values.reshape(-1, 2):
        lo, hi = int(start)-1, int(start)-1+int(length)
        if lo < previous_end or length <= 0 or hi > flat.size:
            raise ValueError('RLE run out of bounds, overlapping or unordered')
        flat[lo:hi] = 1
        previous_end = hi
    return flat.reshape(shape, order='F')

def encode_rle(mask):
    flat = (np.asarray(mask) > 0).astype(np.uint8).reshape(-1, order='F')
    edges = np.flatnonzero(np.diff(np.r_[0, flat, 0])) + 1
    edges[1::2] -= edges[::2]
    return ' '.join(map(str, edges))

assert encode_rle(decode_rle('8 2', (3, 5))) == '8 2'
example_name = presence.any(axis=1).idxmax()
with Image.open(IMAGES / example_name) as original:
    image = original.convert('RGB')
    width, height = image.size
    if (width, height) != EXPECTED_SIZE:
        raise ValueError(f'Image {(width, height)} does not match {DATA_VARIANT} size {EXPECTED_SIZE}; '
                         'check the image folder and CSV source')
    preview = image.copy()
print('example:', example_name, '| width:', width, '| height:', height, '| ratio:', round(width/height, 3))
example_masks = np.stack([decode_rle(index.loc[example_name, label], (height, width)) for label in CLASSES])
for label, mask in zip(CLASSES, example_masks):
    assert encode_rle(mask) == str(index.loc[example_name, label]).strip()
    print(label, 'pixels:', int(mask.sum()))
assert example_masks.shape == (4, height, width)
print('pixel-level overlap in this image:', int((example_masks.sum(axis=0) >= 2).sum()))
class_colors = {'Fish': '#e76f51', 'Flower': '#48a868', 'Gravel': '#4387d0', 'Sugar': '#9b62c8'}
fig, axes = plt.subplots(2, 3, figsize=(15, 7))
axes = axes.ravel()
axes[0].imshow(preview); axes[0].set_title('actual image')
axes[1].imshow(preview); axes[1].set_title('all four labels (overlap can mix colors)')
for i, (label, mask) in enumerate(zip(CLASSES, example_masks)):
    overlay = np.ma.masked_where(mask == 0, mask)
    cmap = ListedColormap(['white', class_colors[label]])
    axes[1].imshow(overlay, cmap=cmap, vmin=0, vmax=1, alpha=.45, interpolation='nearest')
    axes[i+2].imshow(preview)
    axes[i+2].imshow(overlay, cmap=cmap, vmin=0, vmax=1, alpha=.5, interpolation='nearest')
    axes[i+2].set_title(f'{label} · {int(mask.sum()):,} pixels')
for a in axes: a.axis('off')
fig.legend(handles=[Patch(facecolor=class_colors[label], label=label) for label in CLASSES],
           loc='lower center', bbox_to_anchor=(.5, .015), ncol=4, fontsize=9)
fig.tight_layout(rect=(0, .07, 1, 1), h_pad=2.0, w_pad=1.0); plt.show()

# %% [markdown] 9
# **Visual check:** Do the translucent regions track plausible cloud regions? Each class has a fixed color and its own panel; color mixing in the combined view suggests overlapping labels. If the overlay is striped or rotated, stop and inspect the source RLE convention. The assertion checks only internal encode/decode agreement, not semantic alignment.
# 
# **Try:** Choose another image name from `index.index`, especially one with two categories, and repeat the overlay. The code above remains runnable without any edits.

# %% [markdown] 10
# ## 4 · Split original image IDs once
# 
# We split **filenames**, not the four annotation rows or later crops. This short lab uses a fixed random split and reports category representation. A serious model study may need stratification and grouped sources; inspect the counts before trusting a rare-class score. The validation images will never be used to construct the spatial prior.
# 
# **Predict:** Why might a rare class disappear from a small validation sample even when it exists in the full data?

# %% 11
rng = np.random.default_rng(30)
ids = rng.permutation(index.index.to_numpy())
n_val = max(1, int(round(.2 * len(ids))))
train_ids, val_ids = ids[n_val:], ids[:n_val]
assert len(set(train_ids) & set(val_ids)) == 0
representation = pd.DataFrame({'train_images': presence.loc[train_ids].sum(),
                               'validation_images': presence.loc[val_ids].sum()})
print('train/validation source images:', len(train_ids), len(val_ids))
print(representation.to_string())
print('Some validation class counts may be small; interpret those results cautiously.')
if SMOKE_IDS: print('SMOKE HOLDOUT: only the listed real images were split; no generalization estimate')

# %% [markdown] 12
# ## 5 · A deliberately weak baseline
# 
# A **spatial pixel prior** averages a few training masks at a smaller grid. It uses no image texture and learns no model weights. It asks only “where did this class appear in the sampled training images?” The values are uncalibrated scores, not probabilities of a defect in a new image. Unlike a class-global prevalence number, this map retains location and can inherit coarse alignment bias from the few training masks. We evaluate a few untouched validation images and a handful of cutoffs. The sample sizes are intentionally small for CPU practice; the result is highly unstable and cannot support a competition claim.
# 
# We resize each binary mask with **nearest-neighbor** before averaging, so the labels stay binary. The source image is never used by this baseline. Keep the large native masks one at a time.

# %% 13
SMALL_SIZE = (105, 70)  # (width, height); compare ratio to native dimensions
N_TRAIN_DEMO, N_VAL_DEMO = 12, 6
train_demo = train_ids[:N_TRAIN_DEMO]
val_demo = val_ids[:N_VAL_DEMO]
assert len(train_demo) > 0 and len(val_demo) > 0
assert set(train_demo).isdisjoint(val_demo)

def small_masks(filename):
    with Image.open(IMAGES / filename) as source:
        w, h = source.size
    if (w, h) != EXPECTED_SIZE:
        raise ValueError(f'{filename}: geometry {(w, h)} does not match {DATA_VARIANT} size {EXPECTED_SIZE}')
    output = []
    for label in CLASSES:
        mask = decode_rle(index.loc[filename, label], (h, w))
        reduced = Image.fromarray(mask).resize(SMALL_SIZE, Image.Resampling.NEAREST)
        output.append(np.asarray(reduced, dtype=np.uint8))
    return np.stack(output)

prior = np.zeros((4, SMALL_SIZE[1], SMALL_SIZE[0]), dtype=np.float32)
for name in train_demo:
    prior += small_masks(name)
prior /= len(train_demo)
validation_masks = {name: small_masks(name) for name in val_demo}
print('train-only sample:', len(train_demo), '| held-out sample:', len(val_demo))
print('prior grid C,H,W:', prior.shape, '| mean scores by class:', np.round(prior.mean(axis=(1,2)), 5))
print('sampled training pixel coverage on small grid:', dict(zip(CLASSES, np.round(prior.mean(axis=(1,2)), 5))))
assert prior.shape == (4, 70, 105)
assert prior.min() >= 0 and prior.max() <= 1

# %% [markdown] 14
# ## 6 · Make the validation choice visible
# 
# For one class and image, hard Dice is `2 × overlap / (predicted area + true area)`. Here both-empty gets 1; one-empty gets 0. This convention makes a one-pixel false alarm on a class-absent image/mask pair costly. Report foreground Dice and empty-mask accuracy separately so a high mean score cannot hide a “predict nothing” strategy. These are **separate teaching diagnostics**. The cloud competition officially averages per-image, per-class Dice (with empty/empty scored as 1); we keep foreground and absent-class results apart so a trivial all-empty strategy is visible. These tiny-sample columns are not a measured competition score.
# 
# **Predict:** Which cutoff will produce the most empty predictions? What is the cost on positive validation masks?

# %% 15
def hard_dice(pred, truth):
    p, g = np.asarray(pred, dtype=bool), np.asarray(truth, dtype=bool)
    denominator = int(p.sum()+g.sum())
    return 1.0 if denominator == 0 else 2*int((p&g).sum())/denominator

def evaluate_cutoff(cutoff):
    p = prior >= cutoff
    foreground, clean = [], []
    for name, masks in validation_masks.items():
        for channel in range(4):
            gt = masks[channel] > 0
            if gt.any(): foreground.append(hard_dice(p[channel], gt))
            else: clean.append(int(not p[channel].any()))
    return {'cutoff': cutoff, 'foreground_pairs': len(foreground),
            'foreground_dice': np.mean(foreground) if foreground else np.nan,
            'empty_pairs': len(clean), 'empty_accuracy': np.mean(clean) if clean else np.nan}

results = pd.DataFrame([evaluate_cutoff(t) for t in (.1, .25, .5, .75, 1.01)])
print(results.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
print('1.01 forces an all-empty prediction; compare both columns before judging it.')
if SMOKE_IDS: print('SMOKE RESULT ONLY: a four-image holdout cannot estimate class performance')
assert set(results.columns) == {'cutoff', 'foreground_pairs', 'foreground_dice', 'empty_pairs', 'empty_accuracy'}

# Keep a small evidence record beside the student's notebook. It contains IDs and
# diagnostic numbers only; no image pixels, masks or RLE strings are copied.
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
manifest = {
    'dataset_variant': DATA_VARIANT,
    'native_size_wh': list(EXPECTED_SIZE),
    'rle_contract': '1-based, column-major/F, four independent channels',
    'split_seed': 30,
    'selected_image_ids': index.index.tolist(),
    'train_image_ids': train_ids.tolist(),
    'validation_image_ids': val_ids.tolist(),
    'train_demo_ids': train_demo.tolist(),
    'validation_demo_ids': val_demo.tolist(),
    'small_grid_wh': list(SMALL_SIZE),
    'smoke_mode': bool(SMOKE_IDS),
    'diagnostic_scope': ('four-image code-path smoke; no performance inference' if SMOKE_IDS
                         else 'tiny sampled holdout; illustrative diagnostics only'),
    'diagnostic_cutoffs': [
        {key: (None if pd.isna(value) else int(value) if key in ('foreground_pairs', 'empty_pairs')
               else float(value)) for key, value in row.items()}
        for row in results.to_dict(orient='records')
    ],
}
manifest_path = OUTPUT_ROOT / 'session1_evidence.json'
manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
print('Saved split and diagnostic manifest:', manifest_path.resolve())

# %% [markdown] 16
# ## 7 · Evidence record and exit ticket
# 
# Write a sentence for each question in your own copy of the notebook. The final code cell saves `session1_evidence.json` under `CLOUD_OUTPUT_ROOT` (default `runs/bootcamp_session1_practice/`), recording the split, sampled IDs, dataset variant and diagnostic numbers without copying images or masks:
# 
# 1. Which class is rarest by **image presence**? Did the small validation sample contain it?
# 2. What are the native `(H, W)` and aspect ratio you actually observed? How would a square resize alter geometry?
# 3. In your overlay, do two labels co-occur in the image? Did any pixels overlap? Those are different observations.
# 4. At which cutoff did empty accuracy improve, and what happened to foreground Dice?
# 5. What might a score of `NaN` mean in this tiny sample? Why is the spatial prior weak even if one number looks high?
# 
# **Hint after attempting:** Small samples may contain no positive pair for a category or even across the chosen images; `NaN` then says there is no denominator for that diagnostic, not that the method succeeded. The prior ignores actual image content. Keep the split fixed before trying any idea; the hidden test set is not a place to choose settings.
# 
# For further study, use the Milestone 1 guides on image geometry, RLE, Dice and split design. The next bootcamp meeting turns these observations into a checked data notebook; it does not require a trained model today.
