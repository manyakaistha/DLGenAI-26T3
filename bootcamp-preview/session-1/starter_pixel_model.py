# Generated from the matching notebook by scripts/build_bootcamp_notebooks.py.

# %% [markdown] 1
# # Session 1 · Train a tiny pixel model
# 
# **Starter baseline after the data practice lab · CPU · ~35–50 minutes**
# 
# This is the first *trainable* model in the bootcamp: four independent logistic heads applied to each RGB pixel. A pixel has features `[1, R, G, B]` and four binary targets. Mathematically this is a **1×1 convolution without spatial context**. It can learn color associations, but cannot understand cloud shape, texture or neighbouring pixels. The target regions in this dataset are human-marked coarse areas, so even a color cue may be misleading. Scores here are a classroom diagnostic, not a competition result or a recipe for another dataset.
# 
# **Run the [practice lab](practice_dataset_lab.ipynb) first.** It saves the selected variant and split IDs to `session1_evidence.json`. This notebook requires that manifest, the matching real `train.csv` and JPEGs. It does not create data or choose a new split.
# 
# **Setup:** `CLOUD_DATA_VARIANT=resized_384x576` selects the [resized Kaggle cloud mirror](https://www.kaggle.com/datasets/phunghieu/understanding-clouds-from-satellite-images-384x576) by default; `original` selects the [official competition files](https://www.kaggle.com/competitions/understanding_cloud_organization/data). Set `CLOUD_DATA_ROOT` to the folder with that variant's `train.csv` and `train_images/`. `CLOUD_OUTPUT_ROOT` points to the practice manifest folder (default `runs/bootcamp_session1_practice/`). Kaggle access and rules are handled by the learner. Local script: `uv run --no-project --with numpy==2.2.6 --with pandas==2.3.3 --with matplotlib==3.10.9 --with pillow==12.3.0 python docs/bootcamp_session_1/starter_pixel_model.py`. For local Jupyter add `--with jupyterlab==4.4.9 jupyter lab` instead. A hosted notebook usually already has these packages.
# 
# **Route:** load the fixed split → make aligned small image/mask arrays → inspect shapes → optimize BCE → compare held-out threshold diagnostics → save an evidence record. Think before each code cell.

# %% 2
from pathlib import Path
import os, json
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

CLASSES = ('Fish', 'Flower', 'Gravel', 'Sugar')
CONTRACTS = {
    'resized_384x576': {'size': (576, 384), 'default_root': 'data/bootcamp/clouds_384x576'},
    'original': {'size': (2100, 1400), 'default_root': 'data/bootcamp/understanding_cloud_organization'},
}
DATA_VARIANT = os.environ.get('CLOUD_DATA_VARIANT', 'resized_384x576')
if DATA_VARIANT not in CONTRACTS:
    raise ValueError(f'Unknown CLOUD_DATA_VARIANT={DATA_VARIANT!r}; choose one of {tuple(CONTRACTS)}')
EXPECTED_SIZE = CONTRACTS[DATA_VARIANT]['size']
DATA_ROOT = Path(os.environ.get('CLOUD_DATA_ROOT', CONTRACTS[DATA_VARIANT]['default_root'])).expanduser()
OUTPUT_ROOT = Path(os.environ.get('CLOUD_OUTPUT_ROOT', 'runs/bootcamp_session1_practice')).expanduser()
PRACTICE_MANIFEST = OUTPUT_ROOT / 'session1_evidence.json'
if not PRACTICE_MANIFEST.is_file():
    raise FileNotFoundError(f'Run practice_dataset_lab.ipynb first; expected {PRACTICE_MANIFEST.resolve()}')
manifest = json.loads(PRACTICE_MANIFEST.read_text(encoding='utf-8'))
if manifest.get('dataset_variant') != DATA_VARIANT or tuple(manifest.get('native_size_wh', ())) != EXPECTED_SIZE:
    raise ValueError('Practice manifest variant/geometry does not match this dataset choice')
train_ids = list(manifest['train_demo_ids'])
val_ids = list(manifest['validation_demo_ids'])
if not train_ids or not val_ids or set(train_ids) & set(val_ids):
    raise ValueError('Practice manifest must contain disjoint nonempty train/validation demo IDs')
print('Variant:', DATA_VARIANT, '| native W×H:', EXPECTED_SIZE)
print('Fixed demonstration split:', len(train_ids), 'training,', len(val_ids), 'validation images')
if manifest.get('smoke_mode'):
    print('FOUR-IMAGE SMOKE MODE: this checks execution only, not performance')

# %% [markdown] 3
# ## 1 · Read only the chosen images and their four masks
# 
# A CSV row is one image/class pair. We use the prior lab's exact IDs and channel order. The input RGB image goes through bilinear resizing; each binary target mask goes through **nearest-neighbor** resizing. Keeping the 2:3 aspect ratio avoids stretching. The small 70×105 canvas makes this CPU lesson quick; it discards fine detail, so its metrics are not native-resolution results.
# 
# **Predict:** If one image has shape `(70,105,3)` and its masks have `(70,105,4)`, how many training rows (pixels) will the logistic model receive from that image?

# %% 4
CSV = DATA_ROOT / 'train.csv'
IMAGES = DATA_ROOT / 'train_images'
if not CSV.is_file() or not IMAGES.is_dir():
    raise FileNotFoundError(f'Missing {CSV} or {IMAGES}; use the same selected Kaggle data root as the practice lab')
frame = pd.read_csv(CSV, keep_default_na=False, usecols=['Image_Label', 'EncodedPixels'])
parts = frame.Image_Label.str.rsplit('_', n=1, expand=True)
if parts.shape[1] != 2:
    raise ValueError('Unexpected Image_Label format')
frame['filename'], frame['label'] = parts[0], parts[1]
if not frame.label.isin(CLASSES).all() or frame.duplicated(['filename','label']).any():
    raise ValueError('Unknown or duplicate image/class row')
index = frame.pivot(index='filename', columns='label', values='EncodedPixels').reindex(columns=CLASSES)
if index.isna().any().any():
    raise ValueError('Missing one of the four labels for an image')
needed = train_ids + val_ids
unknown = sorted(set(needed)-set(index.index))
if unknown:
    raise ValueError(f'Manifest IDs missing in selected CSV: {unknown[:3]}')
for name in needed:
    if not (IMAGES/name).is_file():
        raise FileNotFoundError(f'Missing selected JPEG: {IMAGES/name}')
print('Selected images and all four labels located')


def decode_rle(text, shape):
    h,w = shape
    flat = np.zeros(h*w, dtype=np.uint8)
    if pd.isna(text) or not str(text).strip():
        return flat.reshape(shape, order='F')
    pieces = str(text).split()
    try:
        values = np.asarray([int(piece) for piece in pieces], dtype=np.int64)
    except ValueError as exc:
        raise ValueError('RLE contains a non-integer token') from exc
    if len(values)%2:
        raise ValueError('RLE needs start/length pairs')
    end=0
    for start,length in values.reshape(-1,2):
        lo,hi=int(start)-1,int(start)-1+int(length)
        if lo<end or length<=0 or hi>flat.size:
            raise ValueError('RLE run unordered, overlapping or outside this variant canvas')
        flat[lo:hi]=1
        end=hi
    return flat.reshape(shape, order='F')

SMALL_SIZE = (105,70)  # W,H; same 2:3 ratio as both cloud variants

def load_small(filename):
    with Image.open(IMAGES/filename) as source:
        if source.size != EXPECTED_SIZE:
            raise ValueError(f'{filename}: image size {source.size} disagrees with {DATA_VARIANT} {EXPECTED_SIZE}')
        rgb = np.asarray(source.convert('RGB').resize(SMALL_SIZE, Image.Resampling.BILINEAR), dtype=np.float32)/255.0
    masks=[]
    for label in CLASSES:
        native=decode_rle(index.loc[filename,label], (EXPECTED_SIZE[1],EXPECTED_SIZE[0]))
        resized=Image.fromarray(native).resize(SMALL_SIZE, Image.Resampling.NEAREST)
        masks.append(np.asarray(resized, dtype=np.float32))
    y=np.stack(masks,axis=-1)
    assert rgb.shape==(70,105,3) and y.shape==(70,105,4)
    assert set(np.unique(y)).issubset({0.,1.})
    return rgb,y

train_pairs=[load_small(name) for name in train_ids]
val_pairs=[load_small(name) for name in val_ids]
print('one RGB image:',train_pairs[0][0].shape,'| one four-channel mask:',train_pairs[0][1].shape)
print('pixels per image:',70*105,'| training pixels:',len(train_pairs)*70*105)

# %% [markdown] 5
# ## 2 · Four independent logistic heads
# 
# Each pixel row is `[1,R,G,B]`. Weights have shape `(4 input features, 4 classes)`, and `logits = X @ W`. Applying a sigmoid to **each** logit allows none, one or multiple classes at one pixel. We do not use softmax or `argmax`: the cloud labels can overlap.
# 
# The binary cross-entropy for a logit `z` and target `y∈{0,1}` is `logaddexp(0,z) − yz`. Its mean is stable even for large logits. For a minibatch of `M` pixels and four classes, the gradient is `Xᵀ(sigmoid(XW)−Y)/(4M)`. `W ← W − learning_rate × gradient` is the entire learning step.
# 
# **Predict:** Why are there four output columns? What goes wrong if we share one single binary score across all classes?

# %% 6
def features_targets(pairs):
    pixels=np.concatenate([image.reshape(-1,3) for image,_ in pairs],axis=0)
    targets=np.concatenate([mask.reshape(-1,4) for _,mask in pairs],axis=0)
    x=np.column_stack([np.ones(len(pixels),dtype=np.float64),pixels]).astype(np.float64)
    return x,targets.astype(np.float64)

X_train,Y_train=features_targets(train_pairs)
X_val,Y_val=features_targets(val_pairs)
assert X_train.shape[1]==4 and Y_train.shape[1]==4
assert X_val.shape[1]==4 and Y_val.shape[1]==4
assert len(X_train)==len(train_pairs)*70*105
assert X_train.min()>=0 and X_train.max()<=1
print('train features:',X_train.shape,'targets:',Y_train.shape,'| held-out features:',X_val.shape)
print('train positive-pixel fractions:',dict(zip(CLASSES,np.round(Y_train.mean(axis=0),4))))
print('validation positive-pixel fractions (inspection only):',dict(zip(CLASSES,np.round(Y_val.mean(axis=0),4))))


def linear(x, weights):
    # Four features × four heads: tiny explicit contraction, stable on CPU BLAS variants.
    return np.einsum('nf,fc->nc', x, weights, optimize=False)

def sigmoid(logits):
    z=np.clip(logits,-30,30)
    return 1/(1+np.exp(-z))

def bce(logits,targets):
    return float(np.mean(np.logaddexp(0,logits)-targets*logits))

W=np.zeros((4,4),dtype=np.float64)
assert np.allclose(sigmoid(linear(X_train[:2],W)),.5)
print('initial mean training BCE:',round(bce(linear(X_train,W),Y_train),5))

# %% [markdown] 7
# ## 3 · Take small, repeatable training steps
# 
# We cache only the fixed small image sample from the practice manifest, then flatten one image minibatch at a time. This bounds memory by the selected sample and small grid, regardless of the full Kaggle collection. `LEARNING_RATE=2.0`, 12 epochs and a fixed shuffle seed are **teaching settings on the small sampled grid**, not universal hyperparameters. We look at held-out loss for diagnosis, but never use held-out labels in a gradient update or to refit weights.
# 
# **Predict:** If the gradient is correct and the learning rate is reasonable, what trend do you expect in training BCE? Does falling BCE guarantee a good foreground mask?

# %% 8
LEARNING_RATE=2.0
EPOCHS=12
BATCH_IMAGES=2
TRAIN_SEED=31
rng=np.random.default_rng(TRAIN_SEED)
history=[]
for epoch in range(EPOCHS):
    for chosen in np.array_split(rng.permutation(len(train_pairs)),
                                 max(1,int(np.ceil(len(train_pairs)/BATCH_IMAGES)))):
        batch=[train_pairs[int(i)] for i in chosen]
        xb,yb=features_targets(batch)
        logits=linear(xb,W)
        gradient=np.einsum('nf,nc->fc',xb,sigmoid(logits)-yb,optimize=False)/(len(xb)*4)
        W -= LEARNING_RATE*gradient
    history.append({'epoch':epoch+1,'train_bce':bce(linear(X_train,W),Y_train),
                    'validation_bce':bce(linear(X_val,W),Y_val)})
curve=pd.DataFrame(history)
print(curve.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
assert np.isfinite(W).all() and np.isfinite(curve[['train_bce','validation_bce']].to_numpy()).all()
assert curve.train_bce.iloc[-1] < bce(np.zeros_like(Y_train),Y_train)
fig,ax=plt.subplots(figsize=(6,3))
ax.plot(curve.epoch,curve.train_bce,label='training');ax.plot(curve.epoch,curve.validation_bce,label='held-out')
ax.set(xlabel='epoch',ylabel='mean BCE',title='This tiny sampled run');ax.legend();plt.tight_layout();plt.show()

# %% [markdown] 9
# ## 4 · Inspect hard masks on held-out IDs
# 
# A sigmoid produces four independent scores per pixel. A cutoff turns each into a hard mask. Empty/empty gets Dice 1; a class-absent target with any false alarm gets 0. We keep **foreground-pair Dice** and **empty-pair accuracy** separate. The cloud competition officially averages per-image/class Dice; these small-grid sample diagnostics are not an official score. Trying several cutoffs on the fixed validation set makes validation a *selection* set; do not repeat this on hidden test labels or a leaderboard.
# 
# **Predict:** Will a very high cutoff avoid false positives? What happens to recall for actual positive regions?

# %% 10
def hard_dice(pred,truth):
    p=np.asarray(pred,dtype=bool);y=np.asarray(truth,dtype=bool)
    denominator=int(p.sum()+y.sum())
    return 1.0 if denominator==0 else 2*int((p&y).sum())/denominator

val_scores=sigmoid(linear(X_val,W)).reshape(len(val_pairs),70,105,4)
val_truth=Y_val.reshape(len(val_pairs),70,105,4)>0
rows=[]
for cutoff in (.25,.5,.75,.9):
    predicted=val_scores>=cutoff
    foreground=[];absent=[]
    for image_i in range(len(val_pairs)):
        for class_i in range(4):
            gt=val_truth[image_i,:,:,class_i]
            pp=predicted[image_i,:,:,class_i]
            if gt.any():foreground.append(hard_dice(pp,gt))
            else:absent.append(int(not pp.any()))
    rows.append({'cutoff':cutoff,'foreground_pairs':len(foreground),
                 'foreground_dice':np.mean(foreground) if foreground else np.nan,
                 'absent_pairs':len(absent),'absent_accuracy':np.mean(absent) if absent else np.nan})
diagnostics=pd.DataFrame(rows)
print(diagnostics.to_string(index=False,float_format=lambda x:f'{x:.3f}'))
print('Selected IDs:',val_ids,'| smoke mode:',bool(manifest.get('smoke_mode')))
assert val_scores.shape==(len(val_ids),70,105,4)
assert np.isfinite(val_scores).all() and val_scores.min()>=0 and val_scores.max()<=1

# %% [markdown] 11
# ## 5 · Look at one result and save the evidence
# 
# The plot shows one held-out resized image, a true mask and a prediction for one class. A successful run does **not** imply a useful cloud model; a single pixel's RGB lacks neighbourhood and texture information. The plot is produced at run time but **not embedded in the distributed notebook**. The JSON record saves the fixed IDs, learning settings, loss history and diagnostic support, without image pixels or masks.
# 
# **Predict:** Might the model mark all bright regions even where a coarse class rectangle ends? What information could a model with spatial context use?

# %% 12
shown_class=0
image,_=val_pairs[0]
ground=val_truth[0,:,:,shown_class]
proposal=val_scores[0,:,:,shown_class]>=.5
fig,axes=plt.subplots(1,3,figsize=(10,3))
for ax,data,title in zip(axes,[image,ground,proposal],
                         ['held-out RGB',f'{CLASSES[shown_class]} target','prediction at cutoff 0.5']):
    if data.ndim==2:ax.imshow(data,cmap='gray',vmin=0,vmax=1,interpolation='nearest')
    else:ax.imshow(data)
    ax.set_title(title);ax.axis('off')
plt.tight_layout();plt.show()

record={
    'model':'four independent per-pixel RGB logistic heads (1x1 convolution equivalent)',
    'dataset_variant':DATA_VARIANT,'native_size_wh':list(EXPECTED_SIZE),'small_grid_wh':list(SMALL_SIZE),
    'practice_manifest':str(PRACTICE_MANIFEST),'smoke_mode':bool(manifest.get('smoke_mode')),
    'train_ids':train_ids,'validation_ids':val_ids,'feature_shape':list(X_train.shape),
    'target_shape':list(Y_train.shape),'weights_shape':list(W.shape),
    'learning_rate':LEARNING_RATE,'epochs':EPOCHS,'batch_images':BATCH_IMAGES,'train_seed':TRAIN_SEED,
    'history':history,
    'diagnostics':[{key:(None if pd.isna(value) else int(value) if key in ('foreground_pairs','absent_pairs')
                         else float(value)) for key,value in row.items()}
                   for row in diagnostics.to_dict(orient='records')],
    'scope':'real-image four-ID code smoke only' if manifest.get('smoke_mode')
            else 'small sampled holdout; no competition-performance claim',
}
OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
record_path=OUTPUT_ROOT/'starter_pixel_model_evidence.json'
record_path.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print('Saved starter-model evidence:',record_path.resolve())

# %% [markdown] 13
# ## Exit ticket
# 
# 1. Which axes belong to pixels and which to classes in `val_scores`?
# 2. Why does a sigmoid for each class permit overlap?
# 3. What changed during training? Which array held the trainable parameters?
# 4. Why can BCE fall while a hard-mask diagnostic stays weak?
# 5. What would happen if the same cloud photograph appeared in both split lists?
# 
# **Hint after attempting:** `val_scores` is `(image, row, column, class)`; each class has a separate logistic score; the `(4,4)` weight matrix `W` changed; BCE measures continuous scores, while thresholding and class imbalance change hard masks; shared source images leak. This is a starter learning exercise. A later session introduces spatial models and a full segmentation pipeline.
