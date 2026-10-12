# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy==2.2.6", "matplotlib==3.10.9", "pandas==2.3.3", "opencv-python==5.0.0.93"]
# ///
"""Reproducible teaching figures; raw data stays local, only selected illustrations export."""
from pathlib import Path
import json
import argparse
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data-root', type=Path, default=ROOT / 'data/public')
parser.add_argument('--output', type=Path, default=ROOT / 'docs/live_session_materials/figures')
args = parser.parse_args()
DATA = args.data_root
OUT = args.output
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': '#fffefa', 'axes.facecolor': '#fffefa'})

def save(fig, name):
    fig.tight_layout(pad=2)
    fig.savefig(OUT / (name + '.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

def decode(rle):
    flat = np.zeros(128 * 800, dtype=np.uint8)
    if pd.notna(rle) and str(rle).strip():
        for start, length in np.array(str(rle).split(), dtype=int).reshape(-1, 2):
            flat[start-1:min(start-1+length, flat.size)] = 1
    return flat.reshape(128, 800, order='F')

def overlay(image, mask):
    rgb = np.repeat(image[..., None], 3, axis=2).astype(float) / 255
    rgb[mask > 0] = rgb[mask > 0] * .45 + np.array([1, .25, .1]) * .55
    return rgb

frame = pd.read_csv(DATA / 'train.csv')
frame['has'] = frame.mask_rle.fillna('').str.strip().ne('')
presence = frame.pivot(index='image_id', columns='defect_type', values='has')
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
counts = presence.sum()
axes[0].bar(['A', 'B', 'C', 'D'], counts, color=['#087f8c','#dc5b3d','#5a8641','#8c5b97'])
axes[0].set(title='Positive images per class', ylabel='Images')
for i, v in enumerate(counts): axes[0].text(i, v+75, str(v), ha='center')
cardinality = presence.sum(axis=1).value_counts().reindex(range(5), fill_value=0)
axes[1].bar(range(5), cardinality, color='#087f8c'); axes[1].set(title='Labels present per image', xlabel='Cardinality', ylabel='Images')
for i, v in cardinality.items(): axes[1].text(i, v+80, str(v), ha='center')
area = {c: 0 for c in counts.index}
for row in frame[frame.has].itertuples(): area[row.defect_type] += int(decode(row.mask_rle).sum())
total = len(presence)*128*800
axes[2].bar(['A','B','C','D'], [area[c]/total*100 for c in counts.index], color='#dc5b3d')
axes[2].set(title='Foreground pixels / all sensor pixels', ylabel='Percent (%)')
save(fig, 'data_units')
manifest = {'source': 'data/public/train.csv', 'images':len(presence), 'rows':len(frame),
            'class_positive_images': counts.astype(int).to_dict(), 'cardinality':cardinality.astype(int).to_dict(),
            'class_pixels':area, 'selected_examples':{}}
fig, axes = plt.subplots(4, 2, figsize=(14, 7))
for j, c in enumerate(counts.index):
    row = frame[(frame.defect_type == c)&frame.has].iloc[0]
    image = cv2.imread(str(DATA / f'train/{row.image_id}.jpg'), 0)
    mask = decode(row.mask_rle)
    manifest['selected_examples'][c] = row.image_id
    axes[j, 0].imshow(image, cmap='gray', vmin=0, vmax=255)
    axes[j, 1].imshow(overlay(image, mask))
    for k in (0,1): axes[j,k].axis('off'); axes[j,k].set_title(f'{c} · {row.image_id}' + (' · label overlay' if k else ' · raw'))
save(fig, 'real_defects')
row = frame[(frame.defect_type == 'defect_c')&frame.has].iloc[0]
image = cv2.imread(str(DATA / f'train/{row.image_id}.jpg'),0); mask = decode(row.mask_rle)
fig, axes = plt.subplots(3,1,figsize=(13,7))
for ax, data, title in zip(axes,[image,overlay(image,mask),overlay(image,mask.ravel(order='F').reshape(128,800))],
                          ['Raw sensor strip','Correct: column-major decode','Bug: same buffer reshaped row-major']):
    ax.imshow(data,cmap='gray'); ax.set_title(title); ax.axis('off')
save(fig,'rle_order')
# A real image crop, with actual pixel indices: no coarse-cell approximation.
y, x = np.argwhere(mask)[len(np.argwhere(mask))//2]
y0, x0 = max(0,min(122,y-3)), max(0,min(792,x-4))
fig, axes = plt.subplots(1,2,figsize=(13,5))
for ax, with_mask in zip(axes,[False,True]):
    crop = image[y0:y0+6,x0:x0+8]
    ax.imshow(overlay(crop,mask[y0:y0+6,x0:x0+8]) if with_mask else crop,cmap='gray',vmin=0,vmax=255)
    for yy in range(6):
        for xx in range(8):
            ax.text(xx,yy,str((x0+xx)*128+y0+yy+1),ha='center',va='center',fontsize=8,
                    color='white',bbox=dict(facecolor='black',alpha=.55,pad=.8))
    ax.set_xticks(np.arange(-.5,8,1),minor=True); ax.set_yticks(np.arange(-.5,6,1),minor=True)
    ax.grid(which='minor',color='white',lw=.8); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title('Actual sensor-pixel indices' + (' + ground truth' if with_mask else ''))
fig.suptitle(f'{row.image_id}: 8 × 6 crop at x={x0}, y={y0}; k = 128x + y + 1',fontsize=13)
save(fig,'real_pixel_grid')
# Deterministic phase experiment: loss depends on alignment and resampling method.
fig, axes = plt.subplots(1,2,figsize=(12,4))
survival=[]
for x in range(20):
    m=np.zeros((128,800),np.uint8); m[16:112,x:x+2]=1
    down=cv2.resize(m,(256,256),interpolation=cv2.INTER_NEAREST)
    back=cv2.resize(down,(800,128),interpolation=cv2.INTER_NEAREST)
    survival.append(2*(m*back).sum()/max(1,m.sum()+back.sum()))
axes[0].bar(range(20),survival,color='#dc5b3d'); axes[0].set(xlabel='Crack x-position (sensor pixels)',ylabel='Round-trip Dice',ylim=(0,1),title='Same 2px crack, different sampling phase')
axes[1].plot([1,2,3,4,5,6,8,10],np.array([1,2,3,4,5,6,8,10])*256/800,'o-',color='#087f8c')
axes[1].axhline(1,color='#dc5b3d',ls='--'); axes[1].set(xlabel='Native width (px)',ylabel='Scaled width (px)',title='Geometric width after 800 → 256')
save(fig,'sampling_phase')
# Same exact intensity histogram; different spatial arrangements.
rng=np.random.default_rng(42)
checker=((np.indices((32,64)).sum(0)%2)*100+75).astype(np.uint8)
shuffled=rng.permutation(checker.ravel()).reshape(checker.shape)
fig,axes=plt.subplots(1,3,figsize=(13,4))
axes[0].imshow(checker,cmap='gray',vmin=0,vmax=255);axes[0].set_title('Ordered texture: mean 125')
axes[1].imshow(shuffled,cmap='gray',vmin=0,vmax=255);axes[1].set_title('Shuffled texture: mean 125')
for ax in axes[:2]:ax.axis('off')
axes[2].hist(checker.ravel(),bins=np.arange(0,260,10),alpha=.5,label='ordered')
axes[2].hist(shuffled.ravel(),bins=np.arange(0,260,10),histtype='step',lw=2,label='shuffled')
axes[2].legend();axes[2].set(title='Identical histogram; different structure',xlabel='Intensity')
save(fig,'texture_same_histogram')
fig,axes=plt.subplots(1,2,figsize=(12,4))
inside=image[mask>0];outside=image[mask==0]
axes[0].hist(outside,bins=40,density=True,alpha=.5,label='background')
axes[0].hist(inside,bins=40,density=True,alpha=.5,label='defect');axes[0].legend();axes[0].set(title=f'One real example: {row.image_id}',xlabel='Intensity',ylabel='Density')
pooled=np.sqrt(((len(inside)-1)*inside.var(ddof=1)+(len(outside)-1)*outside.var(ddof=1))/(len(inside)+len(outside)-2))
d=(inside.mean()-outside.mean())/pooled
axes[1].axis('off');axes[1].text(.05,.8,f'Inside: n={len(inside):,}, μ={inside.mean():.2f}\nOutside: n={len(outside):,}, μ={outside.mean():.2f}\nPooled SD={pooled:.2f}\nCohen d={d:.3f}\n\nSingle-image descriptive statistic.\nPixels are spatially correlated;\nthis is not a dataset-wide estimate.',va='top',fontsize=13)
save(fig,'real_contrast')
manifest['single_image_cohen_d']={'image_id':row.image_id,'d':float(d),'aggregation':'single-image pixel pooled SD'}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
print('Generated 7 figures and provenance manifest:',OUT)
