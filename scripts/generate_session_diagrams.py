# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy==2.2.6", "matplotlib==3.10.9"]
# ///
"""Deterministic synthetic teaching diagrams; contains no competition data."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Ellipse, FancyArrowPatch
OUT=Path(__file__).resolve().parents[1]/'docs/live_session_materials/figures'
OUT.mkdir(parents=True,exist_ok=True)
PAPER='#fffefa';INK='#172d3a';TEAL='#087f8c';RED='#df653c';PURPLE='#73419d'
plt.rcParams.update({'font.size':11,'figure.facecolor':PAPER,'axes.facecolor':PAPER,'text.color':INK,'axes.spines.top':False,'axes.spines.right':False})
def save(fig,name):
    fig.tight_layout(pad=2);fig.savefig(OUT/f'{name}.png',dpi=160,bbox_inches='tight');plt.close(fig)
def grid(ax,data,title,indices=None):
    ax.imshow(data,cmap='Blues',vmin=0,vmax=1,interpolation='nearest')
    H,W=data.shape
    ax.set_xticks(np.arange(-.5,W,1),minor=True);ax.set_yticks(np.arange(-.5,H,1),minor=True);ax.grid(which='minor',color='#8b999e',lw=1)
    ax.set_xticks(range(W));ax.set_yticks(range(H));ax.set_xlabel('x (column)');ax.set_ylabel('y (row)');ax.set_title(title)
    if indices is not None:
        for y in range(H):
            for x in range(W):ax.text(x,y,str(indices[y,x]),ha='center',va='center',color='white' if data[y,x] else INK,fontsize=13)
# Long / wide transformation
fig,axs=plt.subplots(1,3,figsize=(15,5));pres=np.array([[0,0,0,0],[1,0,0,0],[1,0,1,0],[0,1,1,1]])
for ax in axs:ax.axis('off')
rows=[[f'strip_{i+1}',c,'mask' if pres[i,j] else 'empty'] for i in range(4) for j,c in enumerate('ABCD')]
t=axs[0].table(cellText=rows,colLabels=['image_id','class','mask_rle'],loc='center',cellLoc='center');t.auto_set_font_size(False);t.set_fontsize(9);t.scale(1,1.25);axs[0].set_title('16 long-format annotation rows',pad=25)
t=axs[1].table(cellText=[[f'strip_{i+1}',*pres[i]] for i in range(4)],colLabels=['image','A','B','C','D'],loc='center',cellLoc='center');t.scale(1,2);axs[1].set_title('4 image-level presence vectors',pad=25)
axs[2].text(.05,.8,'Row sums → cardinality\n[0, 1, 2, 3]\n\nColumn sums → class positives\n[2, 1, 2, 1]\n\nTotal ones → positive pairs\n6 of 16\n\nAll-zero rows → clean images\n1 of 4',va='top',fontsize=13);axs[2].set_title('Choose the unit before counting',pad=25);save(fig,'long_to_wide')
# Accuracy trap
m=np.zeros((10,10));m[4:6,4:6]=1
fig,axs=plt.subplots(1,3,figsize=(13,4));grid(axs[0],m,'Ground truth: 4 defect pixels');grid(axs[1],np.zeros_like(m),'Prediction: background everywhere');axs[2].axis('off');axs[2].text(.03,.9,'TP = 0   FP = 0\nFN = 4   TN = 96\n\nPixel accuracy = 96%\nForeground recall = 0%\nForeground Dice = 0\n\nUseful detection can be zero\neven with high accuracy.',va='top',fontsize=13);save(fig,'accuracy_trap')
# Overlap targets
A=np.zeros((4,8));A[:3,1:3]=1;C=np.zeros_like(A);C[1:,5:7]=1;C2=np.zeros_like(A);C2[1:,2:4]=1
fig,axs=plt.subplots(2,3,figsize=(13,6))
for row,c in enumerate([C,C2]):
    grid(axs[row,0],A,'Mask A');grid(axs[row,1],c,'Mask C');occup=A+c;axs[row,2].imshow(occup,cmap='viridis',vmin=0,vmax=2,interpolation='nearest');axs[row,2].set_title(('Disjoint: legal integer targets' if row==0 else 'Overlap: integer map loses a label'))
    for y in range(4):
        for x in range(8):axs[row,2].text(x,y,str(int(occup[y,x])),ha='center',va='center',color='white' if occup[y,x]<2 else INK)
    axs[row,2].set_xlabel('Occupancy sum, not class ID')
save(fig,'overlap_targets')
# Tensor flow diagram
fig,ax=plt.subplots(figsize=(13,5));ax.set_xlim(0,13);ax.set_ylim(0,5);ax.axis('off')
boxes=[(.2,2,2.5,1.3,'Input strip batch\nB × 1 × H × W'),(4,3.2,3.5,1.3,'Four defect logits\nB × 4 × H × W'),(9,3.2,3.5,1.3,'Binary floating targets\nB × 4 × H × W\nBCEWithLogitsLoss'),(4,.4,3.5,1.3,'BG + four defect logits\nB × 5 × H × W'),(9,.4,3.5,1.3,'Integer target IDs 0–4\nB × H × W\nCrossEntropyLoss')]
for x,y,w,h,text in boxes:ax.add_patch(Rectangle((x,y),w,h,facecolor='#e9f3f1',edgecolor=TEAL,lw=1.5));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=12)
for a,b in [((2.7,2.7),(4,3.7)),((2.7,2.7),(4,1)),((7.5,3.8),(9,3.8)),((7.5,1),(9,1))]:ax.add_patch(FancyArrowPatch(a,b,arrowstyle='->',mutation_scale=18,color=INK))
ax.set_title('Choose target semantics before the output head',fontsize=17);save(fig,'head_tensors')
# Full RLE trace
m=np.zeros((4,6),int);m[2:4,0]=1;m[0:2,1]=1;m[1:3,2]=1
fig=plt.figure(figsize=(14,9));gs=fig.add_gridspec(3,2,height_ratios=[3,1,1]);grid(fig.add_subplot(gs[0,0]),m,'F order: down columns',np.arange(1,25).reshape(4,6,order='F'));grid(fig.add_subplot(gs[0,1]),m,'C order: across rows',np.arange(1,25).reshape(4,6))
for row,order in enumerate(['F','C'],start=1):
    ax=fig.add_subplot(gs[row,:]);bits=m.ravel(order=order);ax.set_xlim(-.5,23.5);ax.set_ylim(-.5,1.5);ax.axis('off');pairs=[]
    for i,b in enumerate(bits):ax.add_patch(Rectangle((i-.45,0),.9,.65,facecolor=TEAL if b else '#e2e7e5'));ax.text(i,.33,str(b),ha='center',va='center',color='white' if b else INK);ax.text(i,-.25,str(i+1),ha='center',fontsize=9)
    edges=np.flatnonzero(np.diff(np.r_[0,bits,0]));pairs=list(zip(edges[::2]+1,edges[1::2]-edges[::2]));ax.set_title(f'{order} buffer → RLE: '+ ' '.join(f'{s} {n}' for s,n in pairs),loc='left')
save(fig,'rle_full_trace')
# Boundary diagram
fig,ax=plt.subplots(figsize=(12,3));ax.axis('off');ax.set_xlim(0,12);ax.set_ylim(0,3)
for j,k in enumerate(range(21,27)):
    color=TEAL if k in [23,24] else RED if k>24 else '#e2e7e5'
    ax.add_patch(Rectangle((j*1.7,.8),1.5,1,facecolor=color));ax.text(j*1.7+.75,1.3,str(k),ha='center',va='center',color=INK if k<23 else 'white',fontsize=19)
ax.text(0,2.5,'Run 23 4 asks for 23, 24, 25, 26. Canvas ends at 24.',fontsize=15);ax.text(0,.1,'Strict: reject overflow. Documented tolerance: keep 23, 24; clip [22:26] → [22:24].',fontsize=12);ax.plot([6.7,6.7],[.6,2.1],'--',color=INK);save(fig,'rle_boundary')
# Sensor geometry
fig,axs=plt.subplots(1,2,figsize=(13,4))
ax=axs[0];ax.axis('off');ax.set_xlim(0,10);ax.set_ylim(0,6)
for i in range(12):ax.add_patch(Rectangle((.8+i*.6,1),.4,3.6,facecolor=plt.cm.Greys(.3+i*.04)))
ax.annotate('successive capture times →',xy=(8,5.2),xytext=(.8,5.2),arrowprops=dict(arrowstyle='->'));ax.text(.1,.2,'One captured line per time step → a 2D strip\nIllustrative acquisition, not verified camera metadata',fontsize=11);ax.set_title('Line scans build a surface view')
axs[1].bar(['Horizontal sx','Vertical sy'],[.4,2.4],color=[TEAL,RED]);axs[1].set_title('600 × 100 → 240 × 240');axs[1].set_ylabel('Scale factor');axs[1].text(-.15,1.55,'Relative stretch\nR = 2.4 / 0.4 = 6',fontsize=13);save(fig,'sensor_geometry')
# Geometric coordinate illustrations — vector shapes separate from sampling experiment.
fig,axs=plt.subplots(1,3,figsize=(14,6))
for ax,(W,H,title) in zip(axs,[(600,100,'Native: 600 × 100'),(240,240,'Square warp: 240 × 240'),(300,50,'Isotropic: 300 × 50')]):
    sx,sy=W/600,H/100;ax.set_xlim(0,W);ax.set_ylim(H,0);ax.set_aspect('equal');ax.set_facecolor('#e2e7e5');ax.add_patch(Ellipse((200*sx,50*sy),80*sx,80*sy,facecolor=TEAL));ax.add_patch(Rectangle((500*sx,12*sy),2*sx,76*sy,facecolor=RED));ax.set_title(title);ax.set_xlabel(f'sx={sx:g}, sy={sy:g}, R={sy/sx:g}\nGeometric crack width={2*sx:g}px');ax.set_xticks([]);ax.set_yticks([])
fig.suptitle('Geometry shown continuously; survival requires a discrete sampling experiment',fontsize=14);save(fig,'geometry_comparison')
# Contrast cases
fig,axs=plt.subplots(1,3,figsize=(14,4));x=np.linspace(-4,4,500);gauss=lambda sd:np.exp(-.5*(x/sd)**2)/(sd*np.sqrt(2*np.pi))
axs[0].plot(x,gauss(1),color=TEAL,label='background');axs[0].plot(x,np.exp(-.5*((x-1)/1)**2)/np.sqrt(2*np.pi),color=RED,label='defect');axs[0].set_title('Different means: d = 1');axs[0].legend();axs[0].set_xlabel('Standardized intensity')
axs[1].plot(x,gauss(.5),color=TEAL,label='narrow');axs[1].plot(x,gauss(1.5),color=RED,label='wide');axs[1].set_title('Equal means, different spread: d = 0');axs[1].legend();axs[1].set_xlabel('Standardized intensity')
pattern=np.tile(np.array([75,75,175,175]),(12,6));scramble=np.random.default_rng(42).permutation(pattern.ravel()).reshape(pattern.shape);combo=np.concatenate([pattern,scramble],axis=0);axs[2].imshow(combo,cmap='gray',vmin=0,vmax=255,interpolation='nearest');axs[2].set_title('Equal histogram, changed arrangement');axs[2].set_xticks([]);axs[2].set_yticks([5,17],['ordered','shuffled']);save(fig,'contrast_cases')
# Receptive fields
fig,axs=plt.subplots(1,4,figsize=(14,4));texture=np.indices((11,11)).sum(0)%2
for ax,r in zip(axs,[1,3,5,7]):
    ax.imshow(texture,cmap='gray',vmin=0,vmax=2);ax.add_patch(Rectangle((5-r/2,5-r/2),r,r,fill=False,edgecolor=RED,lw=3));ax.set_title(f'{r} × {r} receptive field');ax.set_xticks([]);ax.set_yticks([])
fig.suptitle('1 × 1 observes brightness; 3 × 3 layers expand neighborhood context',fontsize=14);save(fig,'receptive_field')


# Fully synthetic grayscale image, native toy pixels and numbered mask overlay.
rng=np.random.default_rng(42)
image=rng.normal(90,8,(6,8));mask=np.zeros((6,8),int);mask[1:5,3:5]=1;image+=mask*65
rgb=np.repeat((image/255)[...,None],3,axis=2);rgb[mask>0]=rgb[mask>0]*.4+np.array([1,.25,.1])*.6
fig,axes=plt.subplots(1,2,figsize=(12,5))
for ax,arr,title in zip(axes,[image,rgb],['Synthetic intensities','Same pixels + binary mask']):
    ax.imshow(arr,cmap='gray',vmin=0,vmax=255)
    for y in range(6):
        for x in range(8): ax.text(x,y,str(x*6+y+1),ha='center',va='center',color='white',bbox=dict(facecolor='black',alpha=.5,pad=1))
    ax.set_xticks(np.arange(-.5,8,1),minor=True);ax.set_yticks(np.arange(-.5,6,1),minor=True);ax.grid(which='minor',color='white');ax.set_xticks([]);ax.set_yticks([]);ax.set_title(title)
fig.suptitle('F-order indices on a fictional 8 × 6 image: k = 6x + y + 1');save(fig,'synthetic_pixel_grid')
# Correct vs scrambled decode on a generated toy strip.
image=rng.normal(90,9,(32,192));mask=np.zeros((32,192),int);mask[6:27,70:73]=1;mask[13:18,125:145]=1;image+=mask*65
fig,axes=plt.subplots(3,1,figsize=(12,6))
for ax,m,title in zip(axes,[np.zeros_like(mask),mask,mask.ravel(order='F').reshape(mask.shape)],['Synthetic strip','F-order decode matches the signal','Same buffer, C-order reshape: wrong positions']):
    rgb=np.repeat((image/255)[...,None],3,axis=2);rgb[m>0]=rgb[m>0]*.4+np.array([1,.25,.1])*.6;ax.imshow(rgb);ax.axis('off');ax.set_title(title)
save(fig,'synthetic_rle_order')
# Phase experiment on fictional 600 × 100 sensor geometry, matching the lab.
fig,axes=plt.subplots(1,2,figsize=(12,4));scores=[]
for x in range(20):
    m=np.zeros((100,600),int);m[12:88,x:x+2]=1
    ys=np.floor((np.arange(240)+.5)*100/240).astype(int);xs=np.floor((np.arange(240)+.5)*600/240).astype(int);small=m[np.ix_(ys,xs)]
    back=small[np.ix_(np.floor((np.arange(100)+.5)*240/100).astype(int),np.floor((np.arange(600)+.5)*240/600).astype(int))]
    scores.append(2*(m*back).sum()/(m.sum()+back.sum()))
axes[0].bar(range(20),scores,color=RED);axes[0].set(xlabel='Toy crack x-position',ylabel='Round-trip Dice',ylim=(0,1),title='Same 2px width, different phase')
widths=np.array([1,2,3,4,5,6,8,10]);axes[1].plot(widths,widths*240/600,'o-',color=TEAL);axes[1].axhline(1,color=RED,ls='--');axes[1].set(xlabel='Toy source width (px)',ylabel='Geometric target width (px)',title='Fictional resize: 600 → 240');save(fig,'sampling_phase')
# Same multiset of intensities, different arrangements: no project examples.
checker=((np.indices((32,64)).sum(0)%2)*100+75).astype(np.uint8);shuffled=rng.permutation(checker.ravel()).reshape(checker.shape)
fig,axes=plt.subplots(1,3,figsize=(13,4))
for ax,img,title in zip(axes[:2],[checker,shuffled],['Ordered synthetic texture','Shuffled synthetic texture']):ax.imshow(img,cmap='gray',vmin=0,vmax=255);ax.axis('off');ax.set_title(title)
axes[2].hist(checker.ravel(),bins=np.arange(0,260,10),alpha=.5,label='ordered');axes[2].hist(shuffled.ravel(),bins=np.arange(0,260,10),histtype='step',lw=2,label='shuffled');axes[2].legend();axes[2].set(title='Identical histogram',xlabel='Intensity');save(fig,'texture_same_histogram')
import json
(OUT/'manifest.json').write_text(json.dumps({'source':'Deterministic synthetic teaching examples only','seed':42,'sensor_example':{'width':600,'height':100},'figures':[p.name for p in sorted(OUT.glob('*.png'))]},indent=2))
print('Generated synthetic-only figures:',OUT)
