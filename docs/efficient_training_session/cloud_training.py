# /// script
# requires-python = ">=3.10"
# dependencies = ["torch==2.8.0", "numpy==2.2.6", "Pillow==12.3.0"]
# ///
"""Clouds teaching runtime: real-data AMP, accumulation and epoch-boundary resume.
No industrial data, pretrained download, leaderboard submission or synthetic fallback.
"""
from __future__ import annotations
import copy, csv, hashlib, json, math, os, random, time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

CLASSES = ('Fish','Flower','Gravel','Sugar')
SIZES = {'resized_384x576': (576,384), 'original': (2100,1400)}

@dataclass
class Config:
    data_root: str = 'data/bootcamp/clouds_384x576'
    variant: str = 'resized_384x576'
    output: str = 'runs/cloud_efficiency'
    precision: str = 'fp32'
    device: str = 'auto'
    seed: int = 29
    height: int = 64
    width: int = 96
    batch: int = 2
    accumulation: int = 2
    epochs: int = 3
    train_cap: int = 32
    val_cap: int = 16
    lr: float = .001
    warmup_updates: int = 2
    patience: int = 5
    smoke_ids: str = ''
    split_manifest: str = ''

    @classmethod
    def from_env(cls):
        c=cls()
        for field,key in [('data_root','CLOUD_DATA_ROOT'),('variant','CLOUD_DATA_VARIANT'),
                          ('output','CLOUD_OUTPUT_ROOT'),('precision','PRECISION'),
                          ('device','TRAIN_DEVICE'),('smoke_ids','CLOUD_SMOKE_IMAGE_IDS'),
                          ('split_manifest','CLOUD_SPLIT_MANIFEST')]:
            setattr(c,field,os.environ.get(key,getattr(c,field)))
        for field in ['seed','height','width','batch','accumulation','epochs','train_cap','val_cap','warmup_updates','patience']:
            setattr(c,field,int(os.environ.get(field.upper(),getattr(c,field))))
        return c

    def validate(self):
        if self.variant not in SIZES: raise ValueError('Choose original or resized_384x576, using matching CSV and images.')
        if self.precision not in ('fp32','fp16','bf16'): raise ValueError('PRECISION must be fp32, fp16 or bf16')
        if min(self.batch,self.accumulation,self.epochs,self.height,self.width,self.patience)<1: raise ValueError('Batch, accumulation, epochs, geometry and patience must be positive')
        if self.width*2 != self.height*3: raise ValueError('Preserve the cloud 3:2 width:height aspect ratio (e.g. H64/W96).')
        if min(self.train_cap,self.val_cap,self.warmup_updates)<0: raise ValueError('Caps and warmup cannot be negative')
        return self


def rle_runs(text, pixels):
    if not text.strip(): return []
    tokens=text.split()
    if len(tokens)%2: raise ValueError('RLE needs start/length pairs')
    try: values=[int(v) for v in tokens]
    except ValueError as e: raise ValueError('RLE needs integer tokens') from e
    runs=[];previous=0
    for start,length in zip(values[::2],values[1::2]):
        end=start+length-1
        if start<1 or length<1 or end>pixels or start<=previous:
            raise ValueError('RLE outside canvas, nonpositive, unsorted or overlapping')
        runs.append((start-1,end));previous=end
    return runs


def decode(text, size_wh):
    w,h=size_wh;flat=np.zeros(w*h,np.uint8)
    for start,end in rle_runs(text,w*h): flat[start:end]=1
    return flat.reshape((h,w),order='F')


def prepare(c):
    c.validate();root=Path(c.data_root).expanduser();csv_path=root/'train.csv'
    if not csv_path.is_file() or not (root/'train_images').is_dir():
        raise FileNotFoundError(f'{root}: download the selected real Clouds variant; expected train.csv and train_images/. No synthetic fallback.')
    records={};max_end=0;size=SIZES[c.variant]
    with csv_path.open(newline='') as f:
        reader=csv.DictReader(f)
        if not {'Image_Label','EncodedPixels'}.issubset(reader.fieldnames or []): raise ValueError('Unexpected cloud CSV schema')
        for row in reader:
            name,label=row['Image_Label'].rsplit('_',1)
            if Path(name).name!=name or label not in CLASSES: raise ValueError('Bad image/category key')
            entry=records.setdefault(name,{})
            if label in entry: raise ValueError('Duplicate image/category row')
            text=row['EncodedPixels'].strip();runs=rle_runs(text,size[0]*size[1])
            if runs:max_end=max(max_end,runs[-1][1])
            entry[label]=text
    if any(set(v)!=set(CLASSES) for v in records.values()): raise ValueError('Every image needs all four category rows; empty RLE is allowed')
    if c.variant=='original' and max_end<=576*384: raise ValueError('CSV coordinates look like the mirror; do not mix source variants')
    names=sorted(records);smoke=[x.strip() for x in c.smoke_ids.split(',') if x.strip()]
    if smoke:
        if len(smoke)!=len(set(smoke)) or not set(smoke)<=set(names): raise ValueError('Smoke IDs must be distinct IDs present in CSV')
        names=sorted(smoke)
    if c.split_manifest:
        old=json.loads(Path(c.split_manifest).read_text())
        if old['dataset_variant']!=c.variant:raise ValueError('Saved split variant differs')
        train=list(old['train_image_ids']);val=list(old['validation_image_ids'])
        if set(train)&set(val) or len(set(train))!=len(train) or len(set(val))!=len(val): raise ValueError('Split overlaps or contains duplicates')
        if not set(train+val)<=set(names): raise ValueError('Saved split IDs absent from selected data')
    else:
        if len(names)<4:raise ValueError('At least four images required for the teaching holdout')
        rng=random.Random(c.seed);rng.shuffle(names);n=max(1,round(.2*len(names)))
        val=names[:n];train=names[n:]
    # Caps are applied only after assigning image IDs to disjoint partitions.
    train=train[:c.train_cap or None];val=val[:c.val_cap or None]
    if not train or not val:raise ValueError('Empty training/validation selection')
    for name in train+val:
        path=root/'train_images'/name
        if not path.is_file():raise FileNotFoundError(f'{path}: incomplete download; use explicit smoke IDs only for a code-path check')
        with Image.open(path) as image:
            if image.size!=size:raise ValueError(f'{name}: image {image.size} does not match {c.variant} {size}')
    manifest={'dataset_variant':c.variant,'native_size_wh':list(size),'train_image_ids':train,
              'validation_image_ids':val,'csv_sha256':hashlib.sha256(csv_path.read_bytes()).hexdigest(),
              'smoke_mode':bool(smoke),'scope':'code-path smoke; no performance inference' if smoke else 'capped teaching holdout; not a competition score',
              'training_size_hw':[c.height,c.width],'classes':list(CLASSES),'split_source':c.split_manifest or f'image split seed {c.seed}'}
    return records,manifest

class CloudDataset(Dataset):
    def __init__(self,c,records,names):self.c=c;self.records=records;self.names=list(names)
    def __len__(self):return len(self.names)
    def __getitem__(self,i):
        name=self.names[i];size=SIZES[self.c.variant];target=(self.c.width,self.c.height)
        with Image.open(Path(self.c.data_root)/'train_images'/name) as im:
            if im.size!=size:raise ValueError('Dataset geometry changed')
            x=np.asarray(im.convert('RGB').resize(target,Image.Resampling.BILINEAR),dtype=np.float32)/255
        masks=[np.asarray(Image.fromarray(decode(self.records[name][label],size)).resize(target,Image.Resampling.NEAREST),dtype=np.float32) for label in CLASSES]
        return torch.from_numpy(x.transpose(2,0,1).copy()),torch.from_numpy(np.stack(masks)),name

class TinyCloudNet(nn.Module):
    """Small encoder/decoder, four independent logits; no BN/dropout in equivalence demos."""
    def __init__(self):
        super().__init__();self.encoder=nn.Sequential(nn.Conv2d(3,12,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(12,24,3,padding=1),nn.ReLU())
        self.decoder=nn.Sequential(nn.Conv2d(24,12,3,padding=1),nn.ReLU(),nn.Conv2d(12,4,1))
    def forward(self,x):return self.decoder(F.interpolate(self.encoder(x),size=x.shape[-2:],mode='bilinear',align_corners=False))


def sample_losses(logits,y):
    # Per-image objective: mean BCE + mean channel-wise soft Dice loss.
    z=logits.float();y=y.float();p=z.sigmoid()
    bce=F.binary_cross_entropy_with_logits(z,y,reduction='none').mean((1,2,3))
    dice=(2*(p*y).sum((2,3))+1e-6)/(p.sum((2,3))+y.sum((2,3))+1e-6)
    return bce+1-dice.mean(1)


def precision_policy(c,device):
    if c.precision=='fp16' and device.type!='cuda':raise ValueError('FP16 training demo requires CUDA. Use fp32 or CPU bf16; no silent fallback.')
    if c.precision=='bf16' and device.type=='cuda' and not torch.cuda.is_bf16_supported(including_emulation=False):
        raise ValueError('Native CUDA BF16 unsupported on this GPU; explicitly select fp16 or fp32.')
    dtype={'fp32':torch.float32,'fp16':torch.float16,'bf16':torch.bfloat16}[c.precision]
    def context():return torch.autocast(device.type,dtype=dtype) if c.precision!='fp32' else nullcontext()
    scaler=torch.amp.GradScaler(device.type,enabled=device.type=='cuda' and c.precision=='fp16')
    return context,scaler


def set_seed(seed):
    random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def make_scheduler(optimizer,total,warmup):
    def factor(step):
        if warmup and step<warmup:return (step+1)/warmup
        t=min(1,max(0,(step-warmup)/max(1,total-warmup)))
        return .1+.9*.5*(1+math.cos(math.pi*t))
    return torch.optim.lr_scheduler.LambdaLR(optimizer,factor)


def train_epoch(model,loader,optimizer,scheduler,context,scaler,c,device):
    model.train();iterator=iter(loader);loss_sum=0.;seen=0;updates=0;skipped=0;groups=[]
    while True:
        window=list(islice(iterator,c.accumulation))
        if not window:break
        count=sum(len(x) for x,_,_ in window);groups.append(count)
        optimizer.zero_grad(set_to_none=True)
        for j,(x,y,_) in enumerate(window):
            x=x.to(device);y=y.to(device);sync=model.no_sync() if hasattr(model,'no_sync') and j<len(window)-1 else nullcontext()
            # DDP no_sync must include forward AND backward.
            with sync:
                with context():losses=sample_losses(model(x),y);loss=losses.mean()*(len(x)/count)
                scaler.scale(loss).backward()
            loss_sum+=float(losses.detach().sum());seen+=len(x)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        old_scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
        did_step=scaler.get_scale()>=old_scale
        if did_step:scheduler.step();updates+=1
        else:skipped+=1
    if not seen:raise ValueError('No training batches: reduce batch/world size or provide more training images')
    return {'train_loss':loss_sum/seen,'updates':updates,'skipped_updates':skipped,'window_samples_per_rank':groups}


def validation_stats(model,loader,context,device):
    model.eval();stats=torch.zeros(10,dtype=torch.float64,device=device)
    with torch.inference_mode():
        for x,y,_ in loader:
            x=x.to(device);y=y.to(device)
            with context():z=model(x)
            loss=sample_losses(z,y);p=z.float().sigmoid()>=.5;g=y.bool()
            ps=p.sum((2,3));gs=g.sum((2,3));intersection=(p&g).sum((2,3));positive=gs>0;empty=~positive
            dice=2*intersection/(ps+gs).clamp_min(1)
            stats[0]+=loss.double().sum();stats[1]+=len(x)
            stats[2]+=dice[positive].double().sum();stats[3]+=positive.sum()
            stats[4]+=((ps==0)&empty).sum();stats[5]+=empty.sum();stats[6:]+=positive.sum(0)
    return stats


def summarize(stats):
    s=stats.cpu().tolist()
    return {'val_loss':s[0]/s[1] if s[1] else None,'images':int(s[1]),
            'foreground_dice':s[2]/s[3] if s[3] else None,'foreground_pairs':int(s[3]),
            'empty_specificity':s[4]/s[5] if s[5] else None,'empty_pairs':int(s[5]),
            'positive_pairs_by_class':dict(zip(CLASSES,map(int,s[6:])))}


def rng_state():return {'torch':torch.get_rng_state(),'python':random.getstate(),
                        'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

def restore_rng(s):
    torch.set_rng_state(s['torch']);random.setstate(s['python'])
    if s['cuda'] and torch.cuda.is_available():torch.cuda.set_rng_state_all(s['cuda'])

def atomic_save(state,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp');torch.save(state,temp);temp.replace(path)


def run(c=None,resume=None,stop_after=None):
    c=(c or Config.from_env()).validate();torch.set_num_threads(min(4,os.cpu_count() or 1));set_seed(c.seed)
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if c.device=='auto' else c.device)
    records,manifest=prepare(c);out=Path(c.output);out.mkdir(parents=True,exist_ok=True)
    context,scaler=precision_policy(c,device);model=TinyCloudNet().to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=c.lr)
    total=c.epochs*math.ceil(math.ceil(len(manifest['train_image_ids'])/c.batch)/c.accumulation)
    scheduler=make_scheduler(optimizer,total,c.warmup_updates);start=0;best=float('inf');bad=0;history=[]
    if resume:
        state=torch.load(resume,map_location='cpu',weights_only=True)
        if state['config']!=asdict(c) or state['manifest']!=manifest:raise ValueError('Resume requires the saved config, split, source hash and schedule horizon; do not change epochs/precision/batch.')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);scaler.load_state_dict(state['scaler'])
        start=state['next_epoch'];best=state['best'];bad=state['bad_epochs'];history=state['history'];restore_rng(state['rng'])
    valset=CloudDataset(c,records,manifest['validation_image_ids']);vloader=DataLoader(valset,batch_size=c.batch,shuffle=False,num_workers=0)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'device':str(device),'precision':c.precision,'gradient_scaling':scaler.is_enabled(),'scope':manifest['scope']}))
    if bad>=c.patience:print('Saved early-stopping patience is exhausted; no further updates.');return history
    for epoch in range(start,c.epochs):
        generator=torch.Generator().manual_seed(c.seed+epoch)
        loader=DataLoader(CloudDataset(c,records,manifest['train_image_ids']),batch_size=c.batch,shuffle=True,generator=generator,num_workers=0)
        train=train_epoch(model,loader,optimizer,scheduler,context,scaler,c,device)
        metrics=summarize(validation_stats(model,vloader,context,device));improved=metrics['val_loss']<best
        if improved:best=metrics['val_loss'];bad=0
        else:bad+=1
        row={'epoch':epoch+1,**train,**metrics,'next_lr':optimizer.param_groups[0]['lr']};history.append(row)
        state={'config':asdict(c),'manifest':manifest,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
               'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'next_epoch':epoch+1,'best':best,
               'bad_epochs':bad,'history':history,'rng':rng_state()}
        atomic_save(state,out/'last.pt')
        if improved:atomic_save(state,out/'best.pt')
        (out/'history.json').write_text(json.dumps(history,indent=2,allow_nan=False));print(json.dumps(row))
        if (stop_after and epoch+1>=stop_after) or bad>=c.patience:break
    return history


def numeric_demo():
    rows=[]
    for dtype in [torch.float32,torch.float16,torch.bfloat16]:
        f=torch.finfo(dtype);values=torch.tensor([1.001,1e5,1e-8],dtype=torch.float32).to(dtype).float().tolist()
        row={'dtype':str(dtype),'max':f.max,'smallest_normal':f.tiny,'epsilon_at_one':f.eps,'casts_1.001_1e5_1e-8':values};rows.append(row);print(row)
    return rows


def benchmark(c,steps=10):
    """Matched real batch / initial seed / update count. End-to-end training-step timing, not dataloading."""
    records,m=prepare(c);x,y,_=next(iter(DataLoader(CloudDataset(c,records,m['train_image_ids']),batch_size=c.batch)))
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if c.device=='auto' else c.device);rows=[]
    for mode in ['fp32','fp16','bf16']:
        cfg=copy.copy(c);cfg.precision=mode
        try:context,scaler=precision_policy(cfg,device)
        except ValueError as e:rows.append({'precision':mode,'status':'skipped','reason':str(e)});continue
        set_seed(c.seed);model=TinyCloudNet().to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=c.lr);a=x.to(device);b=y.to(device)
        def step():
            optimizer.zero_grad(set_to_none=True)
            with context():loss=sample_losses(model(a),b).mean()
            scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
            return loss
        for _ in range(3):step()
        if device.type=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        begin=time.perf_counter()
        for _ in range(steps):loss=step()
        if device.type=='cuda':torch.cuda.synchronize()
        seconds=time.perf_counter()-begin
        rows.append({'precision':mode,'device':str(device),'steps':steps,'seconds':seconds,'images_per_second':steps*len(x)/seconds,
                     'peak_allocated_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else None,'final_loss':float(loss.detach()),'finite':bool(torch.isfinite(loss))})
        del model,optimizer,scaler,a,b
        if device.type=='cuda':torch.cuda.empty_cache()
    print(json.dumps(rows,indent=2));return rows

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--demo',choices=['numbers','benchmark','train'],default='train');parser.add_argument('--resume');parser.add_argument('--stop-after',type=int)
    args=parser.parse_args()
    if args.demo=='numbers':numeric_demo()
    elif args.demo=='benchmark':benchmark(Config.from_env())
    else:run(resume=args.resume,stop_after=args.stop_after)
