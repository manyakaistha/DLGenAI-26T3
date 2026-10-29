# /// script
# requires-python = ">=3.10"
# dependencies = ["torch==2.8.0", "numpy==2.2.6", "Pillow==12.3.0"]
# ///
"""Optional Clouds DDP teaching worker. Run through torchrun; one process per GPU.
CPU/Gloo is a synchronization demonstration, never a multi-GPU speed measurement.
Download cloud_training.py into the same directory (the notebook writes both files).
"""
from __future__ import annotations
import argparse, json, math, os, random
from dataclasses import asdict
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from cloud_training import (Config,CloudDataset,TinyCloudNet,prepare,set_seed,precision_policy,
                            make_scheduler,train_epoch,validation_stats,summarize,atomic_save)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu',action='store_true');parser.add_argument('--resume')
    parser.add_argument('--stop-after',type=int)
    # File rendezvous is also usable for bounded independent CPU-process checks on macOS.
    parser.add_argument('--init-file');parser.add_argument('--rank',type=int);parser.add_argument('--world-size',type=int)
    args=parser.parse_args();c=Config.from_env().validate()
    if args.init_file:
        rank=args.rank;world=args.world_size;local=rank
        if rank is None or world is None:raise ValueError('File rendezvous needs --rank and --world-size')
    else:rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    if world<2:raise ValueError('Use at least two processes for this DDP lesson; use cloud_training.py for one device.')
    if args.cpu:device=torch.device('cpu');backend='gloo'
    else:
        if not torch.cuda.is_available():raise RuntimeError('No CUDA device. Explicitly use --cpu for the Gloo mechanics demo.')
        torch.cuda.set_device(local);device=torch.device('cuda',local);backend='nccl'
    kwargs={'init_method':'file://'+str(Path(args.init_file).resolve()),'rank':rank,'world_size':world} if args.init_file else {}
    dist.init_process_group(backend=backend,**kwargs)
    try:
        torch.set_num_threads(1);set_seed(c.seed);records,manifest=prepare(c)
        trainset=CloudDataset(c,records,manifest['train_image_ids'])
        # Both drops are explicit. All ranks perform the same number of forward/backward calls.
        sampler=DistributedSampler(trainset,num_replicas=world,rank=rank,shuffle=True,seed=c.seed,drop_last=True)
        loader=DataLoader(trainset,batch_size=c.batch,sampler=sampler,shuffle=False,drop_last=True,num_workers=0)
        if not len(loader):raise ValueError('Need at least world_size × batch training images; lower BATCH for a tiny smoke.')
        val_ids=manifest['validation_image_ids'][rank::world]  # unique, non-padding evaluation shard
        vloader=DataLoader(CloudDataset(c,records,val_ids),batch_size=c.batch,shuffle=False,num_workers=0)
        context,scaler=precision_policy(c,device)
        raw=TinyCloudNet().to(device);model=DDP(raw,device_ids=[local] if device.type=='cuda' else None)
        optimizer=torch.optim.AdamW(model.parameters(),lr=c.lr)
        scheduler=make_scheduler(optimizer,c.epochs*math.ceil(len(loader)/c.accumulation),c.warmup_updates)
        start=0;best=float('inf');bad=0;history=[]
        if args.resume:
            state=torch.load(args.resume,map_location='cpu',weights_only=True)
            if state['config']!=asdict(c) or state['manifest']!=manifest or state['world_size']!=world:
                raise ValueError('DDP resume requires the same config, source/split and world size.')
            raw.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);scaler.load_state_dict(state['scaler'])
            start=state['next_epoch'];best=state['best'];bad=state['bad_epochs'];history=state['history']
            rng=state['rank_rng'][rank];torch.set_rng_state(rng['torch']);random.setstate(rng['python'])
            if device.type=='cuda':torch.cuda.set_rng_state(rng['cuda'],device)
        out=Path(c.output)
        if rank==0:out.mkdir(parents=True,exist_ok=True);(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
        print(json.dumps({'rank':rank,'world_size':world,'device':str(device),'validation_ids':val_ids,'scope':manifest['scope']}),flush=True)
        if bad>=c.patience:return
        for epoch in range(start,c.epochs):
            sampler.set_epoch(epoch)  # new shared shuffle before each rank selects its shard
            shard=[trainset.names[i] for i in sampler];used=shard[:len(loader)*c.batch]
            print(json.dumps({'rank':rank,'epoch':epoch+1,'training_ids_used':used}),flush=True)
            train=train_epoch(model,loader,optimizer,scheduler,context,scaler,c,device)
            # raw model avoids DDP forward collectives while validation shards have uneven lengths.
            stats=validation_stats(raw,vloader,context,device);dist.all_reduce(stats,op=dist.ReduceOp.SUM)
            metrics=summarize(stats);assert metrics['images']==len(manifest['validation_image_ids'])
            n=sum(train['window_samples_per_rank']);training=torch.tensor([train['train_loss']*n,n],dtype=torch.float64,device=device)
            dist.all_reduce(training);train['train_loss']=float(training[0]/training[1])
            flat=torch.cat([p.detach().reshape(-1) for p in raw.parameters()])
            copies=[torch.empty_like(flat) for _ in range(world)];dist.all_gather(copies,flat)
            max_diff=max(float((v-flat).abs().max()) for v in copies)
            if max_diff!=0:raise RuntimeError('Replicas diverged; inspect gradients, update order and rank control flow')
            improved=metrics['val_loss']<best
            if improved:best=metrics['val_loss'];bad=0
            else:bad+=1
            row={'epoch':epoch+1,**train,**metrics,'replica_max_abs_difference':max_diff,
                 'world_size':world,'effective_batch_full_window':c.batch*c.accumulation*world,
                 'train_images_used_global':int(training[1]),'train_images_omitted':len(trainset)-int(training[1])}
            history.append(row)
            rng={'torch':torch.get_rng_state(),'python':random.getstate(),'cuda':torch.cuda.get_rng_state(device) if device.type=='cuda' else None}
            all_rng=[None]*world;dist.all_gather_object(all_rng,rng)  # collective OUTSIDE rank-zero branch
            if rank==0:
                state={'config':asdict(c),'manifest':manifest,'world_size':world,'model':raw.state_dict(),
                       'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),
                       'next_epoch':epoch+1,'best':best,'bad_epochs':bad,'history':history,'rank_rng':all_rng}
                atomic_save(state,out/'last.pt')
                if improved:atomic_save(state,out/'best.pt')
                (out/'history.json').write_text(json.dumps(history,indent=2,allow_nan=False));print(json.dumps(row),flush=True)
            dist.barrier()
            if (args.stop_after and epoch+1>=args.stop_after) or bad>=c.patience:break
    finally:dist.destroy_process_group()

if __name__=='__main__':main()
