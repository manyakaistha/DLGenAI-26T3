"""Generate clean, self-contained Clouds lesson notebooks with stable Colab cell IDs."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'docs/efficient_training_session'


def cell(kind,text,id):
    value={'cell_type':kind,'id':id,'metadata':{'id':id},'source':text.splitlines(keepends=True)}
    if kind=='code':value.update(execution_count=None,outputs=[])
    return value


def build():
    runtime=(SOURCE/'cloud_training.py').read_text();worker=(SOURCE/'train_cloud_ddp.py').read_text()
    setup="from pathlib import Path\n"+f"Path('cloud_training.py').write_text({runtime!r})\n"+'import cloud_training as ct\nimport torch, copy, json, os\nfrom dataclasses import replace\nprint(torch.__version__, "CUDA:", torch.cuda.is_available())\n'
    data='''# SETUP ONCE, then use the named demonstration sections below.
# Point at matching train.csv and train_images/ from ONE cloud dataset variant.
cfg = ct.Config(
    data_root=os.environ.get('CLOUD_DATA_ROOT', 'data/bootcamp/clouds_384x576'),
    variant=os.environ.get('CLOUD_DATA_VARIANT', 'resized_384x576'),
    output=os.environ.get('CLOUD_OUTPUT_ROOT', 'runs/cloud_efficiency'),
    smoke_ids=os.environ.get('CLOUD_SMOKE_IMAGE_IDS', ''),
    split_manifest=os.environ.get('CLOUD_SPLIT_MANIFEST', ''),
    precision='fp32', epochs=3, batch=2, accumulation=2)
records, manifest = ct.prepare(cfg)
print(json.dumps(manifest, indent=2))
train_set = ct.CloudDataset(cfg, records, manifest['train_image_ids'])
x, y, image_id = train_set[0]
print(image_id, 'input', tuple(x.shape), 'target', tuple(y.shape))
assert x.shape == (3, 64, 96) and y.shape == (4, 64, 96)
assert set(y.unique().tolist()) <= {0., 1.}
'''
    core=[
      cell('markdown','''# Train efficiently, keep the evidence · 29 October 2026
60-minute core session. Real **Understanding Clouds from Satellite Images** data,
four independent targets (Fish, Flower, Gravel, Sugar). This is a small training
mechanics lesson, not a competition solution or a promised speedup.

Run Setup first. Later slide links identify the section and cell to revisit.
The notebook writes its complete, editable runtime to `cloud_training.py` beside
this notebook. The code is embedded below, so no project checkout is required.

Download the 384×576 mirror yourself, or explicitly select original 2100×1400
files. Never mix CSV/RLE variants. No raw data or synthetic fallback is bundled.
A `CLOUD_SPLIT_MANIFEST` can reuse bootcamp `session1_evidence.json` image splits.

Hardware: FP32 works on CPU; BF16 CPU autocast demonstrates execution only.
CUDA FP16 uses GradScaler; native CUDA BF16 uses no scaler in this lesson.
Unsupported CUDA/precision requests fail clearly. Keep outputs private if they
contain downloaded images or fitted results. [Study guide](https://manyakaistha.github.io/DLGenAI-26T3/session-4/).
''','overview'),
      cell('markdown','''## 00 · Setup — run these cells first
Use the existing compatible PyTorch runtime if available. If dependencies are
missing, run the next installation cell, restart after replacing imported binary
packages, then run the runtime and dataset cells. Network access is needed for
installation and your separate Kaggle download, not for pretrained weights.
''','setup'),
      cell('code','''# Run only if this runtime needs the teaching dependencies.
# %pip install torch==2.8.0 numpy==2.2.6 Pillow==12.3.0
''','install'),
      cell('code',setup,'runtime'),cell('code',data,'data'),
      cell('markdown','''## 01 · FP16 versus BF16: range is not precision
Predict the cast of 1.001, 100000 and 1e-8 before running. Both low-precision
formats use 16 bits. FP16 has 5 exponent + 10 stored fraction bits; BF16 has 8 + 7.
`finfo.tiny` means smallest **normal**, not smallest subnormal. Tensor casts
illustrate representation; they do not emulate an entire AMP training run.
''','precision-demo'),
      cell('code','ct.numeric_demo()\n','precision-code'),
      cell('markdown','''## 02 · AMP on one real cloud batch
Start from the same random seed, batch and number of updates for every policy.
Read the implementation before benchmarking: autocast wraps forward/loss;
backward is outside. Our per-image BCE+Dice reduces logits in FP32.

Three warmup updates precede the measured training steps. CUDA timers synchronize
before and after measurement; peak memory counts allocated tensors, not all GPU
reserved/device memory. A tiny model can be overhead-bound. No fixed speedup is
expected. Unsupported policies are labelled skipped. CPU timings are not CUDA
Tensor Core evidence. Compare finite loss and validate quality separately.
''','amp-demo'),
      cell('code','''import inspect
print(inspect.getsource(ct.precision_policy))
benchmark_rows = ct.benchmark(cfg, steps=10)
''','amp-code'),
      cell('markdown','''## 03 · Accumulation, including an uneven last microbatch
Compare gradients for the SAME real samples in a physical batch and microbatches.
For a window of N images, weight each microbatch mean loss by B_i/N, not always
1/K. This handles smaller final microbatches. We compare gradients before clipping
or updating. The model has no BatchNorm/dropout and the loss is per-image separable;
those restrictions matter. Roundoff means allclose, not identical bits.
''','accumulation-demo'),
      cell('code','''from torch.utils.data import DataLoader
ct.set_seed(29)
x, y, ids = next(iter(DataLoader(train_set, batch_size=min(5, len(train_set)))))
a = ct.TinyCloudNet(); b = copy.deepcopy(a)
ct.sample_losses(a(x), y).mean().backward()
for xx, yy in zip(x.split(2), y.split(2)):
    (ct.sample_losses(b(xx), yy).mean() * (len(xx)/len(x))).backward()
errors = [(pa.grad-pb.grad).abs().max().item() for pa,pb in zip(a.parameters(),b.parameters())]
assert all(torch.allclose(pa.grad,pb.grad,atol=2e-6,rtol=2e-5) for pa,pb in zip(a.parameters(),b.parameters()))
print('Samples:', len(x), 'microbatch sizes:', [len(t) for t in x.split(2)])
print('Maximum gradient difference:', max(errors))
print(inspect.getsource(ct.train_epoch))
''','accumulation-code'),
      cell('markdown','''## 04 · Interrupt, resume, then compare with an uninterrupted run
Use fresh output folders for a new experiment. The complete schedule horizon is
three epochs from the beginning; interruption after epoch one does not change it.
The saved checkpoint restores model, optimizer, scheduler, scaler, next epoch,
best value, patience, split/config/source fingerprint and RNG state.

This is epoch-boundary resume with deterministic sample order, workers=0 and no
augmentation. Exact arbitrary mid-epoch/distributed/hardware replay is not promised.
The monitored value is validation loss; best.pt and last.pt serve different purposes.
''','resume-demo'),
      cell('code','''resume_cfg = replace(cfg, output=str(Path(cfg.output)/'resume_demo'))
reference_cfg = replace(cfg, output=str(Path(cfg.output)/'reference_demo'))
ct.run(resume_cfg, stop_after=1)
checkpoint = Path(resume_cfg.output)/'last.pt'
saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
print('Restoring keys:', list(saved), 'next epoch:', saved['next_epoch'])
resumed_history = ct.run(resume_cfg, resume=checkpoint)
reference_history = ct.run(reference_cfg)
r = torch.load(checkpoint, map_location='cpu', weights_only=True)['model']
u = torch.load(Path(reference_cfg.output)/'last.pt', map_location='cpu', weights_only=True)['model']
resume_error = max((r[k]-u[k]).abs().max().item() for k in r)
print('Resumed vs uninterrupted maximum weight difference:', resume_error)
if cfg.device=='cpu' or not torch.cuda.is_available():
    assert all(torch.allclose(r[k],u[k],atol=1e-7,rtol=1e-6) for k in r)
''','resume-code'),
      cell('markdown','''## 05 · Inspect the schedule and validation evidence
The scheduler steps after a successful optimizer update, not every microbatch.
A skipped FP16 update does not advance it. Print a fresh schedule without training.
Then read REAL held-out loss, foreground-pair Dice, empty-pair specificity and
class support from the measured run. An absent denominator is null, not success.
These are fixed 64×96 teaching-grid diagnostics, not official cloud or industrial
competition scores. Best weights need not be last weights.
''','validation-demo'),
      cell('code','''probe = torch.optim.SGD(ct.TinyCloudNet().parameters(), lr=.001)
schedule = ct.make_scheduler(probe, total=12, warmup=2)
lrs=[]
for update in range(12):
    lrs.append(probe.param_groups[0]['lr']); probe.step(); schedule.step()
print('Learning rates used:', lrs)
print('Actual final validation record:', resumed_history[-1])
print('Best checkpoint exists:', (Path(resume_cfg.output)/'best.pt').is_file())
''','validation-code'),
      cell('markdown','''## Reflect
What did AMP change? Did memory, time and quality move together? Why does an
incomplete accumulation window need its actual denominator? What state was needed
to resume? Choose one next experiment and keep split, data order, model start,
effective batch and validation protocol fixed. DDP is a separate optional notebook.
''','reflect')]
    ddp=[cell('markdown','''# Optional · Clouds DDP from first principles to a complete worker
Outside the 60-minute core. Use the SAME real cloud images, RLE parser, four-output
model and loss as the core notebook. DDP means replicated models and synchronized
gradients, not pooled GPU memory. The worker below is fully embedded and editable.

This notebook launches a separate script using torchrun; it does not launch
notebook-defined functions through multiprocessing. Read the optional guide before
changing sampler, validation or rank control flow. CPU/Gloo demonstrates mechanics;
CUDA/NCCL needs at least two visible GPUs and a tested runtime. No GPU speed result
is invented or substituted with CPU timing.
''','ddp-overview'),
      cell('markdown','''## D0 · Setup once
Run the compatible dependency setup if needed, write both files, and configure
real cloud paths. You can reuse the bootcamp split via CLOUD_SPLIT_MANIFEST.
''','ddp-setup'),cell('code','# %pip install torch==2.8.0 numpy==2.2.6 Pillow==12.3.0\n','ddp-install'),
      cell('code',setup+f"Path('train_cloud_ddp.py').write_text({worker!r})\n",'ddp-runtime'),
      cell('code',data+'''# Explicit CPU mechanics default. Choose 'cuda' only with 2+ visible GPUs.
DDP_DEMO_DEVICE = 'cpu'
if DDP_DEMO_DEVICE == 'cuda' and torch.cuda.device_count() < 2:
    raise RuntimeError('Select a two-GPU runtime, or explicitly choose cpu for Gloo.')
# Two ranks × one image × two microbatches = global batch 4 on full windows.
# Tiny smoke selections may have smaller final windows and an explicitly dropped tail.
env = dict(os.environ, CLOUD_DATA_ROOT=str(Path(cfg.data_root).resolve()),
           CLOUD_DATA_VARIANT=cfg.variant, CLOUD_OUTPUT_ROOT=str(Path(cfg.output)/'ddp_demo'),
           CLOUD_SMOKE_IMAGE_IDS=cfg.smoke_ids, CLOUD_SPLIT_MANIFEST=cfg.split_manifest,
           BATCH='1', ACCUMULATION='2', EPOCHS='3', PRECISION='fp32')
''','ddp-data'),
      cell('markdown','''## D1 · See how image indices are partitioned
One global shuffle, then different rank shards. set_epoch changes the permutation.
The example drops the training tail to equalize work; inspect what is omitted.
Validation instead assigns unique, non-padding shards and sums metric numerators
and denominators before division. A mean of unequal rank means is generally wrong.
''','ddp-shards'),
      cell('code','''from torch.utils.data import DistributedSampler
for epoch in [0,1]:
    rank_sets=[]
    for rank in range(2):
        sampler=DistributedSampler(train_set,num_replicas=2,rank=rank,shuffle=True,seed=29,drop_last=True)
        sampler.set_epoch(epoch); indices=list(sampler);rank_sets.append(set(indices))
        print('epoch',epoch,'rank',rank,'indices',indices[:12])
    assert not rank_sets[0].intersection(rank_sets[1])
    print('Training indices omitted:',len(train_set)-sum(map(len,rank_sets)))
''','ddp-shards-code'),
      cell('markdown','''## D2 · Inspect the complete worker, then launch two processes
Read device assignment → process group → DistributedSampler → DDP → train →
unique-shard validation → all_reduce → rank-zero checkpoint → cleanup.
The worker asserts that replicas agree after updates and validation counts each
selected image exactly once. no_sync includes forward and backward and is used
only before the last microbatch of a window. Loss scaling weights actual window
samples; equal per-rank batch sizes make DDP gradient averaging appropriate.

Run on Linux for the ordinary CPU/Gloo torchrun demo. Some macOS installations
need the file-rendezvous alternative documented in the guide. The CUDA path is
provided for a two-GPU runtime; CPU execution does not verify CUDA performance.
''','ddp-launch'),
      cell('code','''import sys, subprocess
print(Path('train_cloud_ddp.py').read_text())
command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2','train_cloud_ddp.py']
if DDP_DEMO_DEVICE=='cpu': command += ['--cpu']
subprocess.run(command+['--stop-after','1'],env=env,check=True)
''','ddp-launch-code'),
      cell('markdown','''## D3 · Resume collectively, save once
Every rank reloads the same weights/optimizer/scheduler/scaler and its own RNG state.
World size, effective batch, schedule horizon and split must match. All ranks take
part in collectives; only file writing is rank-zero-only. Read the worker's
all_gather_object BEFORE the rank-zero condition. Changing that order can hang.
''','ddp-resume'),
      cell('code','''ddp_last=str(Path(env['CLOUD_OUTPUT_ROOT'])/'last.pt')
subprocess.run(command+['--resume',ddp_last],env=env,check=True)
measured=json.loads((Path(env['CLOUD_OUTPUT_ROOT'])/'history.json').read_text())
print(json.dumps(measured,indent=2))
assert all(row['replica_max_abs_difference']==0 for row in measured)
assert all(row['images']==len(manifest['validation_image_ids']) for row in measured)
''','ddp-resume-code'),
      cell('markdown','''## D4 · Extend carefully
First match the global batch and compare one-GPU FP32 with two-process FP32 under
the same data protocol. Then test AMP, accumulation/no_sync and hardware throughput
one at a time. Ordinary DDP retains a full model/optimizer per rank; BatchNorm
statistics are local unless an explicit strategy is chosen. This tiny network has
no BatchNorm/dropout. Training-tail drops mean its seen sample set can differ from
a single-device run: report the omitted count, not an equivalence claim.

Troubleshoot hangs by checking equal training iterations, collective order, rank
failures and device assignment. Reduce global sums/counts; don't average rank Dice
ratios. CPU results prove only the tested mechanics. Multi-node rendezvous, elastic
membership, exact mid-epoch recovery and sharded training are beyond this demo.
''','ddp-extend')]
    for name,cells in [('cloud_training',core),('cloud_ddp',ddp)]:
        doc={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.10'},'colab':{'name':name+'.ipynb','provenance':[]}},'cells':cells}
        (SOURCE/(name+'.ipynb')).write_text(json.dumps(doc,indent=1,ensure_ascii=False)+'\n')
        print(name,len(cells),'cells')
if __name__=='__main__':build()
