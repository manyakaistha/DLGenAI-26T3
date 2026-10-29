"""Bounded real-Clouds smoke: notebook cells, gradients, resume and two-rank Gloo.
Run with uv run --with torch==2.8.0 --with numpy==2.2.6 --with Pillow==12.3.0.
Requires the explicit four real mirror images used by the local bootcamp audit.
"""
import json,os,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'docs/efficient_training_session'
OUT=ROOT/'runs/efficiency_checks'
OUT.mkdir(parents=True,exist_ok=True)
os.environ.update(CLOUD_DATA_ROOT=str(ROOT/'data/bootcamp/clouds_mirror_audit'),CLOUD_SMOKE_IMAGE_IDS='0011165.jpg,002be4f.jpg,0031ae9.jpg,0035239.jpg',CLOUD_OUTPUT_ROOT=str(OUT/'notebook'),TRAIN_DEVICE='cpu')
sys.path.insert(0,str(SOURCE));import cloud_training as ct
import torch
# Every shipped core code cell executes against the same real smoke dataset.
with tempfile.TemporaryDirectory(prefix='cloud_notebook_') as tmp:
    prev=Path.cwd();os.chdir(tmp);namespace={'__name__':'__lesson_check__'}
    try:
        nb=json.loads((SOURCE/'cloud_training.ipynb').read_text())
        for cell in nb['cells']:
            if cell['cell_type']=='code':
                print('Executing core notebook cell:',cell['id'],flush=True)
                exec(compile(''.join(cell['source']),cell['id'],'exec'),namespace)
    finally:os.chdir(prev)
# Decoder has an independently specified non-square expected coordinate.
m=ct.decode('2 2',(5,3));assert m.shape==(3,5) and m[1,0]==1 and m[2,0]==1 and m.sum()==2
for invalid in ['0 2','1 0','15 2','1','1 3 3 1']:
    try:ct.decode(invalid,(5,3))
    except ValueError:pass
    else:raise AssertionError(invalid)
try:ct.precision_policy(ct.Config(precision='fp16'),torch.device('cpu'))
except ValueError:pass
else:raise AssertionError('CPU FP16 must fail explicitly')
# Native CPU/Gloo multi-process test with file rendezvous, including resume.
# macOS socket rendezvous is not reliable across all local installations.
def distributed(extra,tag):
    rendezvous=OUT/('rendezvous_'+tag);rendezvous.unlink(missing_ok=True)
    env=dict(os.environ,BATCH='1',ACCUMULATION='2',EPOCHS='3',PRECISION='fp32',CLOUD_OUTPUT_ROOT=str(OUT/'ddp'))
    processes=[]
    for rank in range(2):
        log=(OUT/f'ddp_{tag}_rank{rank}.log').open('w')
        cmd=[sys.executable,str(SOURCE/'train_cloud_ddp.py'),'--cpu','--init-file',str(rendezvous),'--rank',str(rank),'--world-size','2',*extra]
        processes.append((subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT),log))
    try:
        for proc,log in processes:
            rc=proc.wait(timeout=60)
            if rc:raise RuntimeError(f'DDP failed {rc}; see {log.name}')
    finally:
        for proc,log in processes:
            if proc.poll() is None:proc.kill();proc.wait()
            log.close()
distributed(['--stop-after','1'],'start')
distributed(['--resume',str(OUT/'ddp/last.pt')],'resume')
history=json.loads((OUT/'ddp/history.json').read_text())
assert [r['epoch'] for r in history]==[1,2,3]
assert all(r['replica_max_abs_difference']==0 and r['images']==1 and r['train_images_used_global']==2 and r['train_images_omitted']==1 for r in history)
# Check notebook cell IDs for all presentation handoffs.
for name in ['cloud_training','cloud_ddp']:
    nb=json.loads((SOURCE/(name+'.ipynb')).read_text());ids=[c['id'] for c in nb['cells']]
    assert len(ids)==len(set(ids))
    assert all(c['metadata']['id']==c['id'] for c in nb['cells'])
    assert all(c.get('execution_count') is None and not c.get('outputs') for c in nb['cells'])
result={'scope':'four real mirror images; code-path correctness only','core_code_cells':'passed','resume_weight_error':namespace['resume_error'],'real_gradient_max_error':max(namespace['errors']),'ddp_world_size':2,'ddp_epochs':len(history),'replicas_identical':True,'validation_unique_count':1,'cuda':'not available; untested','raw_data_published':False}
(OUT/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
