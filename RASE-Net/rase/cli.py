"""Portable entry points for the archived training and evaluation scripts."""
import argparse, json, os, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent/'_historical'
STAGES={'base':'train.py','trc':'train_trc.py','sacf':'sacf_dev/train_sacf.py',
        'refiner':'sacf_dev/train_refiner.py','bank':'build_bank.py',
        'cache':'precompute_cache.py','egrc':'train_egrc.py','evaluate':'eval_egrc_compare.py'}

def main():
    p=argparse.ArgumentParser(description='RASE-Net stages. Use list, or STAGE -- --help for backend options.')
    p.add_argument('stage',choices=['list',*STAGES]);p.add_argument('--config',type=Path)
    p.add_argument('--dry-run',action='store_true');p.add_argument('--run-dir',type=Path,default=Path.cwd())
    a,extra=p.parse_known_args()
    if a.stage=='list':
        for k,v in STAGES.items():print(f'{k:10} {v}')
        return
    if extra[:1]==['--']:extra=extra[1:]
    args=[]
    if a.config:
        config=json.loads(a.config.read_text(encoding='utf-8'))
        for k,v in config.get(a.stage,{}).items():
            flag='--'+k
            if isinstance(v,bool):
                if v:args.append(flag)
            elif v is not None:args.extend([flag,str(v)])
    command=[sys.executable,str(ROOT/STAGES[a.stage]),*args,*extra]
    env=dict(os.environ);env['PYTHONPATH']=os.pathsep.join([str(ROOT),str(ROOT/'sacf_dev'),env.get('PYTHONPATH','')])
    if a.dry_run:print(json.dumps({'command':command,'cwd':str(a.run_dir.resolve()),'implementation':'historical'},indent=2));return
    raise SystemExit(subprocess.call(command,cwd=a.run_dir.resolve(),env=env))
