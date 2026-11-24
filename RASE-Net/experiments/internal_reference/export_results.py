"""Only measured same-protocol rows; never copy historical paper values."""
from pathlib import Path
import csv,math
import numpy as np
from common import *
from dataflow import split,load_gt
from protocol_oracle import oracle
GROUPS={'SCAR':['base','scar_name','scar_fusion','scar_full'],
        'EGRC':['base','egrc_random','egrc_static_gate','egrc_full']}

def same(a,b):
    if a is None or not np.isfinite(a):return b is None or not np.isfinite(b)
    return b is not None and np.isfinite(b) and abs(a-b)<1e-10

def fmean(values):
    values=[float(v) for v in values if v is not None and np.isfinite(v)]
    return math.fsum(values)/len(values) if values else None

def export(cfg):
    root=Path(cfg['output']);dest=root/'export';dest.mkdir(parents=True,exist_ok=True)
    all_results={};protocol=digest(cfg['evaluation_protocol']);items=split(cfg)['validation']
    for group,names in GROUPS.items():
        missing=[n for n in names if not (root/'runs'/n/'evaluation/result.json').exists()]
        if missing:write(dest/f'{group}_pending.json',dict(status='incomplete',missing=missing));continue
        rows=[];results={}
        for name in names:
            x=read(root/'runs'/name/'evaluation/result.json');identity=x['identity'];summary=x['summary'];raw=x['records']
            assert identity['protocol']==protocol and identity['evaluation_hw']==[512,512]
            assert identity['metric_source']==sha(HERE/'legacy_metrics.py')
            assert len(raw)==len({(r['image'],r['organ']) for r in raw})==2204
            assert len({r['patient'] for r in raw})==6 and len({r['image'] for r in raw})==407
            groups={c:[r for r in raw if r['organ']==c] for c in {r['organ'] for r in raw}};assert len(groups)==13
            for k in METRICS:
                assert same(fmean([r[k] for r in raw]),summary['mean'][k])
                assert same(fmean([fmean([r[k] for r in rs]) for rs in groups.values()]),summary['macro'][k])
            assert same(summary['miou'],summary['macro']['iou'])
            store=TensorStore(root/'runs'/name/'evaluation',identity)
            for si in [0,203,406]:
                shard=store.get(si);assert shard['identity']==identity
                pred=np.unpackbits(shard['predictions'][x['alpha']],axis=-1,count=512)
                gt=load_gt(cfg,items[si]).numpy().astype(bool)
                for j,c in enumerate(shard['organs']):
                    expected=oracle(pred[j],gt[c]);actual=shard['records'][x['alpha']][j]
                    assert all(same(expected[k],actual[k]) for k in METRICS)
            store.close();results[name]=x
            rows.append(dict(method=name,**summary['mean'],macro_dice=summary['macro']['dice'],miou=summary['miou'],
              rare_dice=summary['rare']['dice'],other_dice=summary['other']['dice'],alpha=x['alpha']))
        with (dest/f'{group}_full_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        stats={n:paired_patient_statistics(results[names[-1]],results[n]) for n in names[:-1]}
        write(dest/f'{group}_patient_statistics.json',stats);all_results[group]=dict(rows=rows,statistics=stats)
    assert sha(cfg['base_checkpoint'])==cfg['base_checkpoint_sha256']
    write(dest/'all_results.json',dict(recipe=cfg,groups=all_results,cache_metric_audit_slices=[0,203,406],
      metric_protocol=cfg['evaluation_protocol'],budget_qualification='Historical full references have different training budgets; not an equal-budget component-only causal estimate.'))
    text=['# BTCV内部消融','原始Base；固定原指标代码与同一评价网格；完整参照未核验或任一新组缺失时不导出完整比较表。']
    fmt=lambda v:'NA' if v is None else f'{v:.4f}'
    for group,x in all_results.items():
        text+=['',f'## {group}','| 配置 | Mean Dice (%) | Macro Dice (%) | HD95 (px) | ASSD (px) | NSD |','|---|---:|---:|---:|---:|---:|']
        for r in x['rows']:text.append(f"| {r['method']} | {100*r['dice']:.4f} | {100*r['macro_dice']:.4f} | {fmt(r['hd95'])} | {fmt(r['assd'])} | {fmt(r['nsd'])} |")
    (dest/'results.md').write_text('\n'.join(text),encoding='utf-8')
