"""One archived 512 protocol. No training and no policy changes at import."""
import contextlib,math
from pathlib import Path
import numpy as np
import torch
from common import *
from dataflow import split,Cache,predict_imis

def evaluation_amp(device):
    return torch.autocast('cuda',dtype=torch.float16) if str(device).startswith('cuda') else contextlib.nullcontext()

@torch.no_grad()
def evaluate(cfg,model,name,device,final=False,bank=None):
    from engine import best_path,stage_dir,selected_model
    from reference.egrc import logit
    protocol=cfg['evaluation_protocol']
    assert protocol['evaluation_hw']==[512,512] and protocol['threshold']==.5
    d=split(cfg);cache=None if name=='base' else Cache(cfg,'validation')
    external=cfg.get('external_references',{}).get(name)
    is_eg=name.startswith('egrc_');alphas=cfg['egrc']['alpha_grid'] if is_eg else [1.]
    state={}
    if final:
        if name!='base':state={'alpha':external.get('alpha',1.)} if external else load(stage_dir(cfg,name)/'latest.pt')['state']
        alphas=[state.get('alpha',1.)]
    grouped={a:[] for a in alphas};folder=stage_dir(cfg,name)/'evaluation';store=None
    if final:
        ident=dict(recipe=cfg['recipe_identity'],name=name,checkpoint=sha(best_path(cfg,name)) if name!='base' else sha(cfg['base_checkpoint']),
          protocol=digest(protocol),metric_source=sha(HERE/'legacy_metrics.py'),evaluation_source=sha(__file__),
          precision=cfg['precision'],split=cfg['split_sha256'],evaluation_hw=[512,512],alphas=alphas)
        store=TensorStore(folder,ident)
    # The complete reference and ablated refiners use the SAME inference path.
    sacf=None
    if name not in ['base','scar_fusion']:
        sacf=selected_model(cfg,'sacf_name' if name=='scar_name' else 'sacf_full',device)
    for si,it in enumerate(d['validation']):
        old=store.get(si) if final else None
        if old is not None:
            assert old['identity']==ident and old['image']==it['image']
            for a,rs in old['records'].items():grouped[a].extend(rs)
            continue
        if cache is None:
            _,reference,gt,organs=predict_imis(cfg,model,it,device);prob={1.:reference}
        else:
            x=cache.get(si);organs=x['organs'];reference=x['prob'].float().to(device);gt=x['gt_tensor'].to(device)
            prob={a:reference.clone() for a in alphas}
            context=torch.zeros(1,13,512,512,device=device)
            for j,c in enumerate(organs):context[0,c]=reference[j,0]
            style='name' if name=='scar_name' else 'structured'
            texts=cache.text.get(style+'_eval',cache.text[style])
            for j,c in enumerate(organs):
                if is_eg and c not in bank.supported:continue
                t=texts[c:c+1].float().to(device);emb=x['embedding'][None].float().to(device)
                with evaluation_amp(device):
                    if name=='scar_fusion':field=torch.zeros_like(reference[j:j+1])
                    else:
                        local=context.clone();local[0,c]=0
                        field=sacf(emb,reference[j:j+1],local,t)['fused'].float()
                    if not is_eg:
                        logits=model(emb,reference[j:j+1],field,t)
                        prob[1.][j:j+1]=logits.float().sigmoid()
                    else:
                        ex=bank.exemplars(x,c,reference[j].cpu(),name,si,'validation')[None].to(device)
                        residual=model(emb,reference[j:j+1],field,ex,torch.tensor([c],device=device)).float()
                        for a in alphas:prob[a][j:j+1]=torch.sigmoid(logit(reference[j:j+1])+a*residual)
        records={};predictions={}
        for a,pr in prob.items():
            assert tuple(pr.shape)==tuple(gt.shape) and torch.isfinite(pr).all()
            p=(pr[:,0]>.5).cpu().numpy();g=(gt[:,0]>.5).cpu().numpy();records[a]=[]
            for j,c in enumerate(organs):
                values=metrics(p[j],g[j]) if final else {'dice':float(2*(p[j]&g[j]).sum()/(p[j].sum()+g[j].sum()+1e-6))}
                records[a].append(dict(image=it['image'],patient=patient(it),organ=c,organ_name=d['labels'][str(c+1)],**values))
            grouped[a].extend(records[a]);predictions[a]=np.packbits(p,axis=-1)
        if final:store.put(si,dict(identity=ident,image=it['image'],records=records,predictions=predictions,organs=organs))
        if si%100==0:print('[original validation]',name,si,'/407',flush=True)
    assert all(len(rs)==2204 for rs in grouped.values()),'Incomplete positive-query coverage'
    chosen=max(alphas,key=lambda a:(math.fsum(r['dice'] for r in grouped[a])/2204,-a))
    if not final:return dict(score=math.fsum(r['dice'] for r in grouped[chosen])/2204,alpha=chosen,metric='original-protocol validation Mean Dice')
    store.close();rows=grouped[alphas[0]];summary=summarize(rows,cfg['rare_names'])
    assert summary['patients']==6 and summary['slices']==407 and len(summary['per_organ'])==13
    assert len({(r['image'],r['organ']) for r in rows})==2204
    result=dict(identity=ident,phase='validation',independent_test=False,alpha=alphas[0],summary=summary,records=rows,
      reference_target=external.get('paper_mean_dice_percent') if external else None,
      reference_matches_paper_rounding=(round(100*summary['mean']['dice'],2)==external['paper_mean_dice_percent']) if external and 'paper_mean_dice_percent' in external else None)
    write(folder/'result.json',result);return result
