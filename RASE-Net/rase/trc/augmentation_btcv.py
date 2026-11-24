"""Frozen inputs, original-only frequency statistics and lineage-checked caches."""
from pathlib import Path
from functools import lru_cache
import ast, collections, json, math, os, random, sys
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from scipy import sparse,ndimage
from ._common import HERE,sha,digest,read,write,save,save_cache,load,patient,amp,TensorStore

@lru_cache(maxsize=16)
def label(path):
    p=Path(path);shape=ast.literal_eval(p.name.split('.')[-2])
    a=sparse.load_npz(p).toarray().reshape(shape)
    if a.ndim==4:a=a[...,0]
    if a.shape!=(13,512,512):raise ValueError('Unexpected BTCV label shape '+str(a.shape))
    if not np.all((a==0)|(a==1)):raise ValueError('Non-binary label')
    if np.any(a.sum(0)>1):raise ValueError('Overlapping GT organ channels')
    return a.astype(bool)

def full_path(cfg,item,key):return str((Path(cfg['data_root'])/item[key]).resolve())
def load_gt(cfg,item):return torch.from_numpy(label(full_path(cfg,item,'label')).copy()).float()
def load_image(cfg,item):
    im=Image.open(full_path(cfg,item,'image')).convert('RGB')
    if im.size!=(512,512):im=im.resize((512,512),Image.Resampling.NEAREST)
    a=torch.from_numpy(np.array(im).copy()).permute(2,0,1).float()
    return (a-torch.tensor([123.675,116.28,103.53])[:,None,None])/torch.tensor([58.395,57.12,57.375])[:,None,None]

def split(cfg):
    p=Path(cfg['split']);assert sha(p)==cfg['split_sha256'],'Original split SHA256 changed'
    d=read(p);assert len(d['training'])==1766 and len(d['validation'])==407
    tr={patient(v) for v in d['training']};va={patient(v) for v in d['validation']}
    assert len(tr)==24 and len(va)==6 and not(tr&va)
    return d

def prepare_facts(cfg):
    dest=Path(cfg['output'])/'data';dest.mkdir(parents=True,exist_ok=True);p=dest/'facts.json'
    identity=dict(split=cfg['split_sha256'],definition='Original training slices only; nonzero GT area')
    if p.exists():
        x=read(p);assert x['identity']==identity;return x
    d=split(cfg);classes=[d['labels'][str(i)] for i in range(1,14)];info=[];counts=np.zeros(13,int)
    for si,it in enumerate(d['training']):
        a=label(full_path(cfg,it,'label'));areas=a.sum((1,2));centroids={}
        counts+=(areas>0)
        for c in np.flatnonzero(areas):
            y,x=np.where(a[c]);centroids[str(c)]=[float(y.mean()/512),float(x.mean()/512)]
        info.append(dict(areas=areas.tolist(),centroids=centroids,patient=patient(it)))
        if si%100==0:print('[facts]',si,'/',len(d['training']),flush=True)
    frequency=counts/len(info);ids=np.argsort(frequency,kind='stable')[:math.ceil(.4*13)].tolist()
    # Freeze the actual original-training frequency definition, not an AMOS constant.
    x=dict(identity=identity,classes=classes,info=info,frequency=frequency.tolist(),rare_ids=ids,
           rare_names=[classes[c] for c in ids],counts=counts.tolist())
    expected=set(cfg['rare_names'])
    if set(x['rare_names'])!=expected:
        raise ValueError('Original training frequency does not reproduce the six paper rare organs: '+repr(x['rare_names']))
    write(p,x);return x

def prepare_aug(cfg):
    facts=prepare_facts(cfg);d=split(cfg);tr=d['training'];out=Path(cfg['output'])/'data/aaeca'
    ident=dict(facts=digest(facts),seed=cfg['seed'],count=cfg['augmentation_count'],implementation=sha(__file__))
    if (out/'complete.json').exists():
        assert read(out/'complete.json')['identity']==ident
        return read(out/'split.json')
    out.mkdir(parents=True,exist_ok=True)
    for n in ['image','label','imask']:(out/n).mkdir(exist_ok=True)
    rng=random.Random(cfg['seed']);areas=np.asarray([x['areas'] for x in facts['info']]);info=facts['info']
    cases=[patient(x) for x in tr];zs=[int(Path(x['image']).stem.rsplit('_',1)[1].lstrip('z')) for x in tr]
    bounds={c:(min(z for p,z in zip(cases,zs) if p==c),max(z for p,z in zip(cases,zs) if p==c)) for c in set(cases)}
    z=[(v-bounds[c][0])/max(bounds[c][1]-bounds[c][0],1) for c,v in zip(cases,zs)]
    stable=[i for i,n in enumerate(facts['classes']) if n in ['aorta','inferior_vena_cava','liver','kidney_left','kidney_right','spleen']]
    anchors={};donors={}
    for c in facts['rare_ids']:
        scores=[]
        for anchor in stable:
            if c==anchor:continue
            bins=collections.defaultdict(list)
            for i,x in enumerate(info):
                if str(c) in x['centroids'] and str(anchor) in x['centroids']:
                    delta=(np.array(x['centroids'][str(c)])-x['centroids'][str(anchor)])/np.sqrt(areas[i,anchor]/512**2)
                    bins[min(int(z[i]*10),9)].append(delta)
            valid=[v for v in bins.values() if len(v)>=3]
            if sum(map(len,valid))>=10:
                scores.append((sum(np.std(v,axis=0).sum()*len(v) for v in valid)/sum(map(len,valid)),anchor))
        if not scores:raise RuntimeError('No valid anatomy anchor: '+facts['classes'][c])
        anchors[c]=min(scores)[1];lo,hi=np.percentile(areas[areas[:,c]>0,c],[25,75])
        donors[c]=[i for i in range(len(tr)) if lo<=areas[i,c]<=hi and areas[i,anchors[c]]>0]
    @lru_cache(maxsize=16)
    def image(i):return np.asarray(Image.open(full_path(cfg,tr[i],'image')).convert('RGB'))
    progress=out/'progress.pt';aug=[];trace=[];reuse=collections.Counter();recused=set();attempt=0
    if progress.exists():
        old=load(progress);assert old['identity']==ident
        aug=old['aug'];trace=old['trace'];reuse=old['reuse'];recused=old['recused'];attempt=old['attempt'];rng.setstate(old['rng'])
    for attempt in range(attempt,cfg['augmentation_count']*300):
        if len(aug)==cfg['augmentation_count']:break
        c=rng.choice(facts['rare_ids']);a=anchors[c];ri=rng.randrange(len(tr))
        if ri in recused or areas[ri,c]>0 or areas[ri,a]==0:continue
        candidates=[j for j in donors[c] if cases[j]!=cases[ri] and abs(z[j]-z[ri])<=.1 and reuse[j,c]<3]
        if not candidates:continue
        di=rng.choice(candidates);dm=label(full_path(cfg,tr[di],'label'))[c];rl=label(full_path(cfg,tr[ri],'label')).copy()
        ys,xs=np.where(dm)
        if len(ys)<5:continue
        scale=np.clip(np.sqrt(areas[ri,a]/areas[di,a]),.7,1.4)
        target=np.asarray(info[ri]['centroids'][str(a)])+scale*(np.asarray(info[di]['centroids'][str(c)])-info[di]['centroids'][str(a)])
        shift=np.rint(target*512-[ys.mean(),xs.mean()]).astype(int);ty,tx=ys+shift[0],xs+shift[1]
        ok=(ty>=0)&(ty<512)&(tx>=0)&(tx<512)
        if ok.mean()<.98:continue
        sy,sx=ys[ok],xs[ok];ty,tx=ty[ok],tx[ok]
        if rl[:,ty,tx].any():continue
        recipient=image(ri);donor=image(di)
        if (recipient.mean(2)>10)[ty,tx].mean()<.98:continue
        tm=np.zeros((512,512),bool);tm[ty,tx]=True
        rr=ndimage.binary_dilation(tm,iterations=3)&~tm;rd=ndimage.binary_dilation(dm,iterations=3)&~dm
        if not rr.any() or not rd.any():continue
        delta=float(recipient[rr].mean()-donor[rd].mean())
        if abs(delta)>40:continue
        result=recipient.copy();alpha=ndimage.gaussian_filter(tm.astype(float),.8)[ty,tx,None]
        patch=np.clip(donor[sy,sx].astype(float)+np.clip(delta,-20,20),0,255)
        result[ty,tx]=np.rint(alpha*patch+(1-alpha)*recipient[ty,tx]).astype(np.uint8)
        assert np.array_equal(result[~tm],recipient[~tm]);rl[c,ty,tx]=True
        im=np.load(full_path(cfg,tr[ri],'imask')).copy();im=im[...,0] if im.ndim==3 else im
        im[ty,tx]=float(im.max()+1);name=f'aaeca_{len(aug):05d}'
        ip=out/'image'/f'{name}.png';lp=out/'label'/f'{name}.(13, 512, 512, 1).npz';mp=out/'imask'/f'{name}.npy'
        Image.fromarray(result).save(ip);sparse.save_npz(lp,sparse.csr_matrix(rl.astype(np.int32).reshape(1,-1)));np.save(mp,im[...,None].astype(np.float32))
        aug.append(dict(image=str(ip),label=str(lp),imask=str(mp),patient_id=cases[ri]))
        trace.append(dict(organ=c,donor=di,recipient=ri,anchor=a,donor_patient=cases[di],recipient_patient=cases[ri],retained=float(ok.mean()),intensity=delta))
        reuse[di,c]+=1;recused.add(ri)
        save(progress,dict(identity=ident,aug=aug,trace=trace,reuse=reuse,recused=recused,attempt=attempt+1,rng=rng.getstate()))
        if len(aug)%50==0:print('[AAECA]',len(aug),'/',cfg['augmentation_count'],flush=True)
    if len(aug)!=cfg['augmentation_count']:raise RuntimeError('Insufficient valid AAECA samples; no silent count reduction')
    new=dict(d);new['training']=tr+aug
    write(out/'split.json',new);write(out/'trace.json',trace);write(out/'complete.json',dict(identity=ident,count=len(aug),anchors=anchors))
    return new
