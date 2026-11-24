"""Controlled BTCV ablations. No training is performed at module import."""
from pathlib import Path
import collections, functools, json, math, random, time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from common import *
from dataflow import *
from reference.sacf_model import SemanticContradictionField,FieldGuidedRefiner
from reference.sacf_losses import SACFLoss
from reference.corruptions import CounterfactualCorruptor,CORRUPTION_TYPES
from reference.egrc import EGRCCorrector,logit,soft_dice_loss
from variants import StaticGateEGRC,DirectFusionRefiner,align,retrieval,region_weights

def stage_dir(cfg,name):return Path(cfg.get('stage_root',str(Path(cfg['output'])/'runs')))/name
def best_path(cfg,name):
    if name=='sacf_name' and cfg.get('preflight_stage')=='scar_name':
        root=Path(cfg['output'])/'runs/sacf_name';state=load(root/'latest.pt')['state']
        return root/f"best_slot{state['best_slot']}.pt"
    external=cfg.get('external_references',{}).get(name)
    if external:
        path=Path(external['path'])
        if sha(path)!=external['sha256']:raise ValueError('External reference hash changed: '+name)
        return path
    root=stage_dir(cfg,name);state=load(root/'latest.pt')['state']
    return root/f"best_slot{state['best_slot']}.pt"

def model_state(cfg,name,path):
    payload=load(path)
    key=cfg.get('external_references',{}).get(name,{}).get('state_key','model')
    return payload[key]
def selected_model(cfg,name,device):
    model=build_model(cfg,name,device);model.load_state_dict(model_state(cfg,name,best_path(cfg,name)),strict=True)
    model.eval().requires_grad_(False);return model

def build_model(cfg,name,device):
    if name.startswith('trc_'):return make_imis(cfg,cfg['base_checkpoint'],device,training=True)
    if name.startswith('sacf_'):return SemanticContradictionField(num_organs=13,dim=192,image_size=512).to(device)
    if name=='scar_fusion':m=DirectFusionRefiner(dim=192)
    elif name.startswith('scar_'):m=FieldGuidedRefiner(dim=192)
    elif name=='egrc_static_gate':m=StaticGateEGRC(n_organs=13,K=4)
    else:m=EGRCCorrector(n_organs=13,K=4)
    if name.startswith('scar_'):
        nn.init.zeros_(m.final[-1].weight);nn.init.zeros_(m.final[-1].bias)
    return m.to(device)

def model_statistics(model):
    return dict(total_parameters=sum(p.numel() for p in model.parameters()),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))

def trc_dataset(cfg,name):
    bootstrap(cfg)
    from types import SimpleNamespace
    from data_loader import UniversalDataset
    from dataloaders.data_utils import Resize,PermuteTransform,Normalization,get_bboxes_from_mask,get_points_from_mask
    from monai import transforms
    facts=prepare_facts(cfg);data=prepare_aug(cfg) if name in ['trc_aaeca','trc_full'] else split(cfg)
    args=SimpleNamespace(data_dir=cfg['data_root'],image_size=512,mask_num=cfg['trc']['regions'],test_mode=False)
    weights=region_weights(facts['frequency']);weighted=name in ['trc_isrfos','trc_full']
    class Regions(UniversalDataset):
        def __getitem__(self,index):
            depth=getattr(self,'_depth',0)
            if depth>1000:raise ValueError('No valid interactive training sample after bounded resampling')
            if depth:self.resample_events=getattr(self,'resample_events',0)+1
            self._depth=depth+1
            try:
                result=super().__getitem__(index)
                result.setdefault('actual_training_index',index)
                return result
            finally:self._depth=depth
        def preprocess_label(self,gt,label_ids,selected):
            labels=list(label_ids)
            if len(labels)>self.mask_num and weighted:
                pool=list(labels);chosen=[]
                while len(chosen)<self.mask_num:
                    w=np.array([weights[int(c)] for c in pool]);w=w/w.sum()
                    index=int(np.random.choice(len(pool),p=w));chosen.append(pool.pop(index))
            elif len(labels)>=self.mask_num:chosen=random.sample(labels,self.mask_num)
            else:chosen=random.choices(labels,k=self.mask_num)
            points=[];point_labels=[];boxes=[];names=[]
            for i,c in enumerate(chosen):
                selected[i][0]=gt[c];p=get_points_from_mask(selected[i],top_num=.5)
                points.append(torch.as_tensor(p[0]));point_labels.append(torch.as_tensor(p[1]))
                boxes.append(torch.as_tensor(get_bboxes_from_mask(selected[i],offset=5)));names.append(self.target_list[c])
            return selected,torch.stack(points),torch.stack(point_labels),torch.stack(boxes),names
    transform=transforms.Compose([Resize(keys=['image','label','pseudo'],target_size=(512,512)),
        PermuteTransform(keys=['image'],dims=(2,0,1)),transforms.ToTensord(keys=['image','label','pseudo']),
        Normalization(keys=['image']),transforms.RandScaleIntensityd(keys='image',factors=.2,prob=.2),
        transforms.RandShiftIntensityd(keys='image',offsets=.2,prob=.2)])
    transform.set_random_state(seed=cfg['seed'])
    return Regions(args=args,datalist=data['training'],classes_list=list(data['labels'].values()),transform=transform)

def transform_rng(transform,states=None):
    objects=[];visited=set()
    def walk(t):
        if id(t) in visited:return
        visited.add(id(t))
        if hasattr(t,'R') and hasattr(t.R,'get_state'):objects.append(t)
        # Dictionary transforms own nested randomizers such as .scaler and
        # .shifter, not only Compose.transforms. Their states matter on resume.
        for key,value in vars(t).items():
            if key=='R':continue
            children=value.values() if isinstance(value,dict) else value if isinstance(value,(tuple,list)) else [value]
            for child in children:
                if hasattr(child,'__dict__') and (hasattr(child,'R') or type(child).__module__.startswith('monai.transforms')):walk(child)
    walk(transform)
    if states is not None:
        assert len(states)==len(objects)
        for obj,state in zip(objects,states):obj.R.set_state(state)
    return [obj.R.get_state() for obj in objects]

def trc_step(model,batch,opt,device,cfg):
    from utils import FocalDice_MSELoss
    criterion=FocalDice_MSELoss();images=batch['image'].to(device);gt=batch['label'].long().to(device);ps=batch['pseudo'].to(device)
    if not gt.any() or not ps.any():raise ValueError('Empty TRC training batch; refusing silent skipping')
    text=model.process_text_prompt(batch['target_list']);emb=model.image_forward(images)
    a=random.choices(['bboxes','points','text'],[.4,.3,.3])[0];b=random.choice(['bboxes','points'])
    def prompt(which,kind):
        s=batch[which]
        if kind=='bboxes':return dict(bboxes=s['bboxes'].to(device))
        if kind=='points':return dict(point_coords=s['point_coords'].to(device),point_labels=s['point_labels'].to(device))
        return dict(text)
    with amp(device):
        go=model.forward_decoder(emb,prompt('gt_prompt',a));po=model.forward_decoder(emb,prompt('pseudo_prompt',b))
        loss=criterion(go['masks'],gt.float(),go['iou_pred'])+criterion(po['masks'],ps.float(),po['iou_pred'])
    finite_step(loss,model,opt,clip=None);losses=[float(loss.detach())]
    emb=emb.detach().clone();text={k:v.detach().clone() for k,v in text.items()}
    gl,pl,gp,pp=go['low_res_masks'],po['low_res_masks'],go['masks'],po['masks'];text_round=np.random.randint(0,cfg['trc']['interactions']-1)
    for i in range(cfg['trc']['interactions']):
        if i in [text_round,cfg['trc']['interactions']-1]:
            prm=model.process_mask_prompt(gl);prm.update(text);pprm=model.process_mask_prompt(pl)
        else:
            prm=model.supervised_prompts(None,gt,gp,gl,'points')
            if random.random()>.6:prm.update(text);prm.pop('mask_inputs',None)
            pprm=model.unsupervised_prompts(ps,pp,pl,'points')
        with amp(device):
            go=model.forward_decoder(emb,prm);po=model.forward_decoder(emb,pprm)
            loss=criterion(go['masks'],gt.float(),go['iou_pred'])+criterion(po['masks'],ps.float(),po['iou_pred'])
        if not torch.isfinite(loss):raise FloatingPointError('Non-finite interactive loss')
        loss.backward(retain_graph=True);losses.append(float(loss.detach()))
        gl,pl,gp,pp=go['low_res_masks'],po['low_res_masks'],go['masks'],po['masks']
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    if not grads or not all(torch.isfinite(g).all() for g in grads):raise FloatingPointError('Non-finite interactive gradients')
    opt.step();opt.zero_grad(set_to_none=True)
    return sum(losses)/len(losses)

def sacf_batch(cache,ids,device,cfg,text_style='structured'):
    b=selected_batch(cache,ids,cfg['scar']['organs_per_slice'],device);N=len(b['organs'])
    # Original mixed GT/predicted/perturbed-predicted context; target channel zero.
    ctx=b['context'].clone()
    # Recover exact per-query source GT contexts in the same sampled order by
    # matching selected masks/organ ID to the cached image embedding sequence.
    # selected_batch stores the source slice for this purpose.
    for i,(si,c) in enumerate(zip(b['source_slices'],b['organs'])):
        item=cache.get(si);ground=torch.zeros_like(ctx[i])
        for j,o in enumerate(item['organs']):ground[o]=item['gt_tensor'][j,0].to(device)
        ground[c]=0;r=random.random()
        if r<.4:ctx[i]=ground
        elif r>=.8:
            candidates=[o for o in item['organs'] if o!=c]
            if candidates:
                o=random.choice(candidates);m=ctx[i,o:o+1][None];kind=random.randrange(3)
                ctx[i,o]=0 if kind==2 else (F.max_pool2d(m,3,1,1) if kind==0 else 1-F.max_pool2d(1-m,3,1,1))[0,0]
    corruptor=CounterfactualCorruptor(types=list(CORRUPTION_TYPES),spatial_scale=2.)
    neg,err,_=corruptor.corrupt(b['gt'].repeat(2,1,1,1),ctx.repeat(2,1,1,1),types=list(CORRUPTION_TYPES))
    text=cache.text[text_style][b['organs']].to(device)
    inputs=[b['emb'].repeat(3,1,1,1),torch.cat([b['gt'],neg]),ctx.repeat(3,1,1,1),text.repeat(3,1)]
    targets=dict(error=torch.cat([torch.zeros_like(b['gt']),err]),
                 is_neg=torch.cat([torch.zeros(N,device=device),(err.flatten(1).sum(1)>0).float()]),
                 group=torch.arange(N,device=device).repeat(3))
    return inputs,targets

@torch.no_grad()
def validate_sacf(cfg,model,name,device):
    cache=Cache(cfg,'validation');scores=[];labels=[];style='name' if name=='sacf_name' else 'structured'
    with preserved_rng(1234):
        for si in range(len(cache.items)):
            inputs,target=sacf_batch(cache,[si],device,cfg,style)
            with amp(device):out=model(*inputs)
            scores.extend(out['score'].float().cpu().tolist());labels.extend(target['is_neg'].cpu().tolist())
    return dict(score=auc(scores,labels),metric='counterfactual query detection AUROC',n=len(scores))

def cache_fields(cfg,parent,device):
    folder=Path(cfg['output'])/'cache'/parent;folder.mkdir(parents=True,exist_ok=True)
    upstream=Cache(cfg,'training');identity=dict(base=upstream.identity,sacf=sha(best_path(cfg,parent)),style=parent,
                                                implementation=sha(__file__))
    if (folder/'complete.json').exists():assert read(folder/'complete.json')['identity']==identity;return
    model=selected_model(cfg,parent,device);style='name' if parent=='sacf_name' else 'structured'
    store=TensorStore(folder,identity)
    for phase in ['training','validation']:
        cache=Cache(cfg,phase)
        for si,it in enumerate(cache.items):
            key=f'{phase}/{si:05d}';old=store.get(key)
            if old is not None:assert old['identity']==identity;continue
            x=cache.get(si);ctx=torch.zeros(1,13,512,512,device=device)
            for j,c in enumerate(x['organs']):ctx[0,c]=x['prob'][j,0].to(device)
            fields=[]
            with torch.no_grad(),amp(device):
                for j,c in enumerate(x['organs']):
                    local=ctx.clone();local[0,c]=0
                    fields.append(model(x['embedding'][None].float().to(device),x['prob'][j:j+1].float().to(device),local,cache.text[style][c:c+1].to(device))['fused'].float().cpu())
            store.put(key,dict(identity=identity,image=it['image'],field=torch.cat(fields) if fields else torch.empty(0,1,512,512)))
            if si%50==0:print('[field cache]',parent,phase,si,flush=True)
    store.close();write(folder/'complete.json',dict(identity=identity));del model

@functools.lru_cache(maxsize=12)
def _intermediate_store(path,identity):return TensorStore(path,json.loads(identity))

def intermediate(cfg,kind,phase,si):
    root=Path(cfg['output'])/'cache'/kind;ident=read(root/'complete.json')['identity']
    store=_intermediate_store(str(root),json.dumps(ident,sort_keys=True))
    x=store.get(f'{phase}/{si:05d}')
    if x is None:raise ValueError('Incomplete intermediate cache: '+kind)
    assert x['identity']==ident
    return x

def refiner_batch(cfg,name,cache,ids,device):
    assert name=='scar_name'
    from name_runtime import field_model
    b=selected_batch(cache,ids,cfg['scar']['organs_per_slice'],device)
    sacf=field_model(cfg,device);fields=[]
    for si,c in zip(b['source_slices'],b['organs']):
        x=cache.get(si);j=x['organs'].index(c)
        ctx=torch.zeros(1,13,512,512,device=device)
        for jj,o in enumerate(x['organs']):ctx[0,o]=x['prob'][jj,0].to(device)
        ctx[0,c]=0
        with torch.no_grad(),amp(device):
            field=sacf(x['embedding'][None].float().to(device),x['prob'][j:j+1].float().to(device),ctx,cache.text['name'][c:c+1].to(device))['fused'].float().cpu()
        fields.append(field[0])
    return b,[b['emb'],b['p'],torch.stack(fields).to(device),cache.text['name'][b['organs']].to(device)]


def refiner_loss(logits,gt):
    from refiner_original_loss import batch_loss
    return batch_loss(logits.float(),gt)

@torch.no_grad()
def refined_item(cfg,cache,si,model,name,device):
    x=cache.get(si);style='name' if name=='scar_name' else 'structured'
    if not x['organs']:return torch.empty(0,1,512,512,device=device)
    parent='sacf_name' if name=='scar_name' else 'sacf_full'
    fields=None if name in ['scar_no_field','scar_fusion'] else intermediate(cfg,parent,cache.phase,si)['field']
    output=[]
    with amp(device):
        for j,c in enumerate(x['organs']):
            field=torch.zeros(1,1,512,512,device=device) if fields is None else fields[j:j+1].float().to(device)
            logits=model(x['embedding'][None].float().to(device),x['prob'][j:j+1].float().to(device),field,cache.text[style][c:c+1].to(device))
            output.append(logits.float().sigmoid())
    return torch.cat(output)

def cache_reference(cfg,device):
    if cfg.get('egrc_upstream')!='base':raise ValueError('EGRC internal ablations require the original Base predictions')
    root=Path(cfg['output'])/'cache/egrc_reference';root.mkdir(parents=True,exist_ok=True)
    ident=dict(base=Cache(cfg,'training').identity,sacf=sha(best_path(cfg,'sacf_full')),reference_kind='original_base',implementation=sha(__file__))
    if (root/'complete.json').exists():assert read(root/'complete.json')['identity']==ident;return
    sacf=selected_model(cfg,'sacf_full',device)
    store=TensorStore(root,ident)
    for phase in ['training','validation']:
        cache=Cache(cfg,phase)
        for si,it in enumerate(cache.items):
            key=f'{phase}/{si:05d}';old=store.get(key)
            if old is not None:assert old['identity']==ident;continue
            x=cache.get(si);reference=x['prob'].float().to(device)
            ctx=torch.zeros(1,13,512,512,device=device)
            for j,c in enumerate(x['organs']):ctx[0,c]=reference[j,0]
            fields=[]
            with torch.no_grad(),amp(device):
                for j,c in enumerate(x['organs']):
                    local=ctx.clone();local[0,c]=0
                    fields.append(sacf(x['embedding'][None].float().to(device),reference[j:j+1],local,cache.text['structured'][c:c+1].to(device))['fused'].float())
            store.put(key,dict(identity=ident,image=it['image'],reference=reference.cpu(),field=torch.cat(fields).cpu() if fields else torch.empty(0,1,512,512)))
            if si%50==0:print('[EGRC upstream]',phase,si,flush=True)
    store.close();write(root/'complete.json',dict(identity=ident));del sacf

class ExemplarBank:
    def __init__(self,cfg):
        self.cfg=cfg;self.cache=Cache(cfg,'training');facts=prepare_facts(cfg);emb=[];self.pool=collections.defaultdict(list)
        ident=dict(base=self.cache.identity,facts=digest(facts),original_training_only=True,patient_exclusion=True,K=4)
        stored=Path(cfg['output'])/'cache/exemplar_index.pt'
        if stored.exists():
            x=load(stored);assert x['identity']==ident
            for k in ['emb','layouts','pool','rare','supported','training','masks']:setattr(self,k,x[k])
            self.identity=ident;return
        self.layouts=np.array([x['areas'] for x in facts['info']],np.float32)/512**2
        self.masks={}
        for si in range(len(self.cache.items)):
            x=self.cache.get(si);e=x['embedding'].float().mean((1,2));emb.append(e/e.norm().clamp_min(1e-8))
            for j,c in enumerate(x['organs']):
                self.pool[c].append(si)
                m=F.interpolate(x['gt_tensor'][j:j+1],size=(256,256),mode='nearest')[0,0].numpy().astype(bool)
                self.masks[si,c]=np.packbits(m,axis=-1)
        self.emb=torch.stack(emb);self.rare=facts['rare_ids']
        self.supported={c for c in self.rare if len({patient(self.cache.items[i]) for i in self.pool[c]})>1}
        self.training=[(i,c) for i in range(len(self.cache.items)) for c in self.cache.get(i)['organs'] if c in self.supported]
        self.identity=ident
        save_cache(stored,dict(identity=ident,**{k:getattr(self,k) for k in ['emb','layouts','pool','rare','supported','training','masks']}))
    def exemplars(self,query,organ,pred,name,si,phase):
        candidates=self.pool[organ]
        if phase=='training':candidates=[i for i in candidates if patient(self.cache.items[i])!=query['patient']]
        e=query['embedding'].float().mean((1,2));e=e/e.norm().clamp_min(1e-8)
        layout=np.zeros(13,np.float32)
        for j,c in enumerate(query['organs']):layout[c]=float(query['prob'][j].float().mean())
        mode='random' if name=='egrc_random' else 'appearance' if name=='egrc_appearance' else 'joint'
        chosen=retrieval(e,layout,candidates,self.emb,self.layouts,4,mode,random.Random(self.cfg['seed']+si*101+organ))
        masks=[];pred256=F.interpolate(pred[None],size=(256,256),mode='bilinear',align_corners=False)[0,0].cpu()
        for i in chosen:
            m=torch.from_numpy(np.unpackbits(self.masks[i,organ],axis=-1,count=256).copy()).float()
            masks.append(m if name=='egrc_no_align' else align(m,pred256))
        masks=(masks+[torch.zeros(256,256)]*4)[:4]
        return torch.stack(masks)[:,None]

def egrc_batch(cfg,name,bank,ids,device):
    em=[];p=[];f=[];gt=[];ex=[];org=[]
    for idx in ids:
        si,c=bank.training[idx];x=bank.cache.get(si);j=x['organs'].index(c)
        r=intermediate(cfg,'egrc_reference','training',si)
        assert r['identity']['reference_kind']=='original_base'
        assert r['identity']['base']==bank.cache.identity
        assert r['identity']['sacf']==sha(best_path(cfg,'sacf_full'))
        em.append(x['embedding'].float());p.append(r['reference'][j].float());gt.append(x['gt_tensor'][j])
        f.append(torch.zeros_like(p[-1]) if name=='egrc_no_residual_field' else r['field'][j].float())
        ex.append(bank.exemplars(x,c,p[-1],name,si,'training'));org.append(c)
    return [torch.stack(em).to(device),torch.stack(p).to(device),torch.stack(f).to(device),
            torch.stack(ex).to(device),torch.tensor(org,device=device)],torch.stack(gt).to(device)

from frozen_evaluation import evaluate


def train(cfg,name,device,batch,capacity_only=False):
    if not cfg.get('execution',{}).get('training_enabled',False):raise RuntimeError('Prepare-only package: training and GPU preflight disabled by user instruction')
    if name not in cfg['allowed_training']:raise ValueError('Outside the four selected ablations: '+name)
    seed(cfg['seed']);model=build_model(cfg,name,device);root=stage_dir(cfg,name);root.mkdir(parents=True,exist_ok=True)
    family='trc' if name.startswith('trc_') else 'sacf' if name.startswith('sacf_') else 'scar' if name.startswith('scar_') else 'egrc'
    recipe=cfg['trc'] if family=='trc' else cfg['scar'] if family in ['sacf','scar'] else cfg['egrc']
    parent_hashes={};bank=None
    if family=='trc':
        ds=trc_dataset(cfg,name);stream=Stream(len(ds),cfg['seed']);per_epoch=math.ceil(1766/batch);epochs=recipe['virtual_epochs']
        from data_loader import train_collate_fn
        make_batch=lambda ids:train_collate_fn([ds[i] for i in ids])
    elif family in ['sacf','scar']:
        cache=Cache(cfg,'training');eligible=[i for i,x in enumerate(prepare_facts(cfg)['info']) if sum(x['areas'])>0]
        stream=Stream(len(eligible),cfg['seed']);per_epoch=math.ceil(1766/batch)
        epochs=recipe['sacf_epochs'] if family=='sacf' else recipe['refiner_epochs']
        parent_hashes['base_cache']=digest(cache.identity)
        if family=='scar' and name not in ['scar_no_field','scar_fusion']:
            parent_hashes['sacf']=sha(best_path(cfg,'sacf_name' if name=='scar_name' else 'sacf_full'))
    else:
        bank=ExemplarBank(cfg);stream=Stream(len(bank.training),cfg['seed']);per_epoch=math.ceil(len(bank.training)/batch);epochs=recipe['epochs']
        parent_hashes=dict(bank=digest(bank.identity),upstream=read(Path(cfg['output'])/'cache/egrc_reference/complete.json')['identity'])
    lr=recipe['lr'];wd=recipe['weight_decay'];total=epochs*per_epoch
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=wd)
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:1. if family=='trc' else .5*(1+math.cos(math.pi*min(s,total)/total)))
    identity=dict(recipe=cfg['recipe_identity'],stage=name,seed=cfg['seed'],batch=batch,budget=total,parents=parent_hashes)
    state=dict(step=0,optimizer_updates=0,best=-1.,best_slot=1,history=[])
    if not capacity_only:state=resume(root,model,opt,scheduler,stream,identity)
    if family=='trc' and state.get('dataset_rng') is not None:transform_rng(ds.transform,state['dataset_rng'])
    if family=='trc':ds.resample_events=state.get('resample_events',0)
    target=2 if capacity_only else total
    start_step=state['step'];started=time.monotonic();prior_seconds=state.get('active_seconds',0.)
    model.train()
    for step in range(state['step'],target):
        ids=stream.next(batch)
        if family in ['sacf','scar']:ids=[eligible[i] for i in ids]
        if family=='trc':loss=trc_step(model,make_batch(ids),opt,device,cfg);updates=2
        else:
            if family=='sacf':
                inputs,targets=sacf_batch(cache,ids,device,cfg,'name' if name=='sacf_name' else 'structured')
                with amp(device):loss,_=SACFLoss().to(device)(model(*inputs),targets)
            elif family=='scar':
                b,inputs=refiner_batch(cfg,name,cache,ids,device)
                with amp(device):loss=refiner_loss(model(*inputs),b['gt'])
            else:
                inputs,gt=egrc_batch(cfg,name,bank,ids,device)
                with amp(device):res=model(*inputs).float();pr=torch.sigmoid(logit(inputs[1])+res)
                loss=soft_dice_loss(pr,gt)+F.binary_cross_entropy(pr.clamp(1e-6,1-1e-6),gt)+recipe['residual_l2']*res.square().mean()
            finite_step(loss,model,opt);loss=float(loss.detach());updates=1
        scheduler.step();state['step']=step+1;state['optimizer_updates']+=updates
        duration=time.monotonic()-started;state['active_seconds']=prior_seconds+duration
        if (step+1)%25==0 or capacity_only:print('[train]',name,'step',step+1,'/',target,'loss',loss,'resamples',getattr(locals().get('ds'),'resample_events',0),flush=True)
        if capacity_only:continue
        write(root/'progress.json',dict(stage=name,step=step+1,total_steps=total,virtual_epoch=(step+1)/per_epoch,optimizer_updates=state['optimizer_updates'],loss=loss,
              updated=time.time(),active_seconds=state['active_seconds'],seconds_per_batch=duration/(step+1-start_step),estimated_remaining_seconds=(total-step-1)*duration/(step+1-start_step)))
        at_val=(step+1)%per_epoch==0 or step+1==total
        best=False
        if at_val:
            model.eval()
            with preserved_rng(1234):val=validate_sacf(cfg,model,name,device) if family=='sacf' else evaluate(cfg,model,name,device,bank=bank)
            state['history'].append(dict(step=step+1,virtual_epoch=(step+1)//per_epoch,**val))
            if val['score']>state['best']:
                state['best']=val['score'];state['alpha']=val.get('alpha',1.);state['best_step']=step+1;best=True
            model.train();write(root/'history.json',state['history'])
        if at_val or (step+1)%cfg['checkpoint_every']==0:
            if family=='trc':state['dataset_rng']=transform_rng(ds.transform)
            if family=='trc':state['resample_events']=ds.resample_events
            state=checkpoint(root,model,opt,scheduler,stream,state,identity,best=best)
    if capacity_only:
        # Full optimizer/scheduler/sampler/RNG roundtrip, without retaining preflight updates.
        state['step']=2;state=checkpoint(root,model,opt,scheduler,stream,state,identity)
        restored=build_model(cfg,name,'cpu');other=torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad],lr=lr,weight_decay=wd)
        sched2=torch.optim.lr_scheduler.LambdaLR(other,lambda s:1.)
        resumed=resume(root,restored,other,sched2,stream,identity)
        assert resumed['step']==2 and all(torch.equal(v.cpu(),restored.state_dict()[k]) for k,v in model.state_dict().items())
        assert state_equal(opt.state_dict(),other.state_dict())
        assert state_equal(scheduler.state_dict(),sched2.state_dict())
        return model_statistics(model)
    write(root/'complete.json',dict(identity=identity,state={k:v for k,v in state.items() if k not in ['history','dataset_rng']},parameters=model_statistics(model)))
    if family!='sacf':
        model.load_state_dict(load(best_path(cfg,name))['model'],strict=True);model.eval()
        evaluate(cfg,model,name,device,final=True,bank=bank)
