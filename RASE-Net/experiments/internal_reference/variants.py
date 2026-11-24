"""Explicit interventions; the full EGRC is imported unchanged from the reference."""
import torch
from torch import nn
from torch.nn import functional as F
from reference.sacf_model import FieldGuidedRefiner, ConvBlock
from reference.egrc import EGRCCorrector

class DirectFusionRefiner(FieldGuidedRefiner):
    """Architecture comparator: concatenate mask, image and broadcast text; no SACF.

    The decoder and skip connections equal the reference. Removed field encoder
    and FiLM are replaced by a two-layer text feature projection. Parameter
    counts are reported; this is not claimed to be exactly FLOP-matched.
    """
    def __init__(self,dim=192):
        super().__init__(dim=dim)
        del self.field_enc;del self.text_mlp
        self.text_feature=nn.Sequential(nn.Linear(768,dim*2),nn.GELU(),nn.Linear(dim*2,dim))
    def forward(self,image_embed,mask,field,text_embed):
        f16,f4,f2=self.mask_encoder(mask);fi=self.img_proj(image_embed)
        ft=self.text_feature(text_embed)[:,:,None,None].expand_as(f16)
        base=self.fuse(torch.cat([f16,fi,ft],1))
        x=self.dec4(torch.cat([F.interpolate(base,size=f4.shape[-2:],mode='bilinear',align_corners=False),f4],1))
        x=self.dec2(torch.cat([F.interpolate(x,size=f2.shape[-2:],mode='bilinear',align_corners=False),f2],1))
        x=F.interpolate(x,size=mask.shape[-2:],mode='bilinear',align_corners=False)
        edge=(F.max_pool2d(mask,5,1,2)-(1-F.max_pool2d(1-mask,5,1,2))).clamp(0,1)
        delta=self.final(torch.cat([x,mask,edge],1));p=mask.clamp(1e-6,1-1e-6)
        return torch.log(p/(1-p))+delta

class StaticGateEGRC(EGRCCorrector):
    """Only the state-to-gate path changes; preserve all three gate applications."""
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        # Construct reference first, preserving other layers' seeded initialization.
        del self.sample_gate;nn.init.constant_(self.organ_gate.weight,-2.)
    def forward(self,img_emb,pred,field,ex_masks,organ_id):
        B=img_emb.shape[0];h,w=img_emb.shape[-2:]
        down=lambda x:F.interpolate(x,size=(h,w),mode='bilinear',align_corners=False)
        p,f=down(pred),down(field);q=self.query_proj(img_emb)+self.cond_proj(torch.cat([p,f],1))
        e=F.interpolate(self.ex_enc(ex_masks[:,:,0]),size=(h,w),mode='bilinear',align_corners=False)
        y,x=torch.meshgrid(torch.linspace(-1,1,h,device=q.device,dtype=q.dtype),torch.linspace(-1,1,w,device=q.device,dtype=q.dtype),indexing='ij')
        pos=self.coordinate_proj(torch.stack([x,y])[None])
        a,_=self.attn((q+pos).flatten(2).transpose(1,2),(e+pos).flatten(2).transpose(1,2),e.flatten(2).transpose(1,2),need_weights=False)
        a=a.transpose(1,2).reshape(B,self.d,h,w);vote=down(ex_masks.mean(1))
        valid=(ex_masks.flatten(1).sum(1)>0).to(q.dtype).view(B,1,1,1)
        g=torch.sigmoid(self.organ_gate(organ_id)).view(B,1,1,1)*valid
        r=self.decoder(torch.cat([q,q+g*a,p,f,g*vote],1))
        return g*F.interpolate(r,size=pred.shape[-2:],mode='bilinear',align_corners=False)

def align(ex,pred):
    """Reference 256-grid centroid translation: no scale, no rotation."""
    H=ex.shape[-1];eb=ex>.5
    if eb.sum()<4:return ex*0
    pb=pred>.5
    if pb.sum()<4:
        pb=pred>=pred.max()*.5
        if pb.sum()<1:return ex
    py,px=[v.float().mean() for v in torch.nonzero(pb,as_tuple=True)]
    ey,exx=[v.float().mean() for v in torch.nonzero(eb,as_tuple=True)]
    theta=torch.tensor([[[1.,0.,2*(exx-px)/H],[0.,1.,2*(ey-py)/H]]],dtype=ex.dtype,device=ex.device)
    grid=F.affine_grid(theta,(1,1,H,H),align_corners=False)
    return F.grid_sample(ex[None,None],grid,mode='bilinear',padding_mode='zeros',align_corners=False)[0,0]

def retrieval(q_emb,q_layout,candidates,embeddings,layouts,K,mode,random_state):
    if not candidates:return []
    if mode=='random':return random_state.sample(candidates,min(K,len(candidates)))
    c=torch.as_tensor(candidates);similarity=(embeddings[c]@q_emb).numpy()
    if mode!='appearance':
        import numpy as np
        d=np.linalg.norm(layouts[candidates]-q_layout,axis=1);similarity+=np.exp(-d/(d.mean()+1e-6))
    return [candidates[i] for i in (-similarity).argsort(kind='stable')[:K]]

def region_weights(frequencies):return {i:min(3.,1./max(float(f),1e-12)**.5) for i,f in enumerate(frequencies)}
