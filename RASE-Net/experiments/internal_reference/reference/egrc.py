"""EGRC revision: coordinate-aware exemplar fusion and conservative residual gating.

The exemplar masks are separate input channels, mixed by convolution. This is
not a claim of independently attended exemplar tokens or relative-position bias.
"""
import torch
from torch import nn
from torch.nn import functional as F

class EGRCCorrector(nn.Module):
    def __init__(self,d=128,n_organs=13,n_heads=4,K=4):
        super().__init__();self.d=d;self.K=K
        def block(cin,cout):return nn.Sequential(nn.Conv2d(cin,cout,3,padding=1),nn.GroupNorm(8,cout),nn.GELU())
        self.query_proj=nn.Sequential(nn.Conv2d(768,d,1),nn.GroupNorm(8,d),nn.GELU())
        self.cond_proj=nn.Sequential(block(2,d),block(d,d))
        self.ex_enc=nn.Sequential(nn.Conv2d(K,64,3,2,1),nn.GELU(),nn.Conv2d(64,d,3,2,1),nn.GELU(),nn.Conv2d(d,d,3,2,1),nn.GELU())
        self.coordinate_proj=nn.Conv2d(2,d,1,bias=False)
        self.attn=nn.MultiheadAttention(d,n_heads,batch_first=True)
        self.organ_gate=nn.Embedding(n_organs,1);nn.init.zeros_(self.organ_gate.weight)
        self.sample_gate=nn.Sequential(nn.Linear(d+4,32),nn.GELU(),nn.Linear(32,1))
        nn.init.zeros_(self.sample_gate[-1].weight);nn.init.constant_(self.sample_gate[-1].bias,-2.)
        self.decoder=nn.Sequential(block(2*d+3,256),nn.Dropout2d(.1),block(256,128),nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),nn.Conv2d(128,64,3,padding=1),nn.GELU(),nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),nn.Conv2d(64,32,3,padding=1),nn.GELU(),nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),nn.Conv2d(32,1,3,padding=1))
        nn.init.zeros_(self.decoder[-1].weight);nn.init.zeros_(self.decoder[-1].bias)
    def forward(self,img_emb,pred,field,ex_masks,organ_id):
        B,K=ex_masks.shape[:2];h,w=img_emb.shape[-2:]
        down=lambda x:F.interpolate(x,size=(h,w),mode='bilinear',align_corners=False)
        p,f=down(pred),down(field)
        q=self.query_proj(img_emb)+self.cond_proj(torch.cat([p,f],1))
        ex=ex_masks[:,:,0];e=self.ex_enc(ex)
        e=F.interpolate(e,size=(h,w),mode='bilinear',align_corners=False)
        y,x=torch.meshgrid(torch.linspace(-1,1,h,device=q.device,dtype=q.dtype),torch.linspace(-1,1,w,device=q.device,dtype=q.dtype),indexing='ij')
        pos=self.coordinate_proj(torch.stack([x,y])[None])
        qt=(q+pos).flatten(2).transpose(1,2);kt=(e+pos).flatten(2).transpose(1,2)
        a,_=self.attn(qt,kt,e.flatten(2).transpose(1,2),need_weights=False)
        a=a.transpose(1,2).reshape(B,self.d,h,w)
        vote=down(ex_masks.mean(1));uncertainty=4*p*(1-p)
        stats=torch.stack([uncertainty.mean((1,2,3)),f.mean((1,2,3)),(vote-p).abs().mean((1,2,3)),vote.mean((1,2,3))],1)
        pooled=q.mean((2,3));g=torch.sigmoid(self.sample_gate(torch.cat([pooled,stats],1))+self.organ_gate(organ_id)).view(B,1,1,1)
        valid=(ex_masks.flatten(1).sum(1)>0).to(q.dtype).view(B,1,1,1)
        g=g*valid
        fused=q+g*a
        res=self.decoder(torch.cat([q,fused,p,f,g*vote],1))
        res=F.interpolate(res,size=pred.shape[-2:],mode='bilinear',align_corners=False)
        return g*res

def logit(p,eps=1e-4):
    p=p.clamp(eps,1-eps);return torch.log(p/(1-p))

def soft_dice_loss(prob,gt):
    s=prob.flatten(1).sum(1);t=gt.flatten(1).sum(1);i=(prob*gt).flatten(1).sum(1)
    return (1-(2*i+1)/(s+t+1)).mean()
