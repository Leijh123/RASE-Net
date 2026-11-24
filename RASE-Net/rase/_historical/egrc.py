"""EGRC v2: Exemplar-Guided Residual Corrector (创新点二).

v1 教训 (离线验证 2204 样本):
  - 仿射对齐 (缩放+旋转搜索) 本身引入大误差: 对齐后投票 Dice 0.735 << base 0.823
  - 纯先验锚点更差 (0.270): 同器官跨 slice 位置方差大, 平均位置无意义
v2 设计:
  - exemplar 只做质心平移对齐 (最鲁棒的单一变换), 不做缩放/旋转
  - K 个平移后 exemplar 保留各自形状细节, 编码为证据通道
  - cross-attention + 相对位置编码: 模型自己学 "该参考 exemplar 哪里"
  - 器官频率门控: 高频器官近似直通, 少样本器官放大 exemplar 通道
  - residual: final = sigmoid(logit(pred) + residual), 初始化直通
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EGRCCorrector(nn.Module):

    def __init__(self, d=128, n_organs=13, n_heads=4, K=4):
        super().__init__()
        self.d = d
        self.K = K
        # query 分支: img_emb (768ch@32) + [pred, field] (2ch@256)
        self.query_proj = nn.Sequential(
            nn.Conv2d(768, d, 1), nn.GroupNorm(8, d), nn.GELU())
        self.cond_proj = nn.Sequential(
            nn.Conv2d(2, d, 3, padding=1), nn.GroupNorm(8, d), nn.GELU(),
            nn.Conv2d(d, d, 3, padding=1), nn.GroupNorm(8, d), nn.GELU())
        # exemplar 分支: K 个质心平移 mask (K ch@256) → d ch@32
        self.ex_enc = nn.Sequential(
            nn.Conv2d(K, 64, 3, stride=2, padding=1), nn.GELU(),   # 128
            nn.Conv2d(64, d, 3, stride=2, padding=1), nn.GELU(),   # 64
            nn.Conv2d(d, d, 3, stride=2, padding=1), nn.GELU())    # 32
        # cross-attention + 可学习相对位置偏置 (3 维, 避免广播升维)
        self.pos_offset = nn.Parameter(torch.zeros(1, 1, d))
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        # 器官频率门控
        self.organ_gate = nn.Embedding(n_organs, 1)
        nn.init.constant_(self.organ_gate.weight, 0.0)
        # 解码: cat[q, fused, pred↓, field↓, vote↓] → residual logit @pred分辨率
        self.decoder = nn.Sequential(
            nn.Conv2d(2 * d + 3, 256, 3, padding=1), nn.GroupNorm(8, 256), nn.GELU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1), nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1), nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 1, 3, padding=1))

    def forward(self, img_emb, pred, field, ex_masks, organ_id):
        """img_emb (B,768,h,w); pred/field (B,1,256,256);
        ex_masks (B,K,1,256,256) 质心平移对齐; organ_id (B,) long.
        返回 residual logit (B,1,256,256)."""
        B, K = ex_masks.shape[:2]
        h = img_emb.shape[-1]
        q = self.query_proj(img_emb) + self.cond_proj(torch.cat([
            F.interpolate(pred, size=(h, h), mode='bilinear', align_corners=False),
            F.interpolate(field, size=(h, h), mode='bilinear', align_corners=False)],
            dim=1))                                              # (B,d,32,32)
        # exemplar: (B,K,1,256,256) → (B,K,256,256) 堆叠为 K 通道
        ex_cat = ex_masks.reshape(B, K, 256, 256)                 # (B,K,256,256)
        e32 = self.ex_enc(ex_cat)                                # (B,d,32,32)
        q_tok = q.flatten(2).transpose(1, 2)                     # (B,1024,d)
        e_tok = e32.flatten(2).transpose(1, 2)                   # (B,1024,d)
        # 相对位置感知: exemplar token 加可学习偏置 (键上), attention 学空间对应
        e_tok = e_tok + self.pos_offset
        att_tok, _ = self.attn(q_tok, e_tok, e_tok)              # (B,1024,d)
        att_map = att_tok.transpose(1, 2).reshape(B, self.d, h, h)
        g = torch.sigmoid(self.organ_gate(organ_id)).view(B, 1, 1, 1)
        fused = q + g * att_map                                  # (B,d,32,32)
        # 显式投票证据: K 个对齐 exemplar 的均值 (降低 attention 学习负担)
        vote = ex_masks.mean(dim=1)                                # (B,1,256,256)
        dec_in = torch.cat([
            q, fused,
            F.interpolate(pred, size=(h, h), mode='bilinear', align_corners=False),
            F.interpolate(field, size=(h, h), mode='bilinear', align_corners=False),
            F.interpolate(vote, size=(h, h), mode='bilinear', align_corners=False)],
            dim=1)
        res = self.decoder(dec_in)
        if res.shape[-1] != pred.shape[-1]:          # emb 分辨率≠256 时对齐
            res = F.interpolate(res, size=pred.shape[-2:], mode='bilinear',
                                align_corners=False)
        return res


def logit(p, eps=1e-4):
    p = p.clamp(eps, 1 - eps)
    return torch.log(p / (1 - p))


def soft_dice_loss(prob, gt):
    """prob/gt (B,1,H,W) in [0,1]."""
    s = prob.flatten(1).sum(1)
    t = gt.flatten(1).sum(1)
    inter = (prob * gt).flatten(1).sum(1)
    return (1 - (2 * inter + 1) / (s + t + 1)).mean()
