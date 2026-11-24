"""SACF 训练目标: Semantic Contradiction Ranking + Dense Field + Score.

核心: Contradiction Ranking Loss
    L_rank = max(0, m + C(M+, T) - C(M-, T))
    正确 mask 的语义矛盾必须低于反事实错误 mask.

Dense field loss: C(x) 在反事实错误区域 (XOR) 应为高, 其余区域为低.
Score loss: 全局 contradiction score 区分 correct / corrupted.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SACFLoss(nn.Module):
    def __init__(self, w_field=1.0, w_rank=1.0, w_score=0.5,
                 margin=0.3, pos_weight=20.0, field_gamma=2.0,
                 w_coarse=0.2):
        super().__init__()
        self.w_field = w_field
        self.w_rank = w_rank
        self.w_score = w_score
        self.w_coarse = w_coarse
        self.margin = margin
        self.register_buffer('pos_weight', torch.tensor([pos_weight]))
        self.field_gamma = field_gamma

    def _dense_field_loss(self, fused, err):
        """focal-style weighted BCE on 矛盾场."""
        p = fused.clamp(1e-6, 1 - 1e-6)
        bce = -(self.pos_weight * err * torch.log(p)
                + (1 - err) * torch.log(1 - p))
        pt = torch.where(err > 0.5, p, 1 - p)
        focal = (1 - pt).pow(self.field_gamma)
        return (bce * focal).mean()

    def forward(self, out, targets):
        """out: SACF 输出 dict; targets: dict(error, is_neg, group)."""
        # 强制 fp32, 避免 amp 下 log(1-p) 下溢
        fused = out['fused'].float()   # (N,1,H,W) 精修后概率
        score = out['score'].float()   # (N,)
        err = targets['error']         # (N,1,H,W) {0,1}
        is_neg = targets['is_neg']     # (N,)
        group = targets['group']       # (N,) 同一 GT 样本的配对 id

        # ---- ① dense contradiction field (精修场 + 粗场辅助) ----
        l_field = self._dense_field_loss(fused, err)
        if self.w_coarse > 0 and 'fused_coarse' in out:
            l_field = l_field + self.w_coarse * self._dense_field_loss(
                out['fused_coarse'].float(), err)

        # ---- ② contradiction ranking (配对 margin) ----
        # 每个 group: 1 个正样本 (M+) + 若干负样本 (M-), 要求 s+ < s-
        losses = []
        for g in torch.unique(group):
            sel = group == g
            if sel.sum() < 2:
                continue
            s = score[sel]
            neg_mask = is_neg[sel].bool()
            if neg_mask.sum() == 0 or (~neg_mask).sum() == 0:
                continue
            s_pos = s[~neg_mask].mean()
            s_neg = s[neg_mask]
            losses.append(F.relu(self.margin + s_pos - s_neg).mean())
        l_rank = torch.stack(losses).mean() if losses else fused.new_zeros(())

        # ---- ③ score 判别 (corrupted vs correct) ----
        l_score = F.binary_cross_entropy_with_logits(score, is_neg.float())

        loss = self.w_field * l_field + self.w_rank * l_rank + self.w_score * l_score
        return loss, {
            'field': l_field.detach(),
            'rank': l_rank.detach(),
            'score': l_score.detach(),
        }


@torch.no_grad()
def field_iou(fused, error, thresh=0.5, adaptive=False, quantile=0.05):
    """Error Localization IoU: 二值化 C(x) 与真实语义违反区域的 IoU.

    adaptive=True: 分位数阈值 (GT-free 协议) —— 选中场值 top-`quantile` 的像素
    作为预测错误区域. 误差区域通常占图像 1~10%, 均值阈值会选中约一半像素,
    IoU 天花板过低, 因此采用固定占比的分位数阈值.
    """
    if adaptive:
        t = fused.flatten(1).quantile(1 - quantile, dim=1).view(-1, 1, 1, 1)
        pred = (fused >= t).float()
    else:
        pred = (fused > thresh).float()
    inter = (pred * error).sum(dim=(1, 2, 3))
    union = ((pred + error) > 0).float().sum(dim=(1, 2, 3))
    iou = inter / (union + 1e-6)
    valid = error.sum(dim=(1, 2, 3)) > 0  # 只统计有错误的样本
    if valid.sum() == 0:
        return None
    return iou[valid].mean().item()


@torch.no_grad()
def field_iou_oracle(fused, error, num_t=32):
    """Oracle 上界: 逐样本扫描阈值取最优 IoU (分析用, 非 GT-free)."""
    valid = error.sum(dim=(1, 2, 3)) > 0
    if valid.sum() == 0:
        return None
    f, e = fused[valid], error[valid]
    flat_f = f.flatten(1)
    best = []
    for i in range(f.shape[0]):
        fi, ei = flat_f[i], e[i].flatten()
        if ei.sum() == 0:
            continue
        # 候选阈值: 场值分位数网格
        qs = torch.linspace(0.5, 0.999, num_t, device=fi.device)
        ts = torch.quantile(fi, qs)
        best_iou = 0.0
        for t in ts:
            pred = (fi >= t).float()
            inter = (pred * ei).sum()
            union = ((pred + ei) > 0).float().sum()
            best_iou = max(best_iou, (inter / (union + 1e-6)).item())
        best.append(best_iou)
    return sum(best) / len(best) if best else None


@torch.no_grad()
def pixel_auroc(fused, error):
    """逐像素 AUROC (按 batch 聚合计算, 简化实现: 基于排序)."""
    scores = fused.flatten(1)  # (N, HW)
    labels = error.flatten(1)
    # 只统计有错误的样本
    valid = labels.sum(1) > 0
    if valid.sum() == 0:
        return None
    scores, labels = scores[valid], labels[valid]
    s = scores.flatten()
    y = labels.flatten()
    if y.sum() == 0 or y.sum() == y.numel():
        return None
    # AUROC via rank statistic (升序 rank: 最低分 rank=1)
    order = s.argsort()
    ranks = torch.empty_like(s, dtype=torch.float64)
    ranks[order] = torch.arange(1, s.numel() + 1, device=s.device, dtype=torch.float64)
    n_pos = y.sum().double()
    n_neg = y.numel() - n_pos
    auroc = (ranks[y > 0.5].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return auroc.item()
