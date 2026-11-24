"""Semantic Anatomical Contradiction Field (SACF) 网络.

第一创新点核心模块: Counterfactual Anatomical Semantic Criterion (CASC).

因果链:
    Structured Medical Semantics (T)
        -> Expected Anatomy   (文本约束编码, FiLM 调制)
    (I, M) -> Observed Anatomy (mask/图像/器官上下文编码)
    Expected vs Observed
        -> 四类 semantic evidence (identity / spatial / shape / relational)
        -> Organ-adaptive Gating (lambda(x) 由文本+视觉动态产生)
        -> Semantic Contradiction Field C(x) ∈ [0,1]^{H×W}
        -> 全局 contradiction score (用于 ranking)

推理阶段仅使用 (I, T, M_t), 完全 GT-free.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

EVIDENCE_NAMES = ['identity', 'spatial', 'shape', 'relational']


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.GroupNorm(8, cout),
            nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.GroupNorm(8, cout),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class MaskEncoder(nn.Module):
    """mask (N,1,H,W) -> 多尺度特征 (f16, f4, f2), 提取 observed anatomy 的 mask 表征.

    f16: (N,D,H/16)  语义证据推断; f4: (N,D/2,H/4), f2: (N,D/4,H/2) 精修跳连.
    """

    def __init__(self, dim=192):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, dim // 4, 4, stride=2, padding=1), nn.GELU(),      # /2
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(dim // 4, dim // 2, 4, stride=2, padding=1), nn.GELU(),  # /4
        )
        self.mid = ConvBlock(dim // 2, dim // 2)
        self.down2 = nn.Sequential(
            nn.Conv2d(dim // 2, dim, 4, stride=2, padding=1), nn.GELU(),    # /8
            nn.Conv2d(dim, dim, 4, stride=2, padding=1), nn.GELU(),         # /16
        )
        self.out = ConvBlock(dim, dim)

    def forward(self, m):
        f2 = self.enc1(m)
        f4 = self.enc2(f2)
        f4 = self.mid(f4) + f4
        f16 = self.out(self.down2(f4))
        return f16, f4, f2


class ContextEncoder(nn.Module):
    """多器官上下文 (N,C,H,W) -> (N,D,H/16,W/16), 编码器官间空间关系."""

    def __init__(self, num_organs, dim=192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(num_organs, dim // 2, 4, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(dim // 2, dim // 2, 4, stride=2, padding=1), nn.GELU(),
            ConvBlock(dim // 2, dim // 2),
            nn.Conv2d(dim // 2, dim, 4, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(dim, dim, 4, stride=2, padding=1), nn.GELU(),
            ConvBlock(dim, dim),
        )

    def forward(self, c):
        return self.net(c)


class TextConstraintEncoder(nn.Module):
    """文本语义约束: text emb (N,E_t) -> 每类 evidence 的 FiLM (gamma, beta).

    T -> Expected Anatomy: 文本不再作为分割 prompt, 而是作为独立的
    anatomical criterion, 通过 FiLM 调制四类 evidence head.
    """

    def __init__(self, text_dim, dim=192, num_evidence=4):
        super().__init__()
        self.num_evidence = num_evidence
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(text_dim, dim * 2), nn.GELU(),
            nn.Linear(dim * 2, dim * 2),
        )
        self.gamma_head = nn.Linear(dim * 2, num_evidence * dim)
        self.beta_head = nn.Linear(dim * 2, num_evidence * dim)

    def forward(self, t):
        h = self.mlp(t)
        g = self.gamma_head(h).view(-1, self.num_evidence, 1, 1, self.dim)
        b = self.beta_head(h).view(-1, self.num_evidence, 1, 1, self.dim)
        return g, b  # (N,K,1,1,D) 各 evidence 的 FiLM 参数


class AnatomicalComparator(nn.Module):
    """Expected vs Observed Anatomy 的显式差异向量 (必改2).

    论文公式:
        Z_M   = Proj( pool(F_mask) ⊕ pool(F_img) )     观测解剖全局表征
        Z_T^k = MLP_k(T)                                期望解剖逐证据约束向量
        D_k   = | Z_M - Z_T^k |                         期望-观测差异向量

    D_k 与 FiLM 调制后的 mask 特征、条件依据一起进入第 k 个 evidence
    head, 使证据推断显式依赖 "期望解剖与观测解剖的全局不一致程度",
    而非仅靠 text-conditioned 特征变换隐式携带约束.
    """

    def __init__(self, dim=192, text_dim=768, num_evidence=4):
        super().__init__()
        self.num_evidence = num_evidence
        self.proj_m = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.proj_t = nn.ModuleList([
            nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(), nn.Linear(dim, dim))
            for _ in range(num_evidence)])

    def forward(self, f_mask, f_img, t_emb):
        zm = torch.cat([
            F.adaptive_avg_pool2d(f_mask, 1).flatten(1),
            F.adaptive_avg_pool2d(f_img, 1).flatten(1)], dim=1)          # (N,2D)
        zm = self.proj_m(zm)                                              # (N,D)
        ds = [(zm - pt(t_emb)).abs() for pt in self.proj_t]               # K x (N,D)
        return torch.stack(ds, dim=1)                                     # (N,K,D)


class EvidenceHead(nn.Module):
    """单类语义证据 -> 16x16 logit 图."""

    def __init__(self, cin, dim=192):
        super().__init__()
        self.inp = nn.Conv2d(cin, dim, 1)
        self.block = ConvBlock(dim, dim)
        self.head = nn.Conv2d(dim, 1, 1)

    def forward(self, x):
        return self.head(self.block(self.inp(x)))  # (N,1,h,w)


class OrganAdaptiveGate(nn.Module):
    """Organ-adaptive semantic gating: lambda = softmax(g(Fm, T)).

    不同器官自动强调不同语义约束 (如 liver 偏 shape, pancreas 偏 relational).
    """

    def __init__(self, dim=192, text_dim=768, num_evidence=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + text_dim, dim), nn.GELU(),
            nn.Linear(dim, num_evidence),
        )

    def forward(self, f_mask, t_emb):
        pooled = F.adaptive_avg_pool2d(f_mask, 1).flatten(1)  # (N,D)
        logits = self.net(torch.cat([pooled, t_emb], dim=1))
        return torch.softmax(logits, dim=1)  # (N,K)


class FieldRefinement(nn.Module):
    """高分辨率矛盾场精修: 32x32 粗场 + mask encoder 多尺度跳连 + 边界先验.

    粗场在 H/16 推断 (语义聚合), 在此解码到全分辨率 (精确定位),
    解决细边界错误无法在低分辨率场中定位的问题.
    """

    def __init__(self, dim=192):
        super().__init__()
        self.inp = nn.Sequential(
            nn.Conv2d(dim + 1, dim, 3, padding=1, bias=False),
            nn.GroupNorm(8, dim), nn.GELU(),
        )
        self.up1 = nn.Sequential(                       # 32 -> 64
            nn.Conv2d(dim, dim // 2, 3, padding=1, bias=False),
            nn.GroupNorm(8, dim // 2), nn.GELU(),
        )
        self.up2 = nn.Sequential(                       # 128, cat f4 (dim//2)
            nn.Conv2d(dim, dim // 2, 3, padding=1, bias=False),
            nn.GroupNorm(8, dim // 2), nn.GELU(),
        )
        self.up3 = nn.Sequential(                       # 256, cat f2 (dim//4)
            nn.Conv2d(dim // 2 + dim // 4, dim // 4, 3, padding=1, bias=False),
            nn.GroupNorm(8, dim // 4), nn.GELU(),
        )
        # 256 分辨率融合边界先验后输出 logit, 再上采样到全分辨率
        self.final = nn.Sequential(
            nn.Conv2d(dim // 4 + 2, 16, 3, padding=1), nn.GELU(),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, f16, coarse, f4, f2, mask):
        # mask 边界先验 @ H/2 (max_pool 下采样保留细结构)
        m2 = F.max_pool2d(mask, 2)
        edge = (F.max_pool2d(m2, 5, 1, 2)
                - (1 - F.max_pool2d(1 - m2, 5, 1, 2))).clamp(0, 1)

        x = self.inp(torch.cat([f16, coarse], dim=1))           # H/16
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.up1(x)                                          # H/8
        x = F.interpolate(x, size=f4.shape[-2:], mode='bilinear', align_corners=False)
        x = self.up2(torch.cat([x, f4], dim=1))                  # H/4
        x = F.interpolate(x, size=f2.shape[-2:], mode='bilinear', align_corners=False)
        x = self.up3(torch.cat([x, f2], dim=1))                  # H/2
        logit = self.final(torch.cat([x, m2, edge], dim=1))      # H/2
        logit = F.interpolate(logit, size=mask.shape[-2:],
                              mode='bilinear', align_corners=False)
        return logit                                             # H


class SemanticContradictionField(nn.Module):
    """SACF 主网络.

    输入:
        image_embed (N,768,h,w): 冻结 IMIS encoder 的视觉特征 (detached)
        mask        (N,1,H,W):   当前 segmentation (训练时为 GT/反事实; 推理时为模型预测)
        context     (N,C,H,W):   其他器官 mask (训练用 GT 其他器官; 推理用其他器官预测)
        text_embed  (N,768):     器官文本语义编码
    输出:
        fields  (N,4,H,W): 四类 evidence 的矛盾概率图
        fused   (N,1,H,W): organ-adaptive 融合的 Semantic Contradiction Field C(x)
        score   (N,):      全局 contradiction score
        gate    (N,4):     gating 权重 lambda
    """

    def __init__(self, num_organs=13, image_embed_dim=768, text_embed_dim=768,
                 dim=192, image_size=256, feat_stride=16):
        super().__init__()
        self.num_evidence = 4
        self.feat_stride = feat_stride
        self.image_size = image_size

        self.mask_encoder = MaskEncoder(dim)
        self.context_encoder = ContextEncoder(num_organs, dim)
        self.image_proj = nn.Conv2d(image_embed_dim, dim, 1)

        # 形态先验: 边界带 + 两个距离尺度 (骨架近似), 编码 shape 证据
        self.edge_enc = nn.Sequential(
            nn.Conv2d(3, dim, 4, stride=4, padding=1), nn.GELU(),
            nn.Conv2d(dim, dim, 4, stride=4, padding=1), nn.GELU(),
        )
        # 空间先验: 归一化坐标网格
        self.coord_enc = nn.Sequential(
            nn.Conv2d(2, dim, 4, stride=4, padding=1), nn.GELU(),
            nn.Conv2d(dim, dim, 4, stride=4, padding=1), nn.GELU(),
        )

        self.text_constraint = TextConstraintEncoder(text_embed_dim, dim, self.num_evidence)

        # Expected vs Observed 显式比较 (必改2)
        self.comparator = AnatomicalComparator(dim, text_embed_dim, self.num_evidence)

        # 四类 evidence head 的输入: [FiLM(Fm), 条件特征, D_k] -> 3D
        self.evidence_heads = nn.ModuleList([
            EvidenceHead(dim * 3, dim) for _ in range(self.num_evidence)
        ])

        self.gate = OrganAdaptiveGate(dim, text_embed_dim, self.num_evidence)
        self.refine = FieldRefinement(dim)

        self.score_head = nn.Sequential(
            nn.Linear(dim + text_embed_dim + 3, dim), nn.GELU(),
            nn.Linear(dim, 1),
        )

        self.register_buffer('coord_grid', self._build_coord_grid(image_size), persistent=False)

    @staticmethod
    def _build_coord_grid(size):
        ys = torch.linspace(-1, 1, size).view(1, 1, size, 1).expand(1, 1, size, size)
        xs = torch.linspace(-1, 1, size).view(1, 1, 1, size).expand(1, 1, size, size)
        return torch.cat([xs, ys], dim=1)

    def forward(self, image_embed, mask, context, text_embed):
        N = mask.shape[0]
        f_mask, f4, f2 = self.mask_encoder(mask)            # f_mask: (N,D,h,w)
        f_ctx = self.context_encoder(context)               # (N,D,h,w)
        f_img = self.image_proj(image_embed)                # (N,D,h,w)

        # 边界带 (shape 先验)
        m = mask
        edge = (F.max_pool2d(m, 5, 1, 2) - (1 - F.max_pool2d(1 - m, 5, 1, 2))).clamp(0, 1)
        dist_in = self._cheap_edt(m)
        f_edge = self.edge_enc(torch.cat([edge, dist_in], dim=1))
        f_coord = self.coord_enc(self.coord_grid.expand(N, -1, -1, -1))

        # Expected anatomy: 文本 -> FiLM 约束 + 期望-观测差异向量
        gamma, beta = self.text_constraint(text_embed)      # (N,K,1,1,D)
        D = self.comparator(f_mask, f_img, text_embed)      # (N,K,D) 必改2

        cond = [f_img, f_coord, f_edge, f_ctx]
        h16, w16 = f_mask.shape[-2:]
        fields = []
        for k, head in enumerate(self.evidence_heads):
            g = gamma[:, k].view(N, -1, 1, 1)
            b = beta[:, k].view(N, -1, 1, 1)
            fm_k = f_mask * (1 + g) + b                       # FiLM 调制
            dk = D[:, k].view(N, -1, 1, 1).expand(-1, -1, h16, w16)
            x = torch.cat([fm_k, cond[k], dk], dim=1)         # (N, 3D, h, w)
            fields.append(head(x))                            # (N,1,h,w)

        fields32 = torch.cat(fields, dim=1)                  # (N,K,h,w) 低分辨率 evidence
        fields = F.interpolate(fields32, size=mask.shape[-2:],
                               mode='bilinear', align_corners=False)  # (N,K,H,W)

        # organ-adaptive gating 融合 (概率域凸组合) -> 粗矛盾场
        lam = self.gate(f_mask, text_embed)                  # (N,K)
        coarse = (fields32.view(N, self.num_evidence, -1).sigmoid()
                  * lam.view(N, self.num_evidence, 1)).sum(dim=1)
        coarse = coarse.view(N, 1, *fields32.shape[-2:])     # (N,1,h,w)

        # 高分辨率精修: 多尺度跳连 + 边界先验 -> 全分辨率 C(x)
        refined_logit = self.refine(f_mask, coarse, f4, f2, mask)   # (N,1,H,W)
        fused = refined_logit.sigmoid()
        fused_coarse = F.interpolate(coarse, size=mask.shape[-2:],
                                     mode='bilinear', align_corners=False).sigmoid()

        # 全局 score: 视觉 + 文本 + field 统计
        pooled = F.adaptive_avg_pool2d(f_mask, 1).flatten(1)
        c = fused.flatten(1)
        stats = torch.stack([c.mean(1), c.max(1).values,
                             c.topk(min(1024, c.shape[1]), dim=1).values.mean(1)], dim=1)
        score = self.score_head(torch.cat([pooled, text_embed, stats], dim=1)).squeeze(1)

        fields_prob = fields.sigmoid()
        return {
            'fields': fields_prob,       # (N,4,H,W) 各 evidence 矛盾概率
            'fused': fused,              # (N,1,H,W) 精修后的 C(x)
            'fused_coarse': fused_coarse,  # (N,1,H,W) 粗场 (辅助监督/分析)
            'score': score,              # (N,)
            'gate': lam,                 # (N,4)
        }

    @staticmethod
    def _cheap_edt(m, iters=16):
        """近似欧氏距离 (内侧多尺度), 编码器官厚度/尺寸先验."""
        ds = []
        cur = m
        for i in range(4):
            cur = F.max_pool2d(cur, 3, 2, 1)  # 逐步 /2
            ds.append(F.interpolate(cur, size=m.shape[-2:], mode='nearest'))
        return torch.cat(ds, dim=1)[:, :2]  # 取两个尺度


class FieldGuidedRefiner(nn.Module):
    """矛盾场指导的 mask 精修 (闭环应用).

    因果链:
        SACF C(x) 指出 "哪里违反解剖语义"
            -> 空间注意力集中于高矛盾区域 (field attention)
            -> 多尺度解码 log 域残差 delta
            -> 修正 mask = sigmoid(logit_pred + delta)

    输入均为推理可得 (GT-free):
        image_embed (N,768,h,w)  冻结 encoder 特征
        mask        (N,1,H,W)    IMISNet 预测概率
        field       (N,1,H,W)    SACF 矛盾场 C(x)
        text_embed  (N,768)      器官文本语义 (FiLM 调制)
    """

    def __init__(self, image_embed_dim=768, text_embed_dim=768, dim=192):
        super().__init__()
        self.mask_encoder = MaskEncoder(dim)          # 多尺度 (f16, f4, f2)
        self.img_proj = nn.Conv2d(image_embed_dim, dim, 1)
        self.field_enc = nn.Sequential(              # C(x) -> (N,dim,h/16)
            nn.Conv2d(1, dim // 2, 4, stride=4, padding=1), nn.GELU(),
            nn.Conv2d(dim // 2, dim, 4, stride=4, padding=1), nn.GELU(),
        )
        self.fuse = ConvBlock(dim * 3, dim)          # [f_mask, f_img, f_field]

        self.text_mlp = nn.Sequential(               # 文本 FiLM
            nn.Linear(text_embed_dim, dim * 2), nn.GELU(),
            nn.Linear(dim * 2, dim * 2),
        )

        # 解码: /16 -> /4 -> /2 -> /1, 跳连 f4, f2, mask, edge
        self.dec4 = ConvBlock(dim + dim // 2, dim // 2)      # cat f4
        self.dec2 = ConvBlock(dim // 2 + dim // 4, dim // 4)  # cat f2
        self.final = nn.Sequential(
            nn.Conv2d(dim // 4 + 2, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, image_embed, mask, field, text_embed):
        N = mask.shape[0]
        f16, f4, f2 = self.mask_encoder(mask)
        f_img = self.img_proj(image_embed)
        f_field = self.field_enc(field)

        # 场空间注意力: 高矛盾区域特征增强 (C(x) 作为 gate)
        attn = F.interpolate(field, size=f16.shape[-2:],
                              mode='bilinear', align_corners=False)
        base = self.fuse(torch.cat([f16, f_img, f_field], dim=1))
        base = base * (1.0 + attn)

        # 文本 FiLM 调制 (器官特异的修正策略)
        h = self.text_mlp(text_embed)
        gamma, beta = h.chunk(2, dim=1)
        base = base * (1.0 + gamma.view(N, -1, 1, 1)) + beta.view(N, -1, 1, 1)

        # 解码 + 跳连
        x = F.interpolate(base, size=f4.shape[-2:],
                          mode='bilinear', align_corners=False)
        x = self.dec4(torch.cat([x, f4], dim=1))
        x = F.interpolate(x, size=f2.shape[-2:],
                          mode='bilinear', align_corners=False)
        x = self.dec2(torch.cat([x, f2], dim=1))

        edge = (F.max_pool2d(mask, 5, 1, 2)
                - (1 - F.max_pool2d(1 - mask, 5, 1, 2))).clamp(0, 1)
        x = F.interpolate(x, size=mask.shape[-2:],
                          mode='bilinear', align_corners=False)
        delta = self.final(torch.cat([x, mask, edge], dim=1))

        # log 域残差: 保留预测mask主体, 只在高矛盾区域修正
        p = mask.clamp(1e-6, 1 - 1e-6)
        logit_m = torch.log(p / (1 - p))
        return logit_m + delta
