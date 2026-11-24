"""Counterfactual Anatomical Corruptions (反事实解剖扰动生成器).

从 GT mask 自动合成六类具有明确医学语义的分割错误状态:
  1. boundary       边界错误 (erosion / dilation / 边界位移)
  2. missing        漏分割 (missing-region / FN)
  3. false_positive 假阳性 (FP, 器官外错误区域)
  4. location       位置错误 (mask 移动到错误解剖位置)
  5. shape          形态错误 (各向异性拉伸 / 弹性形变)
  6. relational     关系错误 (与其他器官的空间关系被破坏)

每个 corruption 返回 (corrupted_mask, error_map)，
error_map = XOR(corrupted, GT) 即"违反医学语义的区域"，
作为 Semantic Contradiction Field 的稠密监督信号。

核心假设: Few labeled volumes != Few semantic training states.
一个 GT 通过本模块可以生成大量语义违反状态, 从而在 BTCV
小样本条件下训练出具有判别力的解剖语义判据。
"""

import torch
import torch.nn.functional as F

CORRUPTION_TYPES = [
    'boundary', 'missing', 'false_positive',
    'location', 'shape', 'relational',
]
TYPE_TO_INDEX = {t: i for i, t in enumerate(CORRUPTION_TYPES)}


# ---------------------------------------------------------------- utils

def _dilate(m, k):
    """形态学膨胀, k 为半径(像素)."""
    if k <= 0:
        return m
    return F.max_pool2d(m, kernel_size=2 * k + 1, stride=1, padding=k)


def _erode(m, k):
    """形态学腐蚀."""
    if k <= 0:
        return m
    return 1.0 - F.max_pool2d(1.0 - m, kernel_size=2 * k + 1, stride=1, padding=k)


def _affine(m, theta):
    """按 batch 应用 affine 变换. theta: (N,2,3) 输出坐标->输入坐标."""
    grid = F.affine_grid(theta, m.shape, align_corners=False)
    return F.grid_sample(m, grid, mode='nearest', align_corners=False)


def _translate(m, dx, dy):
    """平移 mask, (dx,dy) 为像素偏移(正值=右/下)."""
    N, _, H, W = m.shape
    theta = torch.zeros(N, 2, 3, device=m.device, dtype=m.dtype)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    theta[:, 0, 2] = -2.0 * dx / W
    theta[:, 1, 2] = -2.0 * dy / H
    return _affine(m, theta)


def _scale_about_centroid(m, sx, sy):
    """关于 mask 质心的各向异性缩放, 用于 shape corruption."""
    N, _, H, W = m.shape
    pos = m.sum(dim=(1, 2, 3)) + 1e-6
    ys = torch.arange(H, device=m.device, dtype=m.dtype).view(1, H, 1).expand(N, H, W)
    xs = torch.arange(W, device=m.device, dtype=m.dtype).view(1, 1, W).expand(N, H, W)
    w = m[:, 0]
    cy = (w * ys).sum(dim=(1, 2)) / pos
    cx = (w * xs).sum(dim=(1, 2)) / pos
    # 归一化质心 [-1,1]
    cnx = 2.0 * cx / W - 1.0
    cny = 2.0 * cy / H - 1.0
    theta = torch.zeros(N, 2, 3, device=m.device, dtype=m.dtype)
    theta[:, 0, 0] = 1.0 / sx
    theta[:, 1, 1] = 1.0 / sy
    theta[:, 0, 2] = cnx * (1.0 - 1.0 / sx)
    theta[:, 1, 2] = cny * (1.0 - 1.0 / sy)
    return _affine(m, theta)


def _rand_in_mask(m, generator=None):
    """在 mask 内随机采样一个像素坐标, 返回 (cx, cy) 像素坐标 tensor (N,)."""
    N, _, H, W = m.shape
    flat = m[:, 0].reshape(N, -1)
    idx = torch.multinomial(flat + 1e-8, 1, generator=generator).squeeze(1)  # (N,)
    cy = (idx // W).float()
    cx = (idx % W).float()
    return cx, cy


def _smooth_noise(N, H, W, device, generator=None, scale=1.0):
    """生成平滑随机场 (N,2,H,W), 用于弹性形变."""
    noise = torch.randn(N, 2, H // 8, W // 8, device=device, generator=generator)
    noise = F.interpolate(noise, size=(H, W), mode='bilinear', align_corners=False)
    noise = _dilate_v(noise, 3)
    nrm = noise.flatten(2).norm(dim=2, keepdim=True).unsqueeze(-1) + 1e-6
    return noise / nrm * scale


def _dilate_v(v, k):
    """向量场的盒式平滑 (对每个通道做 avg pool)."""
    return F.avg_pool2d(v, kernel_size=2 * k + 1, stride=1, padding=k)


def _disk(H, W, device, cx, cy, r):
    """(N,) 个圆盘叠加成 (N,1,H,W) bool."""
    n = cx.shape[0]
    ys = torch.arange(H, device=device, dtype=torch.float32).view(1, H, 1)
    xs = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, W)
    d = (xs.view(1, 1, W) - cx.view(n, 1, 1)) ** 2 + (ys.view(1, H, 1) - cy.view(n, 1, 1)) ** 2
    return (d <= r.view(n, 1, 1) ** 2).float().unsqueeze(1)


# ---------------------------------------------------------------- corruptions
# 所有像素量级均以 256 分辨率为基准定义, 通过 scale (image_size/256) 缩放,
# 保证不同输入分辨率下 corruption 相对于解剖尺度的强度一致.

def corrupt_boundary(m, severity, generator=None, scale=1.0):
    """边界错误: 随机 erosion / dilation / 小幅边界位移."""
    N = m.shape[0]
    dev = m.device
    out = m.clone()
    for i in range(N):
        s = severity[i].item()
        k = max(1, int(round((1 + s * 6) * scale)))  # 基准 1~7 px
        mode = torch.randint(0, 3, (1,), generator=generator, device=dev).item()
        mi = m[i:i + 1]
        if mode == 0:
            out[i:i + 1] = _erode(mi, k)
        elif mode == 1:
            out[i:i + 1] = _dilate(mi, k)
        else:  # 边界随机位移
            dx = torch.randint(-k, k + 1, (1,), generator=generator, device=dev).item()
            dy = torch.randint(-k, k + 1, (1,), generator=generator, device=dev).item()
            out[i:i + 1] = _translate(mi, dx, dy)
    return out


def corrupt_missing(m, severity, generator=None, scale=1.0):
    """漏分割: 在器官内部删除随机圆盘区域 (FN)."""
    N, _, H, W = m.shape
    dev = m.device
    area = m.sum(dim=(1, 2, 3))
    cx, cy = _rand_in_mask(m, generator)
    r = (area.clamp(min=1.0).sqrt() * severity * 0.45).clamp(min=3.0 * scale)
    rm = _disk(H, W, dev, cx, cy, r)
    return m * (1.0 - rm)


def corrupt_false_positive(m, severity, generator=None, scale=1.0):
    """假阳性: 在器官附近 (边界外) 添加随机错误区域."""
    N, _, H, W = m.shape
    dev = m.device
    area = m.sum(dim=(1, 2, 3))
    # 采样点: 膨胀区域减去器官本身 => 器官邻域
    ring = (_dilate(m, int(20 * scale)) - m).clamp(0, 1)
    ring = ring + 1e-8
    cx, cy = _rand_in_mask(ring, generator)
    r = (area.clamp(min=1.0).sqrt() * severity * 0.4).clamp(min=3.0 * scale)
    add = _disk(H, W, dev, cx, cy, r)
    return (m + add).clamp(0, 1)


def corrupt_location(m, severity, generator=None, scale=1.0):
    """位置错误: 将 mask 平移到错误解剖位置 (基准 40~100 px)."""
    N = m.shape[0]
    dev = m.device
    ang = torch.rand(N, generator=generator, device=dev) * 2 * torch.pi
    rad = (40.0 + severity * 60.0) * scale
    dx = (torch.cos(ang) * rad)
    dy = (torch.sin(ang) * rad)
    return _translate(m, dx, dy)


def corrupt_shape(m, severity, generator=None, scale=1.0):
    """形态错误: 各向异性缩放 或 弹性形变."""
    N, _, H, W = m.shape
    dev = m.device
    out = m.clone()
    for i in range(N):
        s = severity[i].item()
        mi = m[i:i + 1]
        if torch.rand(1, generator=generator, device=dev).item() < 0.5:
            # 各向异性缩放: 一轴放大一轴缩小, 保持面积大致不变
            sx = 1.0 + s * 0.5 * (1 if torch.rand(1, generator=generator, device=dev).item() < 0.5 else -1)
            sy = 1.0 - s * 0.35 * (1 if torch.rand(1, generator=generator, device=dev).item() < 0.5 else -1)
            sx = max(0.55, min(1.6, sx))
            sy = max(0.55, min(1.6, sy))
            out[i:i + 1] = _scale_about_centroid(mi, sx, sy)
        else:
            # 弹性形变
            disp = _smooth_noise(1, H, W, dev, generator, scale=(6.0 + s * 18.0) * scale)
            ys, xs = torch.meshgrid(
                torch.linspace(-1, 1, H, device=dev),
                torch.linspace(-1, 1, W, device=dev), indexing='ij')
            base = torch.stack([xs, ys], dim=0).unsqueeze(0)  # (1,2,H,W) (x,y)
            grid = (base + disp).permute(0, 2, 3, 1)
            out[i:i + 1] = F.grid_sample(mi, grid, mode='nearest', align_corners=False)
    return out


def corrupt_relational(m, severity, context, generator=None, scale=1.0):
    """关系错误: 将目标器官移动到其他器官位置, 破坏器官间空间关系.

    context: (N, C, H, W) 其他器官的 mask (目标器官通道已置零).
    """
    N, _, H, W = m.shape
    dev = m.device
    out = m.clone()
    for i in range(N):
        ctx = context[i]  # (C,H,W)
        areas = ctx.sum(dim=(1, 2))
        if (areas > 20 * scale * scale).sum() == 0:
            # 无其他器官可用, 退化为 location corruption
            s = severity[i:i + 1]
            out[i:i + 1] = corrupt_location(m[i:i + 1], s, generator, scale=scale)
            continue
        probs = (areas / areas.sum())
        c = torch.multinomial(probs + 1e-8, 1, generator=generator).item()
        organ = ctx[c:c + 1]
        pos = organ.sum()
        ys = torch.arange(H, device=dev, dtype=torch.float32).view(H, 1) * organ[0]
        xs = torch.arange(W, device=dev, dtype=torch.float32).view(1, W) * organ[0]
        cy_t = ys.sum() / pos
        cx_t = xs.sum() / pos
        # 目标器官自身质心
        pm = m[i, 0]
        pmass = pm.sum() + 1e-6
        ysm = torch.arange(H, device=dev, dtype=torch.float32).view(H, 1) * pm
        xsm = torch.arange(W, device=dev, dtype=torch.float32).view(1, W) * pm
        cy_m = ysm.sum() / pmass
        cx_m = xsm.sum() / pmass
        jit = severity[i] * 25.0 * scale
        dx = cx_t - cx_m + (torch.rand(1, generator=generator, device=dev).item() * 2 - 1) * jit
        dy = cy_t - cy_m + (torch.rand(1, generator=generator, device=dev).item() * 2 - 1) * jit
        out[i:i + 1] = _translate(m[i:i + 1], dx, dy)
    return out


# ---------------------------------------------------------------- entry

class CounterfactualCorruptor:
    """批量生成反事实解剖错误.

    用法:
        corruptor = CounterfactualCorruptor()
        neg, err, tidx = corruptor.corrupt(masks, context, types=None)
        # masks: (N,1,H,W) GT; context: (N,C,H,W) 其他器官
        # neg: (N,1,H,W) 错误 mask; err: (N,1,H,W) 语义违反区域; tidx: (N,) 类型索引
    """

    def __init__(self, severity_range=(0.2, 0.6), types=None, seed=None,
                 spatial_scale=1.0):
        """spatial_scale = image_size / 256, 像素量级随分辨率缩放."""
        self.severity_range = severity_range
        self.types = types if types is not None else CORRUPTION_TYPES
        self.seed = seed
        self.spatial_scale = spatial_scale

    def corrupt(self, masks, context=None, types=None, generator=None):
        N, _, H, W = masks.shape
        dev = masks.device
        gen = generator
        if gen is None and self.seed is not None:
            gen = torch.Generator(device=dev)
            gen.manual_seed(self.seed)

        types = types if types is not None else self.types
        # 每个样本随机选一种 corruption 类型
        tnames = [types[torch.randint(0, len(types), (1,), generator=gen, device=dev).item()]
                  for _ in range(N)]
        tidx = torch.tensor([TYPE_TO_INDEX[t] for t in tnames], device=dev)

        sev = torch.empty(N, device=dev)
        sev.uniform_(self.severity_range[0], self.severity_range[1],
                     generator=gen) if gen is not None else sev.uniform_(
            self.severity_range[0], self.severity_range[1])

        out = masks.clone()
        sc = self.spatial_scale
        for t in set(tnames):
            sel = [i for i, tn in enumerate(tnames) if tn == t]
            sel_t = torch.tensor(sel, device=dev, dtype=torch.long)
            sub = masks[sel_t]
            sev_sub = sev[sel_t]
            if t == 'boundary':
                res = corrupt_boundary(sub, sev_sub, gen, scale=sc)
            elif t == 'missing':
                res = corrupt_missing(sub, sev_sub, gen, scale=sc)
            elif t == 'false_positive':
                res = corrupt_false_positive(sub, sev_sub, gen, scale=sc)
            elif t == 'location':
                res = corrupt_location(sub, sev_sub, gen, scale=sc)
            elif t == 'shape':
                res = corrupt_shape(sub, sev_sub, gen, scale=sc)
            elif t == 'relational':
                ctx = context[sel_t] if context is not None else None
                if ctx is not None:
                    res = corrupt_relational(sub, sev_sub, ctx, gen, scale=sc)
                else:
                    res = corrupt_location(sub, sev_sub, gen, scale=sc)
            else:
                raise ValueError(f'unknown corruption type: {t}')
            out[sel_t] = res

        err = (out - masks).abs()  # XOR: 语义违反区域
        return out, err, tidx
