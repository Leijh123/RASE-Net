"""EGRC 积木四: 训练 Exemplar-Guided Residual Corrector (创新点二).

流程:
  1. 预加载 cache (work_dir/egrc/cache/{training,validation}) 到 RAM
  2. exemplar bank 检索表: score = cos(emb) + layout_sim, 训练时排除同 case
  3. 每个训练样本 = (slice, organ):
       query  = img_emb(在线 encoder) + refined pred + SACF field
       exemplar = 检索 top-K 同器官训练 GT mask, 以 pred 质心/面积做仿射对齐
       final = sigmoid(logit(refined) + residual)
       loss  = softDice + BCE (+ 微弱 residual L2 稳定直通初始化)
  4. 每 epoch 验证: val organ-samples 上 Base / SCAR / EGRC Dice (同一协议),
     保存最优 checkpoint, 记录 organ gate 行为

用法:
  cd ./ && python scar/train/train_egrc.py
"""

import sys as _sys, os as _os
_ROOT = _os.path.abspath(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
_SACF_DEV = _os.path.join(_ROOT, 'sacf_dev')
if _SACF_DEV not in _sys.path:
    _sys.path.insert(0, _SACF_DEV)

import argparse
import json
import math
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize

from segment_anything import sam_model_registry
from model import IMISNet

from egrc import EGRCCorrector, logit, soft_dice_loss  # noqa: E402

PIXEL_MEAN = (123.675, 116.28, 103.53)
PIXEL_STD = (58.395, 57.12, 57.375)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, default='dataset/BTCV_full')
    p.add_argument('--split_json', type=str, default='dataset_split_full_24_6.json')
    p.add_argument('--image_size', type=int, default=512)
    p.add_argument('--sam_checkpoint', type=str, default='ckpt/IMISNet-B.pth')
    p.add_argument('--imis_ckpt', type=str,
                   default='work_dir/ft-IMISNet-full/IMIS_val_dice_best.pth')
    p.add_argument('--model_type', type=str, default='vit_b')
    p.add_argument('--cache_dir', type=str, default='work_dir/egrc/cache')
    p.add_argument('--bank_path', type=str, default='work_dir/egrc/exemplar_bank.pt')
    p.add_argument('--out_dir', type=str, default='work_dir/egrc')
    p.add_argument('--K', type=int, default=4)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch', type=int, default=12)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--res_l2', type=float, default=1e-3)
    p.add_argument('--val_batch_slices', type=int, default=8)
    p.add_argument('--no_scar', action='store_true',
                   help='消融: EGRC 直接修正 base pred (field 置零), 不经过 SCAR')
    p.add_argument('--alpha', type=float, default=0.5,
                   help='对齐锚点融合系数: pred 质心 vs 检索先验质心')
    p.add_argument('--prior_anchor', action='store_true',
                   help='启用检索先验锚点 (默认关闭: check_anchor 验证 pred 质心锚点最优)')
    p.add_argument('--fewshot', action='store_true',
                   help='方案2: 只在少样本器官上训练/修正, 其余器官直通不回退, '
                        'checkpoint 按少样本子集 mean Dice 选择')
    p.add_argument('--resume', type=str, default='',
                   help='从 checkpoint 恢复模型权重, 续训 --epochs 个 epoch')
    p.add_argument('--start_epoch', type=int, default=1,
                   help='续训起始 epoch 编号 (仅影响日志)')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--patience', type=int, default=0,
                   help='早停: val Dice 连续 N 轮无提升则停止; 0 表示不早停')
    return p.parse_args()


# AMOS 少样本/稀少器官子集 (平均面积最小, 排除 bladder/prostate_uterus 因 test 切片为 0)
FEWSHOT_NAMES = {'adrenal_gland_left','adrenal_gland_right','esophagus','duodenum','gallbladder','pancreas'}


# ----------------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------------

def load_image(data_dir, item, size):
    img = Image.open(os.path.join(data_dir, item['image'])).convert('RGB')
    img = np.array(resize(img, (size, size), interpolation=InterpolationMode.NEAREST))
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    for c in range(3):
        img[c] = (img[c] - PIXEL_MEAN[c]) / PIXEL_STD[c]
    return img


def centroid(mask_bool):
    ys, xs = torch.nonzero(mask_bool, as_tuple=True)
    return ys.float().mean(), xs.float().mean()


def align_exemplar(ex, pred, H=256, anchor=None):
    """v2: 只做质心平移对齐 (无缩放/旋转) — 离线验证证明仿射搜索引入大误差.

    平移目标: anchor (若提供) 否则 pred 质心.
    """
    eb = ex > 0.5
    if eb.sum() < 4:
        return ex * 0.0
    if anchor is not None:
        py, px = anchor[0], anchor[1]
    else:
        pb = pred > 0.5
        if pb.sum() < 4:
            pb = pred >= pred.max() * 0.5
            if pb.sum() < 1:
                return ex
        py, px = centroid(pb)
    ey, exx = centroid(eb)
    ty = (1 - 2 * py / H) + 2 * ey / H - 1
    tx = (1 - 2 * px / H) + 2 * exx / H - 1
    theta = torch.tensor([[[1.0, 0.0, tx], [0.0, 1.0, ty]]], dtype=torch.float32)
    grid = F.affine_grid(theta, (1, 1, H, H), align_corners=False)
    out = F.grid_sample(ex[None, None], grid, mode='bilinear',
                        padding_mode='zeros', align_corners=False)
    return out[0, 0]


def exemplar_anchor(ex_infos, pred, alpha=0.5, H=256):
    """检索先验锚点: K 个 exemplar 的 GT 质心/面积均值, 与 pred 质心/面积按 alpha 融合.

    ex_infos: [(cy, cx, area), ...] 每个 exemplar 在其原图中的归一化质心/面积 (GT-free, 来自 bank).
    pred 质心不可靠时 (面积过小), 完全用先验.
    """
    pb = pred > 0.5
    if pb.sum() >= 4:
        py, px = centroid(pb)
        parea = float(pb.sum())
    else:
        alpha = 0.0                      # pred 无效 → 全先验
        py = px = parea = 0.0
    if not ex_infos:
        return (float(py), float(px), parea) if alpha > 0 else None
    ecy = sum(c for c, _, _ in ex_infos) / len(ex_infos) * H
    ecx = sum(x for _, x, _ in ex_infos) / len(ex_infos) * H
    earea = sum(a for _, _, a in ex_infos) / len(ex_infos) * H * H  # 归一化面积→像素
    cy = alpha * py + (1 - alpha) * ecy
    cx = alpha * px + (1 - alpha) * ecx
    area = alpha * parea + (1 - alpha) * earea
    return (float(cy), float(cx), float(area))


def dice_np(pred_bool, gt_bool):
    p = pred_bool.astype(np.float64).sum()
    t = gt_bool.astype(np.float64).sum()
    i = (pred_bool & gt_bool).astype(np.float64).sum()
    if p + t == 0:
        return 1.0
    return 2 * i / (p + t)


def load_cache(split_dir):
    """返回 {slice: {'organs': list, 'pos': {organ: pos}, 'base','field','refined','gt'}}"""
    import glob
    cache = {}
    files = sorted(glob.glob(os.path.join(split_dir, '*.pt')))
    for f in files:
        idx = int(os.path.basename(f).split('.')[0])
        d = torch.load(f, map_location='cpu')
        organs = d['organs'].tolist()
        cache[idx] = {
            'organs': organs,
            'pos': {o: i for i, o in enumerate(organs)},
            'base': d['base'], 'field': d['field'],
            'refined': d['refined'], 'gt': d['gt'],
        }
    return cache


def build_bank_prior(bank):
    """(slice, organ) -> (cy_norm, cx_norm, area_frac), 来自训练集 bank (GT-free 推理可用)."""
    prior = {}
    info = bank['info']
    S = 512  # bank 构建时的 image_size
    for i, s in enumerate(info):
        for c, (cy, cx) in s['centroids'].items():
            prior[(i, c)] = (cy, cx, float(s['areas'][c]) / (S * S))
    return prior


def _shift_mask(m, dy, dx):
    """整数像素平移 (边界补零), 训练抖动用."""
    H = m.shape[0]
    out = torch.zeros_like(m)
    ys0, ys1 = max(0, dy), min(H, H + dy)
    xs0, xs1 = max(0, dx), min(H, H + dx)
    if ys1 > ys0 and xs1 > xs0:
        out[ys0:ys1, xs0:xs1] = m[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
    return out


# ----------------------------------------------------------------------------
# 数据集
# ----------------------------------------------------------------------------

class EGRCTrainSet(Dataset):
    """items = (slice_idx, organ). exemplar 对齐在 __getitem__ 完成 (CPU)."""

    def __init__(self, items, retrieval, cache_tr, datalist, data_dir, image_size, K,
                 no_scar=False, bank_prior=None, alpha=0.5, jitter=False):
        self.items = items
        self.retrieval = retrieval
        self.cache_tr = cache_tr
        self.datalist = datalist
        self.data_dir = data_dir
        self.image_size = image_size
        self.K = K
        self.no_scar = no_scar
        self.bank_prior = bank_prior
        self.alpha = alpha
        self.jitter = jitter

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        slice_idx, organ = self.items[i]
        d = self.cache_tr[slice_idx]
        pos = d['pos'][organ]
        if self.no_scar:                       # 消融: 基于 base pred, 无 field
            pred = d['base'][pos].float()
            field = torch.zeros_like(pred)
        else:
            pred = d['refined'][pos].float()
            field = d['field'][pos].float()
        gt = d['gt'][pos].float()
        exs = self.retrieval[(slice_idx, organ)]
        ex_infos = [self.bank_prior[(es, organ)] for es in exs] \
            if self.bank_prior else []
        anchor = exemplar_anchor(ex_infos, pred, alpha=self.alpha) \
            if self.bank_prior else None
        masks = [align_exemplar(self.cache_tr[es]['gt'][
            self.cache_tr[es]['pos'][organ]].float(), pred, anchor=anchor)
            for es in exs]
        if self.jitter:                       # 训练增强: 随机平移 ±6px 防记忆精确对齐
            masks = [_shift_mask(m, random.randint(-6, 6),
                                 random.randint(-6, 6)) for m in masks]
        while len(masks) < self.K:
            masks.append(torch.zeros(256, 256))
        ex = torch.stack(masks[:self.K])                       # (K,256,256)
        img = load_image(self.data_dir, self.datalist[slice_idx], self.image_size)
        return {'image': img, 'pred': pred, 'field': field, 'gt': gt,
                'ex': ex, 'organ': organ, 'slice': slice_idx}


def _collate(batch):
    return {
        'image': torch.stack([b['image'] for b in batch]),
        'pred': torch.stack([b['pred'] for b in batch])[:, None],
        'field': torch.stack([b['field'] for b in batch])[:, None],
        'gt': torch.stack([b['gt'] for b in batch])[:, None],
        'ex': torch.stack([b['ex'] for b in batch])[:, :, None],  # (B,K,1,H,W)
        'organ': torch.tensor([b['organ'] for b in batch], dtype=torch.long),
        'slice': torch.tensor([b['slice'] for b in batch], dtype=torch.long),
    }


# ----------------------------------------------------------------------------
# 检索
# ----------------------------------------------------------------------------

def retrieve_topk(q_emb, q_lay, cands, emb_tr, lay_tr, K):
    if not cands:
        return []
    ci = torch.tensor(cands, dtype=torch.long)
    sims = (emb_tr[ci] @ q_emb).numpy()
    lay_d = np.linalg.norm(lay_tr[ci.numpy()] - q_lay[None], axis=1)
    score = sims + np.exp(-lay_d / (lay_d.mean() + 1e-6))
    order = np.argsort(-score)[:K]
    return [cands[i] for i in order]


def build_train_retrieval(items, bank, emb_tr, lay_tr, case_of, K):
    organ_slices = defaultdict(list)
    for e in bank['entries']:
        organ_slices[e['organ']].append(e['slice'])
    info = bank['info']
    retrieval = {}
    for (qs, c) in items:
        cands = [s for s in organ_slices[c] if case_of[s] != case_of[qs]]
        if not cands:
            cands = organ_slices[c]
        retrieval[(qs, c)] = retrieve_topk(info[qs]['emb'], info[qs]['layout'],
                                           cands, emb_tr, lay_tr, K)
    return retrieval, organ_slices


# ----------------------------------------------------------------------------
# 训练 & 验证
# ----------------------------------------------------------------------------

def encode_slices(imis, images, device):
    """images (U,3,512,512) → emb (U,768,64,64) fp32."""
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        return imis.image_encoder(images).float()


def run_val(model, imis, args, cache_va, cache_tr, datalist_va, organ_slices,
            emb_tr, lay_tr, device, log, bank_prior=None, focus_ids=None):
    """val: 每 slice 编码 → organ-sample 检索+对齐+前向; 返回 per-organ 指标.
    no_scar 模式: EGRC 修正 base pred (field 置零), 'egrc' 列 = Base+EGRC.
    focus_ids (方案2): 只修正少样本器官, 其余器官 EGRC=Base 直通 (保证不回退),
    返回值 me = 少样本子集 mean Dice (供 checkpoint 选择)."""
    model.eval()
    per = defaultdict(lambda: {'base': [], 'scar': [], 'egrc': []})
    sls = sorted(cache_va.keys())
    for bi0 in range(0, len(sls), args.val_batch_slices):
        sl_batch = sls[bi0:bi0 + args.val_batch_slices]
        imgs = torch.stack([load_image(args.data_dir, datalist_va[s], args.image_size)
                            for s in sl_batch]).to(device)
        emb = encode_slices(imis, imgs, device)
        pooled = emb.mean(dim=(2, 3))
        pooled = pooled / pooled.norm(dim=1, keepdim=True)

        samples = []
        for ui, s in enumerate(sl_batch):
            d = cache_va[s]
            for organ in d['organs']:
                if focus_ids is not None and organ not in focus_ids:
                    # 非少样本器官: 不修正, 直通 (Base+EGRC 模式取 Base, SCAR+EGRC 取 SCAR)
                    pos = d['pos'][organ]
                    gt_np = d['gt'][pos].float().numpy() > 0.5
                    db = dice_np(d['base'][pos].float().numpy() > 0.5, gt_np)
                    ds = dice_np(d['refined'][pos].float().numpy() > 0.5, gt_np)
                    per[organ]['base'].append(db)
                    per[organ]['scar'].append(ds)
                    per[organ]['egrc'].append(db if args.no_scar else ds)
                else:
                    samples.append((ui, s, organ))
        if not samples:
            continue
        # 检索 + 对齐 (CPU)
        ex_list, pred_list, field_list, gt_list, organs = [], [], [], [], []
        for (ui, s, organ) in samples:
            d = cache_va[s]
            pos = d['pos'][organ]
            if args.no_scar:
                pred = d['base'][pos].float()
                field = torch.zeros_like(pred)
            else:
                pred = d['refined'][pos].float()
                field = d['field'][pos].float()
            q_lay = np.zeros(lay_tr.shape[1], dtype=np.float32)
            for og in d['organs']:
                q_lay[og] = float(d['gt'][d['pos'][og]].float().mean())
            exs = retrieve_topk(pooled[ui].cpu(), q_lay, organ_slices.get(organ, []),
                                emb_tr, lay_tr, args.K)
            ex_infos = [bank_prior[(es, organ)] for es in exs] \
                if bank_prior else []
            anchor = exemplar_anchor(ex_infos, pred, alpha=args.alpha) \
                if bank_prior else None
            masks = [align_exemplar(
                cache_tr[es]['gt'][cache_tr[es]['pos'][organ]].float(),
                pred, anchor=anchor) for es in exs]
            while len(masks) < args.K:
                masks.append(torch.zeros(256, 256))
            ex_list.append(torch.stack(masks[:args.K]))
            pred_list.append(pred)
            field_list.append(d['field'][pos].float())
            gt_list.append(d['gt'][pos].float())
            organs.append(organ)

        # 分批前向
        for i0 in range(0, len(samples), 16):
            i1 = min(i0 + 16, len(samples))
            sub = list(range(i0, i1))
            uq = sorted({samples[i][0] for i in sub})
            emb_sub = emb[[uq.index(samples[i][0]) for i in sub]]
            pred_b = torch.stack([pred_list[i] for i in sub])[:, None].to(device)
            field_b = torch.stack([field_list[i] for i in sub])[:, None].to(device)
            ex_b = torch.stack([ex_list[i] for i in sub])[:, :, None].to(device)
            organ_b = torch.tensor([organs[i] for i in sub], dtype=torch.long, device=device)
            with torch.autocast('cuda', dtype=torch.float16):
                residual = model(emb_sub, pred_b, field_b, ex_b, organ_b)
            final = torch.sigmoid(logit(pred_b) + residual.float())
            for j, i in enumerate(sub):
                organ = organs[i]
                gt_np = gt_list[i].numpy() > 0.5
                d = cache_va[samples[i][1]]
                pos = d['pos'][organ]
                per[organ]['base'].append(dice_np(
                    d['base'][pos].float().numpy() > 0.5, gt_np))
                per[organ]['scar'].append(dice_np(
                    d['refined'][pos].float().numpy() > 0.5, gt_np))
                per[organ]['egrc'].append(dice_np(
                    final[j, 0].detach().cpu().numpy() > 0.5, gt_np))

    classes = json.load(open(os.path.join(args.data_dir, args.split_json)))['labels']
    rows = []
    allb, alls, alle = [], [], []
    for c in sorted(per):
        r = per[c]
        nb, ns, ne = (np.mean(r['base']), np.mean(r['scar']), np.mean(r['egrc']))
        rows.append((classes[str(c + 1)], len(r['base']), nb, ns, ne))
        allb += r['base']; alls += r['scar']; alle += r['egrc']
    log(f'{"organ":<32s} {"n":>5s} {"Base":>7s} {"SCAR":>7s} {"EGRC":>7s} {"Δ":>7s}')
    for name, n, nb, ns, ne in sorted(rows, key=lambda r: r[3]):
        log(f'{name:<32s} {n:>5d} {nb:>7.3f} {ns:>7.3f} {ne:>7.3f} {ne - ns:>+7.3f}')
    mb, ms, me = np.mean(allb), np.mean(alls), np.mean(alle)
    log(f'{"MEAN":<32s} {len(allb):>5d} {mb:>7.3f} {ms:>7.3f} {me:>7.3f} {me - ms:>+7.3f}')
    if focus_ids is not None:
        fb, fs_, fe = [], [], []
        for c in sorted(per):
            if c in focus_ids:
                fb += per[c]['base']; fs_ += per[c]['scar']; fe += per[c]['egrc']
        mfb, mfs, mfe = np.mean(fb), np.mean(fs_), np.mean(fe)
        log(f'{"FEWSHOT-MEAN":<32s} {len(fe):>5d} {mfb:>7.3f} {mfs:>7.3f} '
            f'{mfe:>7.3f} {mfe - mfb:>+7.3f}   (Δ vs Base)')
        return float(mfe), float(mfs), rows
    return me, ms, rows


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(os.path.join(args.out_dir, 'ckpt'), exist_ok=True)
    logf = open(os.path.join(args.out_dir, 'train_log.txt'), 'a')

    def log(s):
        print(s, flush=True)
        logf.write(s + '\n')
        logf.flush()

    log('=' * 80)
    log(f'[start] EGRC training  K={args.K} epochs={args.epochs} batch={args.batch} '
        f'no_scar={args.no_scar} fewshot={args.fewshot} prior_anchor={args.prior_anchor}')

    # ---- bank ----
    bank = torch.load(args.bank_path, map_location='cpu', weights_only=False)
    emb_tr = torch.stack([s['emb'] for s in bank['info']])
    lay_tr = np.stack([s['layout'] for s in bank['info']])
    bank_prior = build_bank_prior(bank) if args.prior_anchor else None
    log(f'[bank] {len(bank["entries"])} entries, {len(bank["info"])} slices, '
        f'prior_anchor={"on" if bank_prior else "off (pred centroid)"}')

    # ---- split / case ----
    ds = json.load(open(os.path.join(args.data_dir, args.split_json)))
    datalist_tr, datalist_va = ds['training'], ds['validation']
    case_of_tr = ['_'.join(it['image'].split('/')[1].split('_')[:2]) for it in datalist_tr]

    # ---- cache ----
    log('[cache] loading train...')
    cache_tr = load_cache(os.path.join(args.cache_dir, 'training'))
    log(f'[cache] train {len(cache_tr)} slices')
    log('[cache] loading val...')
    cache_va = load_cache(os.path.join(args.cache_dir, 'validation'))
    log(f'[cache] val {len(cache_va)} slices')

    # ---- train items ----
    items = [(s, o) for s in sorted(cache_tr) for o in cache_tr[s]['organs']]
    focus_ids = None
    if args.fewshot:
        classes = ds['labels']
        focus_ids = {int(k) - 1 for k, name in classes.items() if name in FEWSHOT_NAMES}
        assert focus_ids, 'fewshot organ names not found in split labels'
        items = [(s, o) for (s, o) in items if o in focus_ids]
        log(f'[fewshot] focus organs: '
            f'{sorted(classes[str(c + 1)] for c in focus_ids)}')
    log(f'[items] {len(items)} train organ-samples')
    retrieval, organ_slices = build_train_retrieval(items, bank, emb_tr, lay_tr,
                                                    case_of_tr, args.K)
    log(f'[retrieval] built (case-excluded), organs={len(organ_slices)}')

    # ---- models ----
    sam = sam_model_registry[args.model_type](args).to(device)
    imis = IMISNet(sam, test_mode=True,
                   category_weights='dataloaders/categories_weight.pkl').to(device)
    sd = torch.load(args.imis_ckpt, map_location='cpu')
    if isinstance(sd, dict) and 'model_state_dict' in sd:
        sd = sd['model_state_dict']
    imis.load_state_dict(sd, strict=False)
    imis.eval()
    for p in imis.parameters():
        p.requires_grad_(False)
    log(f'[IMISNet] frozen, loaded {args.imis_ckpt}')

    n_organs = len(ds['labels']) - 1   # 由 split labels 推出 (ACDC=3)
    model = EGRCCorrector(d=128, n_organs=n_organs, n_heads=4).to(device)
    log(f'[EGRC] n_organs={n_organs}')
    nparam = sum(p.numel() for p in model.parameters())
    log(f'[EGRC] {nparam / 1e6:.2f}M params')

    best = -1.0
    if args.resume:
        ck = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ck['model_state_dict'])
        best = float(ck['val_dice'])
        log(f'[resume] loaded {args.resume} (epoch {ck.get("epoch", "?")}, '
            f'val_dice {best:.4f}); best 初始化为该值, 更差不会覆盖保存')

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * math.ceil(len(items) / args.batch))

    trainset = EGRCTrainSet(items, retrieval, cache_tr, datalist_tr,
                            args.data_dir, args.image_size, args.K,
                            no_scar=args.no_scar, bank_prior=bank_prior,
                            alpha=args.alpha)
    loader = DataLoader(trainset, batch_size=args.batch, shuffle=True,
                        num_workers=args.num_workers, collate_fn=_collate,
                        pin_memory=True, drop_last=True, persistent_workers=True)

    bad = 0
    for ep in range(args.start_epoch, args.start_epoch + args.epochs):
        model.train()
        tot, nb = 0.0, 0
        res_mag = 0.0
        for it, batch in enumerate(loader):
            organs = batch['organ']
            sls = batch['slice'].tolist()
            # encoder 只对 unique slice 前向 (同 slice 多器官共享 emb)
            seen, first_pos = {}, {}
            for i, s in enumerate(sls):
                if s not in seen:
                    seen[s] = len(seen)
                    first_pos[s] = i
            pos_t = torch.tensor([first_pos[s] for s in seen])
            img_uq = batch['image'][pos_t].to(device)
            emb_uq = encode_slices(imis, img_uq, device)
            emb = emb_uq[torch.tensor([seen[s] for s in sls], device=device)]

            pred = batch['pred'].to(device)
            field = batch['field'].to(device)
            gt = batch['gt'].to(device)
            ex = batch['ex'].to(device)
            with torch.autocast('cuda', dtype=torch.float16):
                residual = model(emb, pred, field, ex, organs.to(device))
            final = torch.sigmoid(logit(pred) + residual.float())
            loss = soft_dice_loss(final, gt) + F.binary_cross_entropy(final, gt) \
                + args.res_l2 * residual.float().pow(2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss)
            res_mag += float(residual.float().abs().mean())
            nb += 1
            if it % 200 == 0:
                log(f'  [ep{ep} it{it}/{len(loader)}] loss={tot / nb:.4f} '
                    f'|res|={res_mag / nb:.3f}')

        gate = torch.sigmoid(model.organ_gate.weight).squeeze().detach().cpu().tolist()
        log(f'[ep{ep}] train loss={tot / max(nb, 1):.4f} |res|={res_mag / max(nb, 1):.3f}')
        log(f'[ep{ep}] organ_gate: ' + ' '.join(f'{g:.2f}' for g in gate))

        log(f'[ep{ep}] validating...')
        me, ms, _ = run_val(model, imis, args, cache_va, cache_tr, datalist_va,
                            organ_slices, emb_tr, lay_tr, device, log,
                            bank_prior=bank_prior, focus_ids=focus_ids)
        if me > best:
            best = me
            bad = 0
            ck_name = 'egrc_best_noscar.pth' if args.no_scar else 'egrc_best.pth'
            torch.save({'model_state_dict': model.state_dict(), 'val_dice': me,
                       'val_scar': ms, 'epoch': ep, 'no_scar': args.no_scar},
                       os.path.join(args.out_dir, 'ckpt', ck_name))
            log(f'[ep{ep}] saved best ({"FEWSHOT" if focus_ids else "ALL"} '
                f'EGRC {me:.4f})')
        else:
            bad += 1
            if args.patience and bad >= args.patience:
                log(f'[ep{ep}] early stop: val Dice 连续 {bad} 轮无提升 '
                    f'(best={best:.4f})')
                break
    log(f'[done] best {"Base+EGRC" if args.no_scar else "SCAR+EGRC"} '
        f'val Dice = {best:.4f}')


if __name__ == '__main__':
    main()
