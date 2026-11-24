"""SACF 训练: Counterfactual Anatomical Semantic Criterion.

训练流程 (IMISNet 全程冻结, 只训练 SACF 模块):
  1. 冻结 IMIS encoder 提取视觉特征 F (detached)
  2. 每个器官 GT M+ 作为正确语义状态
  3. CounterfactualCorruptor 从 M+ 合成 6 类语义违反状态 M-
  4. SACF(F, M, context, T) -> Contradiction Field C(x) + score
  5. L = w1*L_field + w2*L_rank + w3*L_score

推理阶段只用 (I, T, M_t), GT-free —— GT 仅在训练期用于生成反事实错误.

用法:
  python train_sacf.py --data_dir dataset/BTCV \
      --split_json dataset_split_fixed24_6.json \
      --sam_checkpoint ckpt/IMISNet-B.pth
"""

import argparse
import os
import time
from functools import partial
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.cuda import amp

from segment_anything import sam_model_registry
from model import IMISNet
from dataloaders.data_utils import get_bboxes_from_mask
from sacf import (SemanticContradictionField, CounterfactualCorruptor, SACFLoss,
                  SACFDataset, sacf_collate_fn, field_iou, pixel_auroc,
                  CORRUPTION_TYPES, get_anatomy_prompt, encode_structured_text)
from sacf.corruptions import _erode, _dilate, _translate


def parse_args():
    p = argparse.ArgumentParser()
    # 与基准模型 (ft-IMISNet-full) 保持一致: BTCV_full @ 512, 24/6 病例划分
    p.add_argument('--data_dir', type=str, default='dataset/BTCV_full')
    p.add_argument('--split_json', type=str, default='dataset_split_full_24_6.json')
    p.add_argument('--image_size', type=int, default=512)
    # sam_checkpoint: SAM 结构初始化权重 (build_sam 内部 strict 加载, 必须是
    #   IMISNet-B.pth 的 SAM 键格式); imis_ckpt: ft-IMISNet-full 微调后权重,
    #   覆盖加载到 IMISNet, 保证训练/推理 (GT-free 评估) 特征分布一致.
    p.add_argument('--sam_checkpoint', type=str, default='ckpt/IMISNet-B.pth')
    p.add_argument('--imis_ckpt', type=str,
                   default='work_dir/ft-IMISNet-full/IMIS_val_dice_best.pth',
                   help='冻结 encoder 最终权重, 与 eval_sacf.py --imis_ckpt 一致')
    p.add_argument('--model_type', type=str, default='vit_b')
    p.add_argument('--work_dir', type=str, default='work_dir/sacf')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=8, help='slices per step')
    p.add_argument('--organs_per_slice', type=int, default=2)
    p.add_argument('--neg_per_pos', type=int, default=2, help='每个 GT 的反事实样本数')
    # lr / weight_decay / epochs 与基准 ft-IMISNet-full 配置对齐 (1e-4, 1e-5, 50)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--dim', type=int, default=192)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--corruption_types', type=str, default=','.join(CORRUPTION_TYPES))
    p.add_argument('--margin', type=float, default=0.3)
    p.add_argument('--w_field', type=float, default=1.0)
    p.add_argument('--w_rank', type=float, default=1.0)
    p.add_argument('--w_score', type=float, default=0.5)
    p.add_argument('--pos_weight', type=float, default=20.0)
    p.add_argument('--val_seed', type=int, default=1234)
    p.add_argument('--resume', type=str, default=None)
    # 必改1: 文本语义来源 —— structured=结构化解剖描述 / name=裸器官名(消融)
    p.add_argument('--text_style', type=str, default='structured',
                   choices=['structured', 'name'])
    # 必改3: relational context 训练分布 (GT / Pred / NoisyPred 概率)
    p.add_argument('--p_ctx_gt', type=float, default=0.4)
    p.add_argument('--p_ctx_pred', type=float, default=0.4)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--patience', type=int, default=0,
                   help='早停: val detect_auroc 连续 N 轮无提升则停止; 0 表示不早停')
    return p.parse_args()


def build_imis(args):
    """加载冻结的 IMISNet, 只用其 image encoder 与 text 通路.

    sam_checkpoint: build_sam 内部 strict 加载的 SAM 初始化权重;
    imis_ckpt: ft-IMISNet-full 微调存档 {'model_state_dict': ...} 覆盖加载,
    与 GT-free 评估 (eval_sacf.py --imis_ckpt) 使用同一份基准权重.
    """
    sam = sam_model_registry[args.model_type](args).to(args.device)
    imis = IMISNet(sam, test_mode=True,
                   category_weights='dataloaders/categories_weight.pkl').to(args.device)
    if args.imis_ckpt and os.path.exists(args.imis_ckpt):
        sd = torch.load(args.imis_ckpt, map_location='cpu')
        if isinstance(sd, dict) and 'model_state_dict' in sd:
            sd = sd['model_state_dict']
        missing, unexpected = imis.load_state_dict(sd, strict=False)
        n_loaded = len(sd) - len(unexpected)
        print(f'[IMISNet] finetuned weights loaded from {args.imis_ckpt} '
              f'(loaded={n_loaded}/{len(sd)}, missing={len(missing)}, '
              f'unexpected={len(unexpected)})')
        assert n_loaded > 400, \
            f'encoder 权重几乎未加载 ({n_loaded}/{len(sd)}), 请检查 checkpoint 格式'
    else:
        print(f'[IMISNet] WARNING: imis_ckpt 不存在, 仅使用 {args.sam_checkpoint} 初始化')
    imis.eval()
    for p_ in imis.parameters():
        p_.requires_grad = False
    return imis


@torch.no_grad()
def imis_encode(imis, images):
    """冻结 encoder: (B,3,H,W) -> (B,768,h,w)."""
    return imis.image_encoder(images)


@torch.no_grad()
def imis_text(imis, names, text_style='structured'):
    """SACF 评判文本: structured=结构化解剖描述 (必改1), name=裸器官名 (消融).

    注意: IMISNet decoder 的分割 prompt 始终用器官名 (基准协议),
    仅 SACF 的 Expected Anatomy 语义走本入口.
    """
    if text_style == 'structured':
        prompts = [get_anatomy_prompt(n) for n in names]
        return encode_structured_text(imis, prompts)      # (N,768)
    return imis.process_text_prompt(names)['text_inputs']  # (N,768)


@torch.no_grad()
def imis_predict_all(imis, img_embed, labels, min_area, classes):
    """冻结 IMISNet 预测 batch 内每个 slice 的全部出现器官 (GT bbox prompt).

    返回 (B,C,H,W) 二值预测 context —— 与 GT-free 推理 (eval_sacf_full)
    的 context 分布一致 (必改3: 消除 relational evidence 的 train/inference gap).
    """
    B, C, H, W = labels.shape
    device = labels.device
    pred_full = torch.zeros(B, C, H, W, device=device)

    # 收集所有 (slice, organ) 对
    pairs = [(b, c) for b in range(B) for c in range(C)
             if labels[b, c].sum() >= min_area]
    if not pairs:
        return pred_full
    bboxes = torch.cat([
        get_bboxes_from_mask(labels[b, c:c + 1].unsqueeze(0), offset=0).to(device)
        for b, c in pairs], dim=0)
    names = [classes[c] for _, c in pairs]              # 按名称查 categories_map
    text = imis.process_text_prompt(names)
    prompts = {'bboxes': bboxes}
    prompts.update(text)
    embed_rep = torch.cat([img_embed[b:b + 1] for b, _ in pairs], dim=0)
    with amp.autocast():
        outputs = imis.forward_decoder(embed_rep, prompts)
    preds = (torch.sigmoid(outputs['masks'].float()) > 0.5).float()
    for i, (b, c) in enumerate(pairs):
        pred_full[b, c] = preds[i, 0]
    return pred_full


def make_noisy_context(pred_ctx, organ_ch):
    """对 predicted context 的随机一个其他器官通道施加扰动 (必改3).

    模拟真实推理时 context 本身含错: 漏分割 / 边界侵蚀 / 边界膨胀.
    """
    N, C, H, W = pred_ctx.shape
    out = pred_ctx.clone()
    for i in range(N):
        areas = out[i].sum(dim=(1, 2))
        areas[organ_ch[i]] = 0.0
        cand = (areas > 100).nonzero().flatten()
        if len(cand) == 0:
            continue
        c = cand[torch.randint(len(cand), (1,), device=out.device)].item()
        m = out[i, c:c + 1].unsqueeze(0)                  # (1,1,H,W)
        mode = torch.randint(0, 3, (1,), device=out.device).item()
        if mode == 0:
            m = _erode(m, 3)                              # 欠分割
        elif mode == 1:
            m = _dilate(m, 3)                             # 过分割
        else:
            m = m * 0.0                                   # 漏分割
        out[i, c] = m[0, 0]
    return out


def make_batch(args, imis, batch, corruptor, device, is_val=False):
    """构造 (正样本 + 反事实负样本) 的训练 batch.

    必改1: SACF 文本用结构化解剖描述;
    必改3: context 从 {GT, Pred, NoisyPred} 采样 (val 固定 GT 保证监控稳定).
    """
    images = batch['images'].to(device)
    labels = batch['labels'].to(device)
    masks = batch['masks'].to(device)
    organ_ch = batch['organ_ch'].to(device)
    slice_idx = batch['slice_idx'].to(device)
    classes = batch['classes']

    B, C, H, W = labels.shape
    N = masks.shape[0]

    img_embed = imis_encode(imis, images)                 # (B,768,h,w)

    # ---- context 三态 (必改3) ----
    ctx_gt = labels[slice_idx].clone()                    # GT 其他器官
    ctx_gt[torch.arange(N), organ_ch] = 0.0
    if is_val or (args.p_ctx_gt + args.p_ctx_pred) >= 1.0:
        context = ctx_gt
    else:
        min_area = 16 * (args.image_size / 256.0) ** 2
        pred_full = imis_predict_all(imis, img_embed, labels, min_area, classes)
        ctx_pred = pred_full[slice_idx].clone()
        ctx_pred[torch.arange(N), organ_ch] = 0.0
        ctx_noisy = make_noisy_context(ctx_pred.clone(), organ_ch)
        r = torch.rand(N, device=device)
        context = torch.where(
            (r < args.p_ctx_gt)[:, None, None, None], ctx_gt,
            torch.where((r < args.p_ctx_gt + args.p_ctx_pred)[:, None, None, None],
                        ctx_pred, ctx_noisy))

    # ---- 反事实负样本 ----
    neg_types = args.corruption_types
    if isinstance(neg_types, str):  # 防御: 允许逗号分隔字符串
        neg_types = [t for t in neg_types.split(',') if t]
    if is_val:
        gen = torch.Generator(device=device)
        gen.manual_seed(args.val_seed + int(slice_idx[0].item()))
    else:
        gen = None
    neg_masks = masks.repeat(args.neg_per_pos, 1, 1, 1)
    neg_ctx = context.repeat(args.neg_per_pos, 1, 1, 1)
    neg_m, neg_err, neg_tidx = corruptor.corrupt(neg_masks, neg_ctx,
                                                 types=neg_types, generator=gen)

    # 拼接: 前 N 个为正样本 M+, 后 N*neg 为反事实 M-
    all_masks = torch.cat([masks, neg_m], dim=0)
    all_ctx = torch.cat([context, neg_ctx], dim=0)
    all_err = torch.cat([torch.zeros_like(masks), neg_err], dim=0)
    is_neg = torch.cat([torch.zeros(N, device=device),
                        torch.ones(N * args.neg_per_pos, device=device)])
    group = torch.cat([torch.arange(N), torch.arange(N).repeat(args.neg_per_pos)]).to(device)
    # image embedding 与 text embedding 复制
    all_embed = img_embed[slice_idx.repeat(args.neg_per_pos + 1)]  # (N*(1+K),768,h,w)
    names = [classes[c] for c in organ_ch.tolist()]
    t_emb = imis_text(imis, names, args.text_style)       # (N,768) 必改1
    all_t = t_emb.repeat(args.neg_per_pos + 1, 1)

    return {
        'image_embed': all_embed,
        'mask': all_masks,
        'context': all_ctx,
        'text': all_t,
        'targets': {
            'error': all_err,
            'is_neg': is_neg,
            'group': group,
        },
        'type_idx': torch.cat([torch.full((N,), -1, device=device, dtype=torch.long),
                               neg_tidx]),
    }


def run_val(args, imis, sacf, val_loader, corruptor, criterion, device):
    sacf.eval()
    scores_all, isneg_all = [], []
    ious, aurocs = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch['classes'] = val_loader.dataset.classes
            try:
                data = make_batch(args, imis, batch, corruptor, device, is_val=True)
            except RuntimeError as e:
                print(f'[val] skip batch: {e}')
                continue
            with amp.autocast():
                out = sacf(data['image_embed'], data['mask'],
                           data['context'], data['text'])
            scores_all.append(out['score'].float().cpu())
            isneg_all.append(data['targets']['is_neg'].cpu())
            iou = field_iou(out['fused'].float(), data['targets']['error'], adaptive=True)
            auroc = pixel_auroc(out['fused'].float(), data['targets']['error'])
            if iou is not None:
                ious.append(iou)
            if auroc is not None:
                aurocs.append(auroc)
    sacf.train()

    s = torch.cat(scores_all)
    y = torch.cat(isneg_all)
    if len(s) < 4 or y.unique().numel() < 2:
        det_auroc = float('nan')
    else:
        # 升序 rank: 最低分 rank=1, AUROC 标准公式
        order = s.argsort()
        ranks = torch.empty_like(s, dtype=torch.float64)
        ranks[order] = torch.arange(1, s.numel() + 1, dtype=torch.float64)
        n_pos, n_neg = y.sum().double(), (1 - y).sum().double()
        det_auroc = ((ranks[y > 0.5].sum() - n_pos * (n_pos + 1) / 2)
                     / (n_pos * n_neg)).item()
    return {
        'detect_auroc': det_auroc,
        'field_iou': float(np.mean(ious)) if ious else float('nan'),
        'pixel_auroc': float(np.mean(aurocs)) if aurocs else float('nan'),
    }


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.work_dir, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)

    args.corruption_types = [t for t in args.corruption_types.split(',') if t]

    # ---- data ----
    min_area = int(16 * (args.image_size / 256.0) ** 2)
    train_ds = SACFDataset(args.data_dir, args.split_json, 'training', args.image_size,
                           min_organ_area=min_area)
    val_ds = SACFDataset(args.data_dir, args.split_json, 'validation', args.image_size,
                         min_organ_area=min_area)
    collate = partial(sacf_collate_fn, organs_per_slice=args.organs_per_slice)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate,
                            pin_memory=True)
    print(f'[data] train={len(train_ds)} slices, val={len(val_ds)} slices, '
          f'organs={train_ds.num_organs}')

    # ---- models ----
    imis = build_imis(args)
    sacf = SemanticContradictionField(num_organs=train_ds.num_organs,
                                      image_embed_dim=768, text_embed_dim=768,
                                      dim=args.dim, image_size=args.image_size).to(device)
    if args.resume and os.path.exists(args.resume):
        sacf.load_state_dict(torch.load(args.resume, map_location='cpu'))
        print(f'[sacf] resumed from {args.resume}')

    corruptor = CounterfactualCorruptor(types=args.corruption_types,
                                        spatial_scale=args.image_size / 256.0)
    criterion = SACFLoss(w_field=args.w_field, w_rank=args.w_rank,
                         w_score=args.w_score, margin=args.margin,
                         pos_weight=args.pos_weight).to(device)
    optimizer = torch.optim.AdamW(sacf.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = amp.GradScaler()

    best = {'detect_auroc': -1.0}
    bad = 0
    log_path = os.path.join(args.work_dir, 'train_sacf.log')
    def log(msg):
        line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
        print(line)
        with open(log_path, 'a') as f:
            f.write(line + '\n')

    log(f'args: {vars(args)}')
    log(f'sacf params: {sum(p.numel() for p in sacf.parameters()) / 1e6:.2f}M')

    # ---- train ----
    for epoch in range(args.epochs):
        sacf.train()
        ep_loss, n_steps = 0.0, 0
        parts_acc = {'field': 0.0, 'rank': 0.0, 'score': 0.0}
        for it, batch in enumerate(train_loader):
            batch['classes'] = train_ds.classes
            try:
                data = make_batch(args, imis, batch, corruptor, device)
            except RuntimeError as e:
                print(f'[train] skip batch: {e}')
                continue
            with amp.autocast():
                out = sacf(data['image_embed'], data['mask'],
                           data['context'], data['text'])
                loss, parts = criterion(out, data['targets'])
            if torch.isnan(loss):
                continue
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            ep_loss += loss.item()
            n_steps += 1
            for k in parts_acc:
                parts_acc[k] += parts[k].item()
            if it % 50 == 0:
                log(f'ep {epoch} it {it}/{len(train_loader)} '
                    f'loss={loss.item():.4f} '
                    + ' '.join(f'{k}={parts[k].item():.3f}' for k in parts))
        scheduler.step()

        if n_steps == 0:
            log(f'epoch {epoch}: no valid steps!')
            continue

        metrics = run_val(args, imis, sacf, val_loader, corruptor, criterion, device)
        log(f'epoch {epoch}: loss={ep_loss / n_steps:.4f} '
            f'({", ".join(f"{k}={v / n_steps:.3f}" for k, v in parts_acc.items())}) | '
            f'VAL detect_auroc={metrics["detect_auroc"]:.4f} '
            f'field_iou={metrics["field_iou"]:.4f} pixel_auroc={metrics["pixel_auroc"]:.4f}')

        ckpt = {
            'sacf_state_dict': sacf.state_dict(),
            'epoch': epoch,
            'metrics': metrics,
            'args': vars(args),
        }
        torch.save(ckpt, os.path.join(args.work_dir, 'sacf_latest.pth'))
        if metrics['detect_auroc'] >= best['detect_auroc']:
            best = metrics
            bad = 0
            torch.save(ckpt, os.path.join(args.work_dir, 'sacf_best.pth'))
            log(f'  -> saved sacf_best.pth (detect_auroc={metrics["detect_auroc"]:.4f})')
        else:
            bad += 1
            if args.patience and bad >= args.patience:
                log(f'  -> early stop: detect_auroc 连续 {bad} 轮无提升 '
                    f'(best={best["detect_auroc"]:.4f})')
                break

    log(f'BEST: {best}')


if __name__ == '__main__':
    main()
