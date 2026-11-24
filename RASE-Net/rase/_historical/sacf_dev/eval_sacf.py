"""SACF 评估.

模式 1  --mode corruption  : 机制实验 —— 在 val GT 上生成受控反事实错误,
    验证 (a) score 的 error detection AUROC (逐 corruption 类型),
          (b) 四类 evidence 对不同语义违反类型的差异化响应 (机制可解释性).

模式 2  --mode prediction : GT-free 真实预测评估 —— IMISNet 逐器官推理
    (text+bbox 交互协议), SACF 只见 (I, T, M_t, 其他器官预测) 输出 C(x),
    与真实误差 (M_t vs GT, 仅用于计算指标) 比较:
        pixel AUROC / Error Localization IoU / slice-level detection AUROC.
"""

import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.cuda import amp

from segment_anything import sam_model_registry
from model import IMISNet
from dataloaders.data_utils import get_bboxes_from_mask
from sacf import (SemanticContradictionField, CounterfactualCorruptor, CORRUPTION_TYPES,
                  SACFDataset, sacf_collate_fn, field_iou, field_iou_oracle,
                  pixel_auroc, EVIDENCE_NAMES, get_anatomy_prompt,
                  encode_structured_text)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', type=str, default='corruption',
                   choices=['corruption', 'prediction'])
    # 与基准模型 (ft-IMISNet-full) 对齐: BTCV_full @ 512, 24/6 病例划分
    p.add_argument('--data_dir', type=str, default='dataset/BTCV_full')
    p.add_argument('--split_json', type=str, default='dataset_split_full_24_6.json')
    p.add_argument('--image_size', type=int, default=512)
    p.add_argument('--sam_checkpoint', type=str, default='ckpt/IMISNet-B.pth')
    p.add_argument('--imis_ckpt', type=str, default='work_dir/ft-IMISNet-full/IMIS_val_dice_best.pth',
                   help='mode=prediction 时加载的微调权重')
    p.add_argument('--sacf_ckpt', type=str, default='work_dir/sacf/sacf_best.pth')
    p.add_argument('--model_type', type=str, default='vit_b')
    p.add_argument('--dim', type=int, default=192)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--organs_per_slice', type=int, default=2)
    p.add_argument('--neg_per_pos', type=int, default=2)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--out', type=str, default='work_dir/sacf/eval_results.txt')
    return p.parse_args()


def load_models(args):
    device = torch.device(args.device)
    sam = sam_model_registry[args.model_type](args).to(device)
    imis = IMISNet(sam, test_mode=True,
                   category_weights='dataloaders/categories_weight.pkl').to(device)
    if args.imis_ckpt and os.path.exists(args.imis_ckpt):
        sd = torch.load(args.imis_ckpt, map_location='cpu')
        if 'model_state_dict' in sd:
            sd = sd['model_state_dict']
        imis.load_state_dict(sd, strict=False)
        print(f'[IMISNet] loaded {args.imis_ckpt}')
    imis.eval()
    for p_ in imis.parameters():
        p_.requires_grad = False

    ckpt = torch.load(args.sacf_ckpt, map_location='cpu')
    sacf = SemanticContradictionField(num_organs=13, image_embed_dim=768,
                                      text_embed_dim=768, dim=args.dim,
                                      image_size=args.image_size).to(device)
    sacf.load_state_dict(ckpt['sacf_state_dict'])
    sacf.eval()
    print(f'[SACF] loaded {args.sacf_ckpt} (epoch {ckpt.get("epoch", "?")}, '
          f'metrics={ckpt.get("metrics", {})})')
    return imis, sacf, device


def rank_auroc(scores, labels):
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels)
    if len(np.unique(y)) < 2:
        return float('nan')
    # 升序 rank: 最低分 rank=1
    order = np.argsort(s)
    ranks = np.empty_like(s)
    ranks[order] = np.arange(1, len(s) + 1)
    n_pos, n_neg = y.sum(), (1 - y).sum()
    return (ranks[y > 0].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


# ---------------------------------------------------------------- mode 1

@torch.no_grad()
def eval_corruption(args, imis, sacf, device):
    """机制实验: 受控反事实错误上的 detection + evidence 分解."""
    min_area = int(16 * (args.image_size / 256.0) ** 2)
    ds = SACFDataset(args.data_dir, args.split_json, 'validation', args.image_size,
                     min_organ_area=min_area)
    from functools import partial
    collate = partial(sacf_collate_fn, organs_per_slice=args.organs_per_slice)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate, pin_memory=True)

    per_type_scores = {t: ([], []) for t in CORRUPTION_TYPES}   # (scores, labels)
    per_type_iou = {t: [] for t in CORRUPTION_TYPES}
    per_type_iou_oracle = {t: [] for t in CORRUPTION_TYPES}
    # evidence 分解: 每种 corruption 下 4 类 evidence 在错误区域内的均值
    evi_resp = {t: np.zeros(len(EVIDENCE_NAMES)) for t in CORRUPTION_TYPES}
    evi_cnt = {t: 0 for t in CORRUPTION_TYPES}

    for batch in loader:
        images = batch['images'].to(device)
        labels = batch['labels'].to(device)
        masks = batch['masks'].to(device)
        organ_ch = batch['organ_ch'].to(device)
        slice_idx = batch['slice_idx'].to(device)
        N = masks.shape[0]

        context = labels[slice_idx].clone()
        context[torch.arange(N), organ_ch] = 0.0

        img_embed = imis.image_encoder(images)[slice_idx]
        names = [ds.classes[c] for c in organ_ch.tolist()]
        # SACF 评判文本: 结构化解剖描述 (必改1, 与训练一致)
        t_emb = encode_structured_text(
            imis, [get_anatomy_prompt(n) for n in names])

        for t in CORRUPTION_TYPES:
            corruptor = CounterfactualCorruptor(types=[t], seed=100,
                                                spatial_scale=args.image_size / 256.0)
            neg, err, _ = corruptor.corrupt(masks, context)
            all_masks = torch.cat([masks, neg])
            all_ctx = torch.cat([context, context])
            all_embed = img_embed.repeat(2, 1, 1, 1)
            all_t = t_emb.repeat(2, 1)
            with amp.autocast():
                out = sacf(all_embed, all_masks, all_ctx, all_t)
            scores = out['score'].float().cpu().numpy()
            per_type_scores[t][0].extend(scores.tolist())
            per_type_scores[t][1].extend([0] * N + [1] * N)

            iou = field_iou(out['fused'][N:].float(), err, adaptive=True)
            if iou is not None:
                per_type_iou[t].append(iou)
            iou_o = field_iou_oracle(out['fused'][N:].float(), err)
            if iou_o is not None:
                per_type_iou_oracle[t].append(iou_o)

            # evidence 响应: 错误区域内各 evidence 场的均值
            fields = out['fields'][N:].float()    # (N,4,H,W)
            err_m = err[:, 0].flatten(1).float()  # (N,HW) 错误区域
            for k in range(len(EVIDENCE_NAMES)):
                fk = fields[:, k].flatten(1)      # (N,HW)
                resp = (fk * err_m).sum(1) / (err_m.sum(1) + 1e-6)
                evi_resp[t][k] += resp.sum().item()
            evi_cnt[t] += N

    print('\n===== 机制实验: 逐类型 Error Detection (score AUROC) =====')
    lines = ['type\tdetect_AUROC\tfield_IoU@5%\toracle_IoU']
    for t in CORRUPTION_TYPES:
        s, y = per_type_scores[t]
        auroc = rank_auroc(s, y)
        iou = np.nanmean(per_type_iou[t]) if per_type_iou[t] else float('nan')
        iou_o = np.nanmean(per_type_iou_oracle[t]) if per_type_iou_oracle[t] else float('nan')
        lines.append(f'{t}\t{auroc:.4f}\t{iou:.4f}\t{iou_o:.4f}')
        print(f'  {t:16s} AUROC={auroc:.4f}  IoU@5%={iou:.4f}  oracleIoU={iou_o:.4f}')

    print('\n===== 机制实验: Evidence 分解 (错误区域内各 evidence 响应) =====')
    header = 'corruption\\evidence\t' + '\t'.join(EVIDENCE_NAMES)
    lines.append('')
    lines.append(header)
    print(header)
    for t in CORRUPTION_TYPES:
        if evi_cnt[t] == 0:
            continue
        vals = evi_resp[t] / evi_cnt[t]
        row = f'{t}\t' + '\t'.join(f'{v:.4f}' for v in vals)
        lines.append(row)
        print(row)

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print(f'\n[saved] {args.out}')


# ---------------------------------------------------------------- mode 2

@torch.no_grad()
def eval_prediction(args, imis, sacf, device):
    """GT-free 评估: 对 IMISNet 真实预测输出矛盾场, 与真实误差对比."""
    min_area = int(16 * (args.image_size / 256.0) ** 2)
    ds = SACFDataset(args.data_dir, args.split_json, 'validation', args.image_size,
                     min_organ_area=min_area)
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    pix_scores, pix_labels = [], []      # pixel-level (聚合到 slice)
    ious = []
    slice_scores, slice_errs = [], []    # slice-level detection
    per_organ = {}

    for bi, batch in enumerate(loader):
        images = batch['image'].to(device)               # (1,3,H,W)
        label = batch['label'].to(device)[0]             # (C,H,W)
        organ_ids = batch['organ_ids'][0].tolist()       # GT 出现的器官 (仅用于枚举协议)
        C, H, W = label.shape

        img_embed = imis.image_encoder(images)           # (1,768,h,w)
        names = [ds.classes[c] for c in organ_ids]
        if len(names) == 0:
            continue

        # ---- 第一遍: 逐器官预测 (text+bbox 交互协议) ----
        bboxes = []
        for c in organ_ids:
            m = label[c:c + 1].unsqueeze(0)
            bboxes.append(get_bboxes_from_mask(m, offset=0).to(device))
        bboxes = torch.cat(bboxes, dim=0)                # (N,4)
        text = imis.process_text_prompt(names)           # {'text_inputs': (N,768)}
        prompts = {'bboxes': bboxes}
        prompts.update(text)
        embed_rep = img_embed.repeat(len(organ_ids), 1, 1, 1)
        with amp.autocast():
            outputs = imis.forward_decoder(embed_rep, prompts)
        preds = (torch.sigmoid(outputs['masks']) > 0.5).float()  # (N,1,H,W)

        # ---- 第二遍: SACF GT-free 评估 (context = 其他器官预测) ----
        pred_ctx = torch.zeros(1, C, H, W, device=device)
        for i, c in enumerate(organ_ids):
            pred_ctx[0, c] = preds[i, 0]

        for i, c in enumerate(organ_ids):
            context = pred_ctx.clone()
            context[0, c] = 0.0
            gt_m = label[c:c + 1].unsqueeze(0)
            true_err = (preds[i:i + 1] - gt_m).abs()
            with amp.autocast():
                out = sacf(img_embed, preds[i:i + 1], context,
                           encode_structured_text(
                               imis, [get_anatomy_prompt(ds.classes[c])]))
            fused = out['fused'].float()
            score = out['score'].float().item()

            dice_err = 1.0 - _dice(preds[i:i + 1], gt_m)
            slice_scores.append(score)
            slice_errs.append(dice_err)
            s, y = fused.flatten().cpu(), true_err.flatten().cpu()
            pix_scores.append(s)
            pix_labels.append(y)
            iou = field_iou(fused, true_err, adaptive=True)
            if iou is not None:
                ious.append(iou)
            per_organ.setdefault(ds.classes[c], []).append((score, dice_err))

    # ---- 汇总 ----
    all_s = torch.cat(pix_scores).numpy()
    all_y = torch.cat(pix_labels).clamp(0, 1).numpy()
    # 负样本下采样 (10x 正样本数), 控制 pixel AUROC 计算量
    pos_idx = np.where(all_y > 0.5)[0]
    neg_idx = np.where(all_y <= 0.5)[0]
    if len(pos_idx) > 0 and len(neg_idx) > 0:
        rng = np.random.RandomState(0)
        keep_neg = rng.choice(neg_idx, size=min(len(neg_idx), 10 * len(pos_idx)),
                              replace=False)
        keep = np.concatenate([pos_idx, keep_neg])
        pix_auroc = rank_auroc(all_s[keep], all_y[keep])
    else:
        pix_auroc = float('nan')

    # slice-level: score 与 dice error 的相关性 / 检测 AUROC (err>0.1 为坏样本)
    sl_err = np.array(slice_errs)
    sl_sc = np.array(slice_scores)
    bad = (sl_err > 0.1).astype(int)
    sl_auroc = rank_auroc(sl_sc, bad) if bad.sum() > 0 and bad.sum() < len(bad) else float('nan')
    corr = float(np.corrcoef(sl_sc, sl_err)[0, 1]) if len(sl_sc) > 2 else float('nan')

    print('\n===== GT-free 评估 (IMISNet 真实预测) =====')
    lines = [
        f'pixel AUROC          : {pix_auroc:.4f}',
        f'Error Localization IoU@5%: {np.mean(ious) if ious else float("nan"):.4f} '
        f'(over {len(ious)} samples with errors)',
        f'slice detection AUROC (dice_err>0.1): {sl_auroc:.4f}',
        f'score-vs-error corr  : {corr:.4f}',
    ]
    for l in lines:
        print(l)

    print('\n----- per-organ score (mean ± std) / mean dice error -----')
    for name, v in sorted(per_organ.items()):
        sc = np.array([x[0] for x in v])
        er = np.array([x[1] for x in v])
        print(f'  {name:28s} score={sc.mean():.3f}±{sc.std():.3f}  dice_err={er.mean():.3f}')
        lines.append(f'{name}\t{sc.mean():.4f}\t{er.mean():.4f}')

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print(f'\n[saved] {args.out}')


def _dice(a, b):
    inter = (a * b).sum()
    return (2 * inter / (a.sum() + b.sum() + 1e-6)).item()


if __name__ == '__main__':
    args = parse_args()
    imis, sacf, device = load_models(args)
    if args.mode == 'corruption':
        eval_corruption(args, imis, sacf, device)
    else:
        eval_prediction(args, imis, sacf, device)
