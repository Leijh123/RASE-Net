"""Refiner 最终评估: 基准 IMISNet vs SACF 精修后, 逐器官全指标对比.

协议与基准评估一致 (GT bbox prompt), 指标口径同用户基准表:
  Dice / IoU / Precision / Recall / HD95 / ASSD / NSD@2px
输出:
  work_dir/refiner/eval_refiner_<tag>.txt

用法:
  python eval_refiner.py                 # 评估场指导精修
  python eval_refiner.py --no_field      # 评估无场消融
"""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.cuda import amp
from scipy import ndimage

from segment_anything import sam_model_registry
from model import IMISNet
from dataloaders.data_utils import get_bboxes_from_mask
from sacf import (SemanticContradictionField, FieldGuidedRefiner, SACFDataset,
                  get_anatomy_prompt, encode_structured_text)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, default='dataset/BTCV_full')
    p.add_argument('--split_json', type=str, default='dataset_split_full_24_6.json')
    p.add_argument('--image_size', type=int, default=512)
    p.add_argument('--sam_checkpoint', type=str, default='ckpt/IMISNet-B.pth')
    p.add_argument('--imis_ckpt', type=str,
                   default='work_dir/ft-IMISNet-full/IMIS_val_dice_best.pth')
    p.add_argument('--sacf_ckpt', type=str, default='work_dir/sacf/sacf_best.pth')
    p.add_argument('--refiner_ckpt', type=str, default=None,
                   help='默认自动选择 work_dir/refiner/refiner_best_{field,no_field}.pth')
    p.add_argument('--model_type', type=str, default='vit_b')
    p.add_argument('--dim', type=int, default=192)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--nsd_tol', type=float, default=2.0)
    p.add_argument('--no_field', action='store_true')
    p.add_argument('--min_organ_area', type=int, default=1,
                   help='1=面积>0全纳入, 对齐基准评估协议 (n~2204)')
    p.add_argument('--out', type=str, default='')
    p.add_argument('--device', type=str, default='cuda')
    return p.parse_args()


# ---------------------------------------------------------------- metrics

def seg_metrics(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    dice = 2 * tp / (2 * tp + fp + fn + 1e-6)
    return {
        'dice': dice,
        'iou': tp / (tp + fp + fn + 1e-6),
        'precision': tp / (tp + fp + 1e-6),
        'recall': tp / (tp + fn + 1e-6),
        'hd95': _hd95(pred, gt),
        'assd': _assd(pred, gt),
        'nsd': _nsd(pred, gt, tol=2.0),
    }


def _surface(m):
    if m.sum() == 0:
        return m
    er = ndimage.binary_erosion(m, np.ones((3, 3), bool), border_value=0)
    return m & ~er


def _surf_dists(a, b):
    if a.sum() == 0 or b.sum() == 0:
        return None, None
    sa, sb = _surface(a), _surface(b)
    db = ndimage.distance_transform_edt(~b)
    da = ndimage.distance_transform_edt(~a)
    return db[sa], da[sb]


def _hd95(a, b):
    d_ab, d_ba = _surf_dists(a, b)
    if d_ab is None:
        return float('nan')
    return max(np.percentile(d_ab, 95), np.percentile(d_ba, 95))


def _assd(a, b):
    d_ab, d_ba = _surf_dists(a, b)
    if d_ab is None:
        return float('nan')
    return (d_ab.mean() + d_ba.mean()) / 2.0


def _nsd(a, b, tol=2.0):
    d_ab, d_ba = _surf_dists(a, b)
    if d_ab is None:
        return float('nan')
    return ((d_ab <= tol).sum() + (d_ba <= tol).sum()) / (len(d_ab) + len(d_ba))


# ---------------------------------------------------------------- main

@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device)
    tag = 'no_field' if args.no_field else 'field'
    if args.refiner_ckpt is None:
        args.refiner_ckpt = f'work_dir/refiner/refiner_best_{tag}.pth'

    sam = sam_model_registry[args.model_type](args).to(device)
    imis = IMISNet(sam, test_mode=True,
                   category_weights='dataloaders/categories_weight.pkl').to(device)
    sd = torch.load(args.imis_ckpt, map_location='cpu')
    if isinstance(sd, dict) and 'model_state_dict' in sd:
        sd = sd['model_state_dict']
    imis.load_state_dict(sd, strict=False)
    imis.eval()
    print(f'[IMISNet] loaded {args.imis_ckpt}')

    ckpt = torch.load(args.sacf_ckpt, map_location='cpu')
    sacf = SemanticContradictionField(num_organs=13, image_embed_dim=768,
                                      text_embed_dim=768, dim=args.dim,
                                      image_size=args.image_size).to(device)
    sacf.load_state_dict(ckpt['sacf_state_dict'])
    sacf.eval()
    print(f'[SACF] loaded {args.sacf_ckpt}')

    rck = torch.load(args.refiner_ckpt, map_location='cpu')
    refiner = FieldGuidedRefiner(image_embed_dim=768, text_embed_dim=768,
                                 dim=args.dim).to(device)
    refiner.load_state_dict(rck['refiner_state_dict'])
    refiner.eval()
    print(f'[Refiner] loaded {args.refiner_ckpt} '
          f'(epoch {rck.get("epoch", "?")}, val refined_dice={rck.get("refined_dice", "?")})')

    min_area = args.min_organ_area
    ds = SACFDataset(args.data_dir, args.split_json, 'validation',
                     args.image_size, min_organ_area=min_area)
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    C = ds.num_organs
    print(f'[data] val={len(ds)} slices')

    per_organ = {}
    n_total = 0

    for bi, batch in enumerate(loader):
        images = batch['image'].to(device)
        label = batch['label'].to(device)[0]
        organ_ids = batch['organ_ids'][0].tolist()
        if not organ_ids:
            continue
        N = len(organ_ids)
        H, W = label.shape[-2:]

        img_embed = imis.image_encoder(images)
        names = [ds.classes[c] for c in organ_ids]
        bboxes = torch.cat([
            get_bboxes_from_mask(label[c:c + 1].unsqueeze(0), offset=0).to(device)
            for c in organ_ids], dim=0)
        text = imis.process_text_prompt(names)
        prompts = {'bboxes': bboxes}
        prompts.update(text)
        embed_rep = img_embed.repeat(N, 1, 1, 1)
        with amp.autocast():
            outputs = imis.forward_decoder(embed_rep, prompts)
            pred_prob = torch.sigmoid(outputs['masks'].float())

            pred_ctx = torch.zeros(1, C, H, W, device=device)
            for i, c in enumerate(organ_ids):
                pred_ctx[0, c] = pred_prob[i, 0]
            # SACF 评判文本: 结构化解剖描述 (必改1, 与训练一致)
            sacf_text = encode_structured_text(
                imis, [get_anatomy_prompt(n) for n in names])
            for i, c in enumerate(organ_ids):
                context = pred_ctx.clone()
                context[0, c] = 0.0
                sacf_out = sacf(img_embed, pred_prob[i:i + 1], context,
                                sacf_text[i:i + 1])
                field = sacf_out['fused'].float()
                f_in = torch.zeros_like(field) if args.no_field else field
                logit = refiner(img_embed, pred_prob[i:i + 1], f_in,
                                sacf_text[i:i + 1])
                refined = (torch.sigmoid(logit.float()) > 0.5).float()

                gt_np = label[c].cpu().numpy() > 0.5
                pb = pred_prob[i, 0].cpu().numpy() > 0.5
                pr = refined[0, 0].cpu().numpy() > 0.5
                mb, mr = seg_metrics(pb, gt_np), seg_metrics(pr, gt_np)

                name = ds.classes[c]
                rec = per_organ.setdefault(name, {'base': [], 'ref': []})
                rec['base'].append(mb)
                rec['ref'].append(mr)
                n_total += 1
        if bi % 50 == 0:
            print(f'  [progress] slice {bi}/{len(loader)}')

    # ---------------------------------------------------------------- 汇总
    L = []
    def add(s=''):
        L.append(s)
        print(s)

    KEYS = ['dice', 'iou', 'precision', 'recall', 'hd95', 'assd', 'nsd']

    def table(title, which):
        add('=' * 100)
        add(f'{title}   validation, {n_total} organ-samples, {args.image_size}x{args.image_size}')
        add('=' * 100)
        add(f'{"organ":<30s} {"n":>4s}  {"Dice":>7s} {"IoU":>7s} {"Prec":>7s} '
            f'{"Recall":>7s} {"HD95":>7s} {"ASSD":>7s} {"NSD":>7s}')
        organ_mean = {}
        for name, rec in per_organ.items():
            organ_mean[name] = {k: np.nanmean([m[k] for m in rec[which]])
                                for k in KEYS}
        for name, m in sorted(organ_mean.items(), key=lambda kv: -kv[1]['dice']):
            add(f'{name:<30s} {len(per_organ[name][which]):>4d}  '
                f'{m["dice"]:>7.3f} {m["iou"]:>7.3f} {m["precision"]:>7.3f} '
                f'{m["recall"]:>7.3f} {m["hd95"]:>7.2f} {m["assd"]:>7.2f} '
                f'{m["nsd"]:>7.3f}')
        all_m = [m for rec in per_organ.values() for m in rec[which]]
        micro = {k: np.nanmean([m[k] for m in all_m]) for k in KEYS}
        macro = {k: np.nanmean([m[k] for m in organ_mean.values()]) for k in KEYS}
        add('-' * 100)
        add(f'{"MEAN":<30s} {n_total:>4d}  {micro["dice"]:>7.3f} {micro["iou"]:>7.3f} '
            f'{micro["precision"]:>7.3f} {micro["recall"]:>7.3f} {micro["hd95"]:>7.2f} '
            f'{micro["assd"]:>7.2f} {micro["nsd"]:>7.3f}')
        add(f'{"MACRO":<30s} {len(organ_mean):>4d}  {macro["dice"]:>7.3f} {macro["iou"]:>7.3f} '
            f'{macro["precision"]:>7.3f} {macro["recall"]:>7.3f} {macro["hd95"]:>7.2f} '
            f'{macro["assd"]:>7.2f} {macro["nsd"]:>7.3f}')
        add('')
        return organ_mean, micro, macro

    mb_organ, mb_micro, mb_macro = table(
        f'[Base] IMISNet ({os.path.basename(args.imis_ckpt)})', 'base')
    mr_organ, mr_micro, mr_macro = table(
        f'[Ours] SACF + FieldGuidedRefiner ({os.path.basename(args.refiner_ckpt)})',
        'ref')

    # delta 表 (按 dDice 降序)
    add('=' * 100)
    add(f'{"Delta (Ours - Base)":<30s} {"n":>4s}  {"dDice":>7s} {"dIoU":>7s} '
        f'{"dPrec":>7s} {"dRecall":>7s} {"dHD95":>7s} {"dASSD":>7s} {"dNSD":>7s}')
    add('=' * 100)
    for name in sorted(mb_organ, key=lambda n: -(mr_organ[n]['dice'] - mb_organ[n]['dice'])):
        db, dr = mb_organ[name], mr_organ[name]
        add(f'{name:<30s} {len(per_organ[name]["base"]):>4d}  '
            f'{dr["dice"]-db["dice"]:>+7.3f} {dr["iou"]-db["iou"]:>+7.3f} '
            f'{dr["precision"]-db["precision"]:>+7.3f} {dr["recall"]-db["recall"]:>+7.3f} '
            f'{dr["hd95"]-db["hd95"]:>+7.2f} {dr["assd"]-db["assd"]:>+7.2f} '
            f'{dr["nsd"]-db["nsd"]:>+7.3f}')
    add('-' * 100)
    add(f'{"MEAN":<30s} {n_total:>4d}  {mr_micro["dice"]-mb_micro["dice"]:>+7.3f} '
        f'{mr_micro["iou"]-mb_micro["iou"]:>+7.3f} {mr_micro["precision"]-mb_micro["precision"]:>+7.3f} '
        f'{mr_micro["recall"]-mb_micro["recall"]:>+7.3f} {mr_micro["hd95"]-mb_micro["hd95"]:>+7.2f} '
        f'{mr_micro["assd"]-mb_micro["assd"]:>+7.2f} {mr_micro["nsd"]-mb_micro["nsd"]:>+7.3f}')
    add(f'{"MACRO":<30s} {len(mb_organ):>4d}  {mr_macro["dice"]-mb_macro["dice"]:>+7.3f} '
        f'{mr_macro["iou"]-mb_macro["iou"]:>+7.3f} {mr_macro["precision"]-mb_macro["precision"]:>+7.3f} '
        f'{mr_macro["recall"]-mb_macro["recall"]:>+7.3f} {mr_macro["hd95"]-mb_macro["hd95"]:>+7.2f} '
        f'{mr_macro["assd"]-mb_macro["assd"]:>+7.2f} {mr_macro["nsd"]-mb_macro["nsd"]:>+7.3f}')

    # 每样本配对 dice 差 (检验显著性)
    diffs = []
    for name, rec in per_organ.items():
        for mb_, mr_ in zip(rec['base'], rec['ref']):
            diffs.append(mr_['dice'] - mb_['dice'])
    diffs = np.array(diffs)
    t_stat = diffs.mean() / (diffs.std() / np.sqrt(len(diffs)) + 1e-9)
    add('')
    add(f'paired dDice: mean={diffs.mean():+.4f}, std={diffs.std():.4f}, '
        f'n={len(diffs)}, t={t_stat:.2f}')
    better = (diffs > 0.001).sum()
    worse = (diffs < -0.001).sum()
    add(f'  改善样本: {better} ({100 * better / len(diffs):.1f}%), '
        f'恶化样本: {worse} ({100 * worse / len(diffs):.1f}%), '
        f'不变: {len(diffs) - better - worse}')

    out = args.out or f'work_dir/refiner/eval_refiner_{tag}.txt'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        f.write('\n'.join(L) + '\n')
    print(f'\n[saved] {out}')


if __name__ == '__main__':
    main()
