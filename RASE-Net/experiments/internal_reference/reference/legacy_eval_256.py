"""256 内部协议全指标评估 (Base / A:SCAR / B:Base+EGRC / A+B:SCAR+EGRC).

协议 = train_egrc.py run_val (即训练日志 0.823/0.847 那套), 完全复现:
  - 256 cache (base/field/refined/gt), query slice 在线编码 (512 输入)
  - 检索: pooled emb + GT 布局 (内部协议), 候选 = 训练集该器官全部 slice
  - 对齐: pred 质心平移 (B→base pred, A+B→refined pred)
  - EGRC 输入 field: 真实 SACF field (与 run_val 行为一致)
  - final = sigmoid(logit(pred) + residual), 0.5 阈值 @256
  - 少样本 6 器官以外直通: B=Base, A+B=SCAR
指标口径: Dice/IoU/Prec/Recall/HD95/ASSD/NSD@2px (同 eval_segmentation.py, 256 像素单位)

用法:
  cd . && python scar/eval/eval_egrc_256.py
"""

import sys as _sys, os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
_EVAL = _os.path.join(_ROOT, 'scar', 'eval')
if _EVAL not in _sys.path:
    _sys.path.insert(0, _EVAL)
_TRAIN = _os.path.join(_ROOT, 'scar', 'train')
if _TRAIN not in _sys.path:
    _sys.path.insert(0, _TRAIN)

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch

from segment_anything import sam_model_registry
from model import IMISNet
from egrc import EGRCCorrector, logit
from train_egrc import (load_cache, load_image, retrieve_topk, align_exemplar,
                        encode_slices)
from eval_egrc_compare import seg_metrics, FEWSHOT_NAMES  # noqa: E402  同口径指标


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, default='dataset/BTCV_full')
    p.add_argument('--split_json', type=str, default='dataset_split_full_24_6.json')
    p.add_argument('--image_size', type=int, default=512,
                   help='query 在线编码输入尺寸 (与 run_val 一致)')
    p.add_argument('--sam_checkpoint', type=str, default='ckpt/IMISNet-B.pth')
    p.add_argument('--imis_ckpt', type=str,
                   default='work_dir/ft-IMISNet-full/IMIS_val_dice_best.pth')
    p.add_argument('--model_type', type=str, default='vit_b')
    p.add_argument('--egrc_noscar_ckpt', type=str,
                   default='work_dir/egrc_fewshot/ckpt/egrc_best_noscar.pth',
                   help='B 列: Base+EGRC (no_scar fewshot 训练)')
    p.add_argument('--egrc_scar_ckpt', type=str,
                   default='work_dir/egrc_fewshot_scar/ckpt/egrc_best.pth',
                   help='A+B 列: SCAR+EGRC (fewshot 训练)')
    p.add_argument('--bank_path', type=str, default='work_dir/egrc/exemplar_bank.pt')
    p.add_argument('--cache_dir', type=str, default='work_dir/egrc/cache')
    p.add_argument('--K', type=int, default=4)
    p.add_argument('--val_batch_slices', type=int, default=8)
    p.add_argument('--out_json', type=str, default='work_dir/egrc/eval_egrc_256.json')
    p.add_argument('--device', type=str, default='cuda')
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device)

    # ---- bank ----
    bank = torch.load(args.bank_path, map_location='cpu', weights_only=False)
    emb_tr = torch.stack([s['emb'] for s in bank['info']])
    lay_tr = np.stack([s['layout'] for s in bank['info']])
    organ_slices = defaultdict(list)
    for e in bank['entries']:
        organ_slices[e['organ']].append(e['slice'])
    print(f'[bank] {len(bank["entries"])} entries, {len(bank["info"])} slices')

    # ---- caches ----
    print('[cache] loading train...')
    cache_tr = load_cache(os.path.join(args.cache_dir, 'training'))
    print(f'[cache] train {len(cache_tr)} slices')
    print('[cache] loading val...')
    cache_va = load_cache(os.path.join(args.cache_dir, 'validation'))
    print(f'[cache] val {len(cache_va)} slices')

    ds = json.load(open(os.path.join(args.data_dir, args.split_json)))
    datalist_va = ds['validation']
    classes = ds['labels']
    focus_ids = {int(k) - 1 for k, name in classes.items() if name in FEWSHOT_NAMES}
    assert focus_ids, 'fewshot organ names not found in split labels'
    print(f'[fewshot] focus organs: {sorted(classes[str(c + 1)] for c in focus_ids)}')

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
    print(f'[IMISNet] frozen, loaded {args.imis_ckpt}')

    egrc_models = {}
    for tag, ck_path in [('noscar', args.egrc_noscar_ckpt), ('scar', args.egrc_scar_ckpt)]:
        eck = torch.load(ck_path, map_location='cpu')
        m = EGRCCorrector(d=128, n_organs=13, n_heads=4).to(device)
        m.load_state_dict(eck['model_state_dict'])
        m.eval()
        egrc_models[tag] = m
        print(f'[models] EGRC-{tag} loaded from ep{eck.get("epoch", "?")} '
              f'(val_dice={eck.get("val_dice", float("nan")):.4f})')

    # ---- 评估主循环 (协议逐行复现 run_val) ----
    per = defaultdict(lambda: defaultdict(list))   # organ -> method -> [metrics]
    n_all = 0
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
                if organ not in focus_ids:
                    # 非少样本器官: 直通 (B=Base, A+B=SCAR), 指标同口径
                    pos = d['pos'][organ]
                    gt_np = d['gt'][pos].float().numpy() > 0.5
                    mb = seg_metrics(d['base'][pos].float().numpy() > 0.5, gt_np)
                    ms = seg_metrics(d['refined'][pos].float().numpy() > 0.5, gt_np)
                    per[organ]['base'].append(mb)
                    per[organ]['scar'].append(ms)
                    per[organ]['b'].append(mb)
                    per[organ]['ab'].append(ms)
                    n_all += 1
                else:
                    samples.append((ui, s, organ))
        if not samples:
            continue

        # 检索 + 对齐 (CPU) — B 对齐 base pred, A+B 对齐 refined pred
        exb_list, exr_list = [], []
        predb_list, predr_list, field_list, gt_list, organs = [], [], [], [], []
        for (ui, s, organ) in samples:
            d = cache_va[s]
            pos = d['pos'][organ]
            base = d['base'][pos].float()
            refined = d['refined'][pos].float()
            q_lay = np.zeros(13, dtype=np.float32)
            for og in d['organs']:
                q_lay[og] = float(d['gt'][d['pos'][og]].float().mean())
            exs = retrieve_topk(pooled[ui].cpu(), q_lay, organ_slices.get(organ, []),
                                emb_tr, lay_tr, args.K)
            mb_ = [align_exemplar(
                cache_tr[es]['gt'][cache_tr[es]['pos'][organ]].float(), base)
                for es in exs]
            mr_ = [align_exemplar(
                cache_tr[es]['gt'][cache_tr[es]['pos'][organ]].float(), refined)
                for es in exs]
            while len(mb_) < args.K:
                mb_.append(torch.zeros(256, 256))
            while len(mr_) < args.K:
                mr_.append(torch.zeros(256, 256))
            exb_list.append(torch.stack(mb_[:args.K]))
            exr_list.append(torch.stack(mr_[:args.K]))
            predb_list.append(base)
            predr_list.append(refined)
            field_list.append(d['field'][pos].float())   # run_val: 真实 field
            gt_list.append(d['gt'][pos].float())
            organs.append(organ)

        # 分批前向 (两个模型, 与 run_val 相同的 batch 方式)
        for i0 in range(0, len(samples), 16):
            i1 = min(i0 + 16, len(samples))
            sub = list(range(i0, i1))
            uq = sorted({samples[i][0] for i in sub})
            emb_sub = emb[[uq.index(samples[i][0]) for i in sub]]
            predb_b = torch.stack([predb_list[i] for i in sub])[:, None].to(device)
            predr_b = torch.stack([predr_list[i] for i in sub])[:, None].to(device)
            field_b = torch.stack([field_list[i] for i in sub])[:, None].to(device)
            exb_b = torch.stack([exb_list[i] for i in sub])[:, :, None].to(device)
            exr_b = torch.stack([exr_list[i] for i in sub])[:, :, None].to(device)
            organ_b = torch.tensor([organs[i] for i in sub], dtype=torch.long,
                                   device=device)
            with torch.autocast('cuda', dtype=torch.float16):
                resb = egrc_models['noscar'](emb_sub, predb_b, field_b, exb_b, organ_b)
                resr = egrc_models['scar'](emb_sub, predr_b, field_b, exr_b, organ_b)
            finb = torch.sigmoid(logit(predb_b) + resb.float())
            finr = torch.sigmoid(logit(predr_b) + resr.float())
            for j, i in enumerate(sub):
                organ = organs[i]
                gt_np = gt_list[i].numpy() > 0.5
                d = cache_va[samples[i][1]]
                pos = d['pos'][organ]
                per[organ]['base'].append(seg_metrics(
                    d['base'][pos].float().numpy() > 0.5, gt_np))
                per[organ]['scar'].append(seg_metrics(
                    d['refined'][pos].float().numpy() > 0.5, gt_np))
                per[organ]['b'].append(seg_metrics(
                    finb[j, 0].detach().cpu().numpy() > 0.5, gt_np))
                per[organ]['ab'].append(seg_metrics(
                    finr[j, 0].detach().cpu().numpy() > 0.5, gt_np))
                n_all += 1
        print(f'  [slice batch {bi0 // args.val_batch_slices + 1}/'
              f'{(len(sls) + args.val_batch_slices - 1) // args.val_batch_slices}] '
              f'n={n_all}', flush=True)

    # ---- 汇总: 每方法一张全指标表 (MEAN + MACRO) ----
    methods = ['base', 'scar', 'b', 'ab']
    KEYS = ['dice', 'iou', 'precision', 'recall', 'hd95', 'assd', 'nsd']
    result = {'n_samples': n_all, 'protocol': '256 cache (run_val一致)', 'per_organ': {},
              'mean': {}, 'macro': {}, 'fewshot_mean': {}}
    tables = []

    def method_table(m, title):
        organ_mean = {}
        for c in per:
            organ_mean[c] = {k: float(np.nanmean([x[k] for x in per[c][m]]))
                             for k in KEYS}
        micro = {k: float(np.nanmean([x[k] for c in per for x in per[c][m]]))
                 for k in KEYS}
        macro = {k: float(np.nanmean([v[k] for v in organ_mean.values()]))
                 for k in KEYS}
        t = [f'{"=" * 100}',
             f'{title}   [256 内部协议]',
             f'{"=" * 100}',
             f'{"organ":<32s} {"n":>4s} {"Dice":>7s} {"IoU":>7s} {"Prec":>7s} '
             f'{"Recall":>7s} {"HD95":>7s} {"ASSD":>7s} {"NSD":>7s}']
        for c in sorted(organ_mean, key=lambda c: -organ_mean[c]['dice']):
            v = organ_mean[c]
            t.append(f'{classes[str(c + 1)]:<32s} {len(per[c][m]):>4d} {v["dice"]:>7.3f} '
                     f'{v["iou"]:>7.3f} {v["precision"]:>7.3f} {v["recall"]:>7.3f} '
                     f'{v["hd95"]:>7.2f} {v["assd"]:>7.2f} {v["nsd"]:>7.3f}')
        t.append('-' * 100)
        t.append(f'{"MEAN":<32s} {n_all:>4d} {micro["dice"]:>7.3f} {micro["iou"]:>7.3f} '
                 f'{micro["precision"]:>7.3f} {micro["recall"]:>7.3f} '
                 f'{micro["hd95"]:>7.2f} {micro["assd"]:>7.2f} {micro["nsd"]:>7.3f}')
        t.append(f'{"MACRO":<32s} {len(organ_mean):>4d} {macro["dice"]:>7.3f} '
                 f'{macro["iou"]:>7.3f} {macro["precision"]:>7.3f} {macro["recall"]:>7.3f} '
                 f'{macro["hd95"]:>7.2f} {macro["assd"]:>7.2f} {macro["nsd"]:>7.3f}')
        return t, organ_mean, micro, macro

    titles = {'base': 'Base (IMISNet)', 'scar': 'A: SCAR (SACF + FieldGuidedRefiner)',
              'b': 'B: Base + EGRC (independent)', 'ab': 'A+B: SCAR + EGRC (full)'}
    for m in methods:
        t, organ_mean, micro, macro = method_table(m, titles[m])
        tables.append('\n'.join(t))
        for c in organ_mean:
            result['per_organ'].setdefault(classes[str(c + 1)],
                                            {'n': len(per[c][m])})[m] = organ_mean[c]
        result['mean'][m] = micro
        result['macro'][m] = macro

    # 少样本子集 micro 均值
    fm = {}
    for m in methods:
        allm = [x for c in per if c in focus_ids for x in per[c][m]]
        fm[m] = {k: float(np.nanmean([x[k] for x in allm])) for k in KEYS}
    result['fewshot_mean'] = fm

    # 四列 Dice 对比表
    cmp_lines = [f'{"organ":<32s} {"n":>4s}  {"Base":>7s} {"SCAR":>7s} {"B":>7s} '
                 f'{"A+B":>7s}  {"Δ(B-Base)":>10s} {"Δ(AB-SCAR)":>10s}']
    for c in sorted(per, key=lambda c: -np.mean([x['dice'] for x in per[c]['base']])):
        db = {m: np.mean([x['dice'] for x in per[c][m]]) for m in methods}
        cmp_lines.append(f'{classes[str(c + 1)]:<32s} {len(per[c]["b"]):>4d}  '
                         f'{db["base"]:>7.3f} {db["scar"]:>7.3f} {db["b"]:>7.3f} '
                         f'{db["ab"]:>7.3f}  {db["b"] - db["base"]:>+10.3f} '
                         f'{db["ab"] - db["scar"]:>+10.3f}')
    cmp_lines.append('-' * 100)
    cmp_lines.append(f'{"MEAN":<32s} {n_all:>4d}  '
                     f'{result["mean"]["base"]["dice"]:>7.3f} '
                     f'{result["mean"]["scar"]["dice"]:>7.3f} '
                     f'{result["mean"]["b"]["dice"]:>7.3f} '
                     f'{result["mean"]["ab"]["dice"]:>7.3f}')
    cmp_lines.append(f'{"MACRO":<32s} {len(per):>4d}  '
                     f'{result["macro"]["base"]["dice"]:>7.3f} '
                     f'{result["macro"]["scar"]["dice"]:>7.3f} '
                     f'{result["macro"]["b"]["dice"]:>7.3f} '
                     f'{result["macro"]["ab"]["dice"]:>7.3f}')
    n_fs = sum(len(per[c]['b']) for c in per if c in focus_ids)
    cmp_lines.append(f'{"FEWSHOT-MEAN":<32s} {n_fs:>4d}  '
                     f'{fm["base"]["dice"]:>7.3f} {fm["scar"]["dice"]:>7.3f} '
                     f'{fm["b"]["dice"]:>7.3f} {fm["ab"]["dice"]:>7.3f}')

    report = '\n'.join(tables) + '\n\n' + '\n'.join(cmp_lines)
    print(report)
    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open(args.out_json.replace('.json', '_tables.txt'), 'w',
              encoding='utf-8') as f:
        f.write(report)
    print(f'[saved] {args.out_json} / {args.out_json.replace(".json", "_tables.txt")}')


if __name__ == '__main__':
    main()
