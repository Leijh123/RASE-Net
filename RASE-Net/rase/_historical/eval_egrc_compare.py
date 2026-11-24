"""EGRC 最终评测: Base (IMISNet) vs SCAR (SACF+Refiner) vs SCAR+EGRC.

协议与 error_attribution.py / oracle 系列完全一致:
  - GT bbox prompt, min_organ_area=1, 全器官 (n~2204)
  - 512 分辨率, 硬阈值 (0.5) Dice
  - 7 指标: Dice, IoU, HD95, Precision, Recall, VOE, RVD
  - EGRC 推理完全 GT-free: 检索布局用预测 context, exemplar 对齐用 refined pred

输出: work_dir/egrc/eval_egrc_compare.json + 控制台表格
"""

import sys as _sys, os as _os
_ROOT = _os.path.abspath(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
_SACF_DEV = _os.path.join(_ROOT, 'sacf_dev')
if _SACF_DEV not in _sys.path:
    _sys.path.insert(0, _SACF_DEV)

import argparse
import glob
import json
import math
import os
import random
import re
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda import amp
from scipy import ndimage

from segment_anything import sam_model_registry
from model import IMISNet
from dataloaders.data_utils import get_bboxes_from_mask
from scar.lib import (SemanticContradictionField, FieldGuidedRefiner, SACFDataset,
                      get_anatomy_prompt, encode_structured_text)

from egrc import EGRCCorrector, logit
from train_egrc import align_exemplar, load_image, FEWSHOT_NAMES  # noqa: E402,F401


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, default='dataset/BTCV_full')
    p.add_argument('--split_json', type=str, default='dataset_split_full_24_6.json')
    p.add_argument('--image_size', type=int, default=512)
    p.add_argument('--sam_checkpoint', type=str, default='ckpt/IMISNet-B.pth')
    p.add_argument('--imis_ckpt', type=str,
                   default='work_dir/ft-IMISNet-full/IMIS_val_dice_best.pth')
    p.add_argument('--sacf_ckpt', type=str, default='work_dir/scar/ckpt/sacf_best.pth')
    p.add_argument('--refiner_ckpt', type=str,
                   default='work_dir/scar/ckpt/refiner_best_field.pth')
    p.add_argument('--egrc_noscar_ckpt', type=str,
                   default='work_dir/egrc_fewshot/ckpt/egrc_best_noscar.pth',
                   help='B 列: Base+EGRC (no_scar 训练)')
    p.add_argument('--egrc_scar_ckpt', type=str,
                   default='work_dir/egrc_fewshot_scar/ckpt/egrc_best.pth',
                   help='A+B 列: SCAR+EGRC 训练')
    p.add_argument('--bank_path', type=str, default='work_dir/egrc/exemplar_bank.pt')
    p.add_argument('--cache_dir', type=str, default='work_dir/egrc/cache')
    p.add_argument('--model_type', type=str, default='vit_b')
    p.add_argument('--dim', type=int, default=192)
    p.add_argument('--K', type=int, default=4)
    p.add_argument('--min_organ_area', type=int, default=1)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--out_json', type=str, default='work_dir/egrc/eval_egrc_compare.json')
    p.add_argument('--kshot', type=int, default=0,
                   help='>0: 限制 exemplar bank 为每器官 K 个 case (随机固定 seed), '
                        'K-shot 预算消融; 0 = 全量 bank')
    p.add_argument('--two_stage', action='store_true',
                   help='两阶段迭代推理: 一阶段 EGRC 输出作为二阶段对齐锚点与修正基线')
    p.add_argument('--phase', type=str, default='validation',
                   help='validation / test / training, 从 split_json 里选哪一列评估')
    p.add_argument('--device', type=str, default=('cuda' if __import__('torch').cuda.is_available() else 'cpu'))
    # ---- CPU 并行分片: 只评估 datalist[start:end] 这一段, 便于多进程分片跑 ----
    p.add_argument('--start', type=int, default=0, help='分片起始切片下标 (含)')
    p.add_argument('--end', type=int, default=0, help='分片结束切片下标 (不含), 0=到末尾')
    p.add_argument('--dump_jsonl', type=str, default='',
                   help='把逐切片逐器官逐方法的指标写成 JSONL (分片合并用)')
    p.add_argument('--base_only', action='store_true',
                   help='只算 Base (IMISNet) 一列, 跳过 SACF/Refiner/EGRC (Base 复评提速)')
    return p.parse_args()


# 少样本器官 (方案2 训练/修正范围, 其余器官直通)
# ACDC 适配: 复用 train_egrc.py 的 FEWSHOT_NAMES = {'right_ventricle'},
# 不再硬编码 WORD 的 6 个器官名 (否则 ACDC 全部器官被旁路, EGRC 空转)。


# ----------------------------------------------------------------------------
# 指标 (口径与 eval_segmentation.py 完全一致: Dice/IoU/Prec/Recall/HD95/ASSD/NSD@2px)
# ----------------------------------------------------------------------------

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


# ----------------------------------------------------------------------------
# exemplar 工具
# ----------------------------------------------------------------------------

def load_exemplar_store(cache_dir):
    """只加载训练 cache 的 gt + organs (exemplar mask 来源)."""
    store = {}
    for f in sorted(glob.glob(os.path.join(cache_dir, 'training', '*.pt'))):
        idx = int(os.path.basename(f).split('.')[0])
        d = torch.load(f, map_location='cpu')
        organs = d['organs'].tolist()
        store[idx] = {'pos': {o: i for i, o in enumerate(organs)}, 'gt': d['gt']}
    return store


def retrieve_topk(q_emb, q_lay, cands, emb_tr, lay_tr, K):
    if not cands:
        return []
    ci = torch.tensor(cands, dtype=torch.long)
    sims = (emb_tr[ci] @ q_emb).numpy()
    lay_d = np.linalg.norm(lay_tr[ci.numpy()] - q_lay[None], axis=1)
    score = sims + np.exp(-lay_d / (lay_d.mean() + 1e-6))
    order = np.argsort(-score)[:K]
    return [cands[i] for i in order]


# ----------------------------------------------------------------------------
# 主评测
# ----------------------------------------------------------------------------

@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device)
    S = args.image_size

    # ---- bank / exemplar store (base_only 不需要检索, 直接跳过: 省 ~1GB 内存和启动 I/O) ----
    emb_tr = lay_tr = None
    organ_slices = defaultdict(list)
    ex_store = {}
    if args.base_only:
        print('[base_only] 跳过 exemplar bank / store 加载')
    else:
        bank = torch.load(args.bank_path, map_location='cpu', weights_only=False)
        emb_tr = torch.stack([s['emb'] for s in bank['info']])
        lay_tr = np.stack([s['layout'] for s in bank['info']])
        for e in bank['entries']:
            organ_slices[e['organ']].append(e['slice'])
        print(f'[bank] {len(bank["entries"])} entries')
        ex_store = load_exemplar_store(args.cache_dir)
        print(f'[exemplar store] {len(ex_store)} train slices')

    # ---- K-shot 预算限制: 每器官只保留 K 个随机 case 的 slices ----
    if args.kshot > 0:
        ds_json = json.load(open(os.path.join(args.data_dir, args.split_json)))
        case_of_slice = []
        for it in ds_json['training']:
            bn = os.path.basename(it['image'])
            m = re.match(r'(.+)_z\d+$', os.path.splitext(bn)[0])
            case_of_slice.append(m.group(1) if m else os.path.splitext(bn)[0])
        rng = random.Random(1234)                    # 固定 seed, K-shot 消融可复现
        slices_keep = set()
        for c in list(organ_slices):
            cases = {}
            for s in organ_slices[c]:
                cases.setdefault(case_of_slice[s], []).append(s)
            picked = rng.sample(sorted(cases), min(args.kshot, len(cases)))
            for cs in picked:
                slices_keep.update(cases[cs])
        n_before = sum(len(v) for v in organ_slices.values())
        organ_slices = defaultdict(list, {
            c: [s for s in sl if s in slices_keep]
            for c, sl in organ_slices.items()})
        n_after = sum(len(v) for v in organ_slices.values())
        print(f'[kshot] K={args.kshot} cases/organ → slices {n_before}→{n_after} '
              f'(retrieval pool only; exemplar GT 来源不变)')

    ex_store = load_exemplar_store(args.cache_dir)
    print(f'[exemplar store] {len(ex_store)} train slices')

    # ---- models ----
    sam = sam_model_registry[args.model_type](args).to(device)
    imis = IMISNet(sam, test_mode=True,
                   category_weights='dataloaders/categories_weight.pkl').to(device)
    sd = torch.load(args.imis_ckpt, map_location='cpu')
    if isinstance(sd, dict) and 'model_state_dict' in sd:
        sd = sd['model_state_dict']
    imis.load_state_dict(sd, strict=False)
    imis.eval()

    ds_json = json.load(open(os.path.join(args.data_dir, args.split_json)))
    n_organs = len(ds_json.get('labels', {})) - 1
    print(f'[info] n_organs={n_organs} (from split_json.labels)')

    sacf = refiner = None
    if args.base_only:
        print('[base_only] 跳过 SACF / Refiner / EGRC, 只评 Base')
    else:
        sacf = SemanticContradictionField(num_organs=n_organs, image_embed_dim=768,
                                          text_embed_dim=768, dim=args.dim,
                                          image_size=args.image_size).to(device)
        sck = torch.load(args.sacf_ckpt, map_location='cpu')
        sacf.load_state_dict(sck['sacf_state_dict'])
        sacf.eval()

        rck = torch.load(args.refiner_ckpt, map_location='cpu')
        refiner = FieldGuidedRefiner(image_embed_dim=768, text_embed_dim=768,
                                     dim=args.dim).to(device)
        refiner.load_state_dict(rck['refiner_state_dict'])
        refiner.eval()

    egrc_models = {}
    for tag, ck_path in ([] if args.base_only else
                         [('noscar', args.egrc_noscar_ckpt), ('scar', args.egrc_scar_ckpt)]):
        if not os.path.exists(ck_path):
            if tag == 'noscar':
                print(f'[WARN] EGRC-noscar ckpt missing, skip B ablation column: {ck_path}')
                continue
            raise FileNotFoundError(ck_path)
        eck = torch.load(ck_path, map_location='cpu')
        m = EGRCCorrector(d=128, n_organs=n_organs, n_heads=4).to(device)
        # ACDC 适配: ckpt 里 organ_gate.weight 是 WORD 尺寸 (13,1), 模型是 ACDC 尺寸 (3,1) -> 跳过
        src = eck['model_state_dict']; dst = m.state_dict()
        filtered = {k: v for k, v in src.items() if k in dst and v.shape == dst[k].shape}
        missing = [k for k in dst if k not in filtered]
        m.load_state_dict(filtered, strict=False)
        if missing:
            print(f'[WARN] EGRC-{tag}: skipped size-mismatched keys: {missing}')
        m.eval()
        egrc_models[tag] = m
        print(f'[models] EGRC-{tag} loaded from ep{eck.get("epoch", "?")} '
              f'(val_dice={eck.get("val_dice", float("nan")):.4f})')

    # ---- data ----
    ds = SACFDataset(args.data_dir, args.split_json, args.phase,
                     args.image_size, min_organ_area=args.min_organ_area)
    # CPU 并行分片: 只保留 datalist[start:end] 这一段
    if args.start or args.end:
        n_full = len(ds.datalist)
        seg_end = args.end if args.end else n_full
        ds.datalist = ds.datalist[args.start:seg_end]
        print(f'[shard] datalist[{args.start}:{seg_end}] of {n_full} -> '
              f'{len(ds.datalist)} slices')
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    C = ds.num_organs
    datalist_va = ds_json[args.phase]

    per = defaultdict(lambda: defaultdict(list))   # organ -> method -> [metrics dicts]
    n_all = 0
    rows = []                                      # 逐切片记录 (JSONL 合并用)

    for bi, batch in enumerate(loader):
        images = batch['image'].to(device)
        label = batch['label'].to(device)[0]
        organ_ids = batch['organ_ids'][0].tolist()
        if not organ_ids:
            continue
        N = len(organ_ids)
        H, W = label.shape[-2:]

        img_embed = imis.image_encoder(images)                    # (1,768,32,32)
        pooled = img_embed[0].mean(dim=(1, 2))
        pooled = (pooled / pooled.norm()).cpu()

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

            # ---- base_only: 只统计 Base 一列, 不算 SACF/Refiner/EGRC ----
            if args.base_only:
                for i, c in enumerate(organ_ids):
                    gt_np = label[c].cpu().numpy() > 0.5
                    m = seg_metrics(pred_prob[i, 0].detach().cpu().numpy() > 0.5, gt_np)
                    per[c]['base'].append(m)
                    rows.append(dict(slice=args.start + bi, organ=ds.classes[c],
                                     method='base', **m))
                    n_all += 1
                if bi % 20 == 0:
                    print(f'  [slice {bi}/{len(loader)}] n={n_all}', flush=True)
                continue

            pred_ctx = torch.zeros(1, C, H, W, device=device)
            for i, c in enumerate(organ_ids):
                pred_ctx[0, c] = pred_prob[i, 0]
            sacf_text = encode_structured_text(
                imis, [get_anatomy_prompt(n) for n in names])

            # ---- 预测布局 (GT-free 检索键) ----
            lay_pred = np.zeros(C, dtype=np.float32)
            for c in range(C):
                lay_pred[c] = float(pred_ctx[0, c].float().mean())

            for i, c in enumerate(organ_ids):
                context = pred_ctx.clone()
                context[0, c] = 0.0
                sacf_out = sacf(img_embed, pred_prob[i:i + 1], context,
                                sacf_text[i:i + 1])
                field = sacf_out['fused'].float()
                r_logit = refiner(img_embed, pred_prob[i:i + 1], field,
                                  sacf_text[i:i + 1])
                refined = torch.sigmoid(r_logit.float())

                # ---- EGRC (方案2: 仅少样本器官修正, 其余直通) ----
                fname = ds.classes[c]
                if fname not in FEWSHOT_NAMES:
                    # 非少样本器官: B=Base, A+B=SCAR (直通, 与训练协议一致)
                    prob_b = pred_prob[i, 0]
                    prob_ab = refined[0, 0]
                else:
                    exs = retrieve_topk(pooled, lay_pred, organ_slices.get(c, []),
                                        emb_tr, lay_tr, args.K)
                    em_list = [ex_store[es]['gt'][ex_store[es]['pos'][c]].float()
                               for es in exs]
                    zero256 = torch.zeros(1, 1, 256, 256, device=device)
                    octx = torch.tensor([c], device=device)

                    def run_egrc(model, pred512, field_in):
                        """单次 EGRC 修正: exemplar 对齐到当前 pred 质心, residual 加回."""
                        pred256 = F.interpolate(pred512, size=(256, 256),
                                                 mode='bilinear',
                                                 align_corners=False)[0, 0].cpu()
                        exs_m = [align_exemplar(em, pred256) for em in em_list]
                        exm = torch.stack(
                            (exs_m + [torch.zeros(256, 256)] * args.K)[:args.K])
                        with amp.autocast():
                            res = model(img_embed, pred512, field_in,
                                        exm[None, :, None].to(device), octx)
                        res512 = F.interpolate(res.float(), size=(H, W),
                                                mode='bilinear', align_corners=False)
                        return torch.sigmoid(logit(pred512.float()) + res512)

                    has_ns = 'noscar' in egrc_models
                    b1 = run_egrc(egrc_models['noscar'], pred_prob[i:i + 1],
                                  zero256) if has_ns else None
                    a1 = run_egrc(egrc_models['scar'], refined, field)
                    if args.two_stage:
                        # 二阶段: 一阶段输出作为更可靠的对齐锚点与修正基线
                        if has_ns:
                            b1 = run_egrc(egrc_models['noscar'], b1, zero256)
                        a1 = run_egrc(egrc_models['scar'], a1, field)
                    prob_b = b1[0, 0] if has_ns else None
                    prob_ab = a1[0, 0]

                gt_np = label[c].cpu().numpy() > 0.5
                recs = [('base', pred_prob[i, 0]),
                        ('scar', refined[0, 0]),
                        ('ab', prob_ab)]
                if prob_b is not None:
                    recs.insert(2, ('b', prob_b))
                for mname, prob in recs:
                    m = seg_metrics(prob.detach().cpu().numpy() > 0.5, gt_np)
                    per[c][mname].append(m)
                    rows.append(dict(slice=args.start + bi, organ=fname,
                                     method=mname, **m))
                n_all += 1
        if bi % 50 == 0:
            print(f'  [slice {bi}/{len(loader)}] n={n_all}', flush=True)

    # ---- 逐切片 JSONL (CPU 分片并行时用于合并) ----
    if args.dump_jsonl:
        os.makedirs(os.path.dirname(args.dump_jsonl) or '.', exist_ok=True)
        with open(args.dump_jsonl, 'w', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f'[dump] {args.dump_jsonl} ({len(rows)} rows)', flush=True)

    if args.base_only:
        # Base 复评只需要逐切片记录; 汇总交给 merge_eval_jsonl.py
        print(f'[base_only] done. n_all={n_all}', flush=True)
        if not args.dump_jsonl:
            print('[warn] 未指定 --dump_jsonl, 结果无法合并', flush=True)
        return

    # ---- 汇总: 每方法一张全指标表 (与 eval_segmentation.py 同格式: MEAN + MACRO) ----
    classes = ds.classes
    methods = (['base', 'scar', 'b', 'ab'] if 'noscar' in egrc_models
               else ['base', 'scar', 'ab'])
    KEYS = ['dice', 'iou', 'precision', 'recall', 'hd95', 'assd', 'nsd']
    result = {'n_samples': n_all, 'per_organ': {}, 'mean': {}, 'macro': {},
              'fewshot_mean': {}}
    tables = []          # 每方法一张表 (写入 *_tables.txt)

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
             f'{title}',
             f'{"=" * 100}',
             f'{"organ":<32s} {"n":>4s} {"Dice":>7s} {"IoU":>7s} {"Prec":>7s} '
             f'{"Recall":>7s} {"HD95":>7s} {"ASSD":>7s} {"NSD":>7s}']
        for c in sorted(organ_mean, key=lambda c: -organ_mean[c]['dice']):
            v = organ_mean[c]
            t.append(f'{classes[c]:<32s} {len(per[c][m]):>4d} {v["dice"]:>7.3f} '
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
            result['per_organ'].setdefault(classes[c], {'n': len(per[c][m])})[
                m] = organ_mean[c]
        result['mean'][m] = micro
        result['macro'][m] = macro

    # 少样本子集 micro 均值 (Dice 对比用)
    fm = {}
    for m in methods:
        allm = [x for c in per if classes[c] in FEWSHOT_NAMES for x in per[c][m]]
        fm[m] = {k: float(np.nanmean([x[k] for x in allm])) for k in KEYS}
    result['fewshot_mean'] = fm

    # 对比表 (Dice 四列)
    has_b = 'b' in methods
    if has_b:
        cmp_lines = [f'{"organ":<32s} {"n":>4s}  {"Base":>7s} {"SCAR":>7s} {"B":>7s} '
                     f'{"A+B":>7s}  {"Δ(B-Base)":>10s} {"Δ(AB-SCAR)":>10s}']
    else:
        cmp_lines = [f'{"organ":<32s} {"n":>4s}  {"Base":>7s} {"SCAR":>7s} '
                     f'{"A+B":>7s}  {"Δ(AB-SCAR)":>10s}']
    for c in sorted(per, key=lambda c: -np.mean([x['dice'] for x in per[c]['base']])):
        db = {m: np.mean([x['dice'] for x in per[c][m]]) for m in methods}
        if has_b:
            cmp_lines.append(f'{classes[c]:<32s} {len(per[c]["b"]):>4d}  '
                             f'{db["base"]:>7.3f} {db["scar"]:>7.3f} {db["b"]:>7.3f} '
                             f'{db["ab"]:>7.3f}  {db["b"] - db["base"]:>+10.3f} '
                             f'{db["ab"] - db["scar"]:>+10.3f}')
        else:
            cmp_lines.append(f'{classes[c]:<32s} {len(per[c]["ab"]):>4d}  '
                             f'{db["base"]:>7.3f} {db["scar"]:>7.3f} '
                             f'{db["ab"]:>7.3f}  {db["ab"] - db["scar"]:>+10.3f}')
    cmp_lines.append('-' * 100)
    if has_b:
        cmp_lines.append(f'{"MEAN":<32s} {n_all:>4d}  '
                         f'{result["mean"]["base"]["dice"]:>7.3f} '
                         f'{result["mean"]["scar"]["dice"]:>7.3f} '
                         f'{result["mean"]["b"]["dice"]:>7.3f} '
                         f'{result["mean"]["ab"]["dice"]:>7.3f}')
    else:
        cmp_lines.append(f'{"MEAN":<32s} {n_all:>4d}  '
                         f'{result["mean"]["base"]["dice"]:>7.3f} '
                         f'{result["mean"]["scar"]["dice"]:>7.3f} '
                         f'{result["mean"]["ab"]["dice"]:>7.3f}')
    mac_b = f'{result["macro"]["b"]["dice"]:>7.3f} ' if has_b else ''
    cmp_lines.append(f'{"MACRO":<32s} {len(classes):>4d}  '
                     f'{result["macro"]["base"]["dice"]:>7.3f} '
                     f'{result["macro"]["scar"]["dice"]:>7.3f} '
                     f'{mac_b}'
                     f'{result["macro"]["ab"]["dice"]:>7.3f}')
    n_fs = sum(len(per[c]['ab']) for c in per if classes[c] in FEWSHOT_NAMES)
    fs_b = f'{fm["b"]["dice"]:>7.3f} ' if has_b else ''
    cmp_lines.append(f'{"FEWSHOT-MEAN":<32s} {n_fs:>4d}  '
                     f'{fm["base"]["dice"]:>7.3f} {fm["scar"]["dice"]:>7.3f} '
                     f'{fs_b}'
                     f'{fm["ab"]["dice"]:>7.3f}')

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
