"""EGRC 积木一: Exemplar Bank 构建 + 检索质量验证.

Step 1: 遍历训练集 (1766 slices), IMISNet encoder 提全局特征
        每器官条目: {slice_idx, organ, img_emb(768d), centroid(2d), area, layout(13d)}
Step 2: 对验证集每个 organ-sample, 检索 Top-K 同器官训练条目
        检索键 = cos_sim(img_emb) + layout_sim(共现布局)
Step 3: 质量判据 — exemplar 器官质心 vs query GT 质心的归一化距离:
        - 检索 top-1 vs 随机 exemplar (基线)
        - 若检索距离显著小 (<0.1 即 ~50px), 可变形对齐传递 mask 可行 → EGRC 机制成立

输出:
  work_dir/egrc/exemplar_bank.pt
  work_dir/egrc/retrieval_quality.txt
"""

import sys as _sys, os as _os
_ROOT = _os.path.abspath(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
_SACF_DEV = _os.path.join(_ROOT, 'sacf_dev')
if _SACF_DEV not in _sys.path:
    _sys.path.insert(0, _SACF_DEV)

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from segment_anything import sam_model_registry
from model import IMISNet
from scar.lib import SACFDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, default='dataset/BTCV_full')
    p.add_argument('--split_json', type=str, default='dataset_split_full_24_6.json')
    p.add_argument('--image_size', type=int, default=512)
    p.add_argument('--sam_checkpoint', type=str, default='ckpt/IMISNet-B.pth')
    p.add_argument('--imis_ckpt', type=str,
                   default='work_dir/ft-IMISNet-full/IMIS_val_dice_best.pth')
    p.add_argument('--model_type', type=str, default='vit_b')
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--min_organ_area', type=int, default=1)
    p.add_argument('--out_dir', type=str, default='work_dir/egrc')
    p.add_argument('--device', type=str, default='cuda')
    return p.parse_args()


def _collate(batch):
    return {
        'image': torch.stack([b['image'] for b in batch]),
        'label': torch.stack([b['label'] for b in batch]),
        'organ_ids': [b['organ_ids'] for b in batch],
    }


@torch.no_grad()
def build_bank(imis, ds, device, args, tag):
    """提一个 split 的全部特征. 返回 entries 列表 (逐 organ) + slice 级信息."""
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True,
                        collate_fn=_collate)
    C = ds.num_organs
    entries = []  # 逐 organ 条目
    slice_info = []  # 逐 slice: (emb_norm(768), layout(13), centroids dict)
    n_slice = 0
    for bi, batch in enumerate(loader):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)  # (B,C,H,W)
        B, _, H, W = labels.shape
        emb = imis.image_encoder(images)  # (B,768,h,w)
        emb_pool = emb.mean(dim=(2, 3))   # (B,768)
        emb_norm = emb_pool / emb_pool.norm(dim=1, keepdim=True)
        for b in range(B):
            lab = labels[b]
            area = lab.sum(dim=(1, 2))  # (C,)
            layout = (area / (H * W)).cpu().numpy()  # 归一化面积布局
            cents = {}
            for c in range(C):
                if area[c] > 0:
                    m = lab[c]
                    ys, xs = torch.nonzero(m, as_tuple=True)
                    cents[c] = (float(ys.float().mean()) / H, float(xs.float().mean()) / W)
            slice_info.append({
                'emb': emb_norm[b].cpu(), 'layout': layout,
                'centroids': cents, 'areas': area.cpu().numpy(),
            })
            for c in range(C):
                if area[c] > 0:
                    entries.append({'slice': n_slice, 'organ': c})
            n_slice += 1
        if bi % 20 == 0:
            print(f'  [{tag}] batch {bi}/{len(loader)}', flush=True)
    return entries, slice_info


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    sam = sam_model_registry[args.model_type](args).to(device)
    imis = IMISNet(sam, test_mode=True,
                   category_weights='dataloaders/categories_weight.pkl').to(device)
    sd = torch.load(args.imis_ckpt, map_location='cpu')
    if isinstance(sd, dict) and 'model_state_dict' in sd:
        sd = sd['model_state_dict']
    imis.load_state_dict(sd, strict=False)
    imis.eval()
    print(f'[IMISNet] loaded {args.imis_ckpt}')

    # ---- train bank ----
    ds_tr = SACFDataset(args.data_dir, args.split_json, 'training',
                        args.image_size, min_organ_area=args.min_organ_area)
    print(f'[data] train={len(ds_tr)} slices')
    entries_tr, info_tr = build_bank(imis, ds_tr, device, args, 'train')

    # ---- val queries ----
    ds_va = SACFDataset(args.data_dir, args.split_json, 'validation',
                        args.image_size, min_organ_area=args.min_organ_area)
    print(f'[data] val={len(ds_va)} slices')
    entries_va, info_va = build_bank(imis, ds_va, device, args, 'val')

    # ---- save bank ----
    torch.save({'entries': entries_tr, 'info': info_tr,
                'classes': ds_tr.classes}, os.path.join(args.out_dir, 'exemplar_bank.pt'))
    print(f'[bank] saved {len(entries_tr)} organ-entries '
          f'from {len(info_tr)} train slices')

    # ---- 检索质量验证 ----
    emb_tr = torch.stack([s['emb'] for s in info_tr])            # (S_tr,768)
    lay_tr = np.stack([s['layout'] for s in info_tr])            # (S_tr,13)
    # organ -> train slice list
    from collections import defaultdict
    organ_slices = defaultdict(list)
    for e in entries_tr:
        organ_slices[e['organ']].append(e['slice'])

    L = []
    def add(s=''):
        L.append(s)
        print(s)

    add('=' * 96)
    add(f'EGRC 检索质量验证 (val organ-samples={len(entries_va)}, '
        f'train bank={len(entries_tr)})')
    add('=' * 96)
    add(f'{"organ":<32s} {"n_val":>5s} {"bank":>5s} {"d_rand":>7s} '
        f'{"d_top1":>7s} {"d_top1@layout":>12s} {"hit@10%":>8s}')
    add('-' * 96)

    rand_d_all, top1_d_all, top1L_d_all = [], [], []
    per_organ = {}
    for e in entries_va:
        c = e['organ']
        qs = e['slice']
        if c not in organ_slices or c not in info_va[qs]['centroids']:
            continue
        cands = organ_slices[c]
        q_cent = info_va[qs]['centroids'][c]
        q_emb = info_va[qs]['emb']
        q_lay = info_va[qs]['layout']

        # 全部候选的距离
        cand_t = torch.tensor(cands)
        sims = (emb_tr[cand_t] @ q_emb)                       # (n,)
        # 布局相似: 高斯核 on L2
        lay_d = np.linalg.norm(lay_tr[cands] - q_lay[None], axis=1)
        lay_sim = np.exp(-lay_d / (lay_d.mean() + 1e-6))

        # 质心距离 (归一化坐标)
        cents = np.array([info_tr[s]['centroids'].get(c, (np.nan, np.nan))
                         for s in cands])
        d = np.linalg.norm(cents - np.array(q_cent)[None], axis=1)

        # top-1 by emb only
        top1 = int(np.argmax(sims.numpy()))
        d_top1 = float(d[top1])
        # top-1 by emb + layout
        score2 = sims.numpy() + lay_sim
        top1L = int(np.argmax(score2))
        d_top1L = float(d[top1L])

        d_rand = float(np.mean(d))  # 随机期望 = 均值

        rec = per_organ.setdefault(c, {'n': 0, 'rand': [], 'top1': [], 'top1L': []})
        rec['n'] += 1
        rec['rand'].append(d_rand)
        rec['top1'].append(d_top1)
        rec['top1L'].append(d_top1L)
        rand_d_all.append(d_rand); top1_d_all.append(d_top1); top1L_d_all.append(d_top1L)

    hit_all = 0
    n_all = 0
    for c in sorted(per_organ, key=lambda k: -per_organ[k]['n']):
        r = per_organ[c]
        dr = np.median(r['rand'])
        dt = np.median(r['top1'])
        dl = np.median(r['top1L'])
        hit = np.mean([1.0 if x < 0.1 else 0.0 for x in r['top1']])
        hit_all += hit * r['n']
        n_all += r['n']
        add(f'{ds_tr.classes[c]:<32s} {r["n"]:>5d} {len(organ_slices[c]):>5d} '
            f'{dr:>7.3f} {dt:>7.3f} {dl:>12.3f} {100*hit:>7.1f}%')
    add('-' * 96)
    add(f'{"MEAN(median)":<32s} {n_all:>5d} {"":>5s} '
        f'{np.median(rand_d_all):>7.3f} {np.median(top1_d_all):>7.3f} '
        f'{np.median(top1L_d_all):>12.3f} {100*hit_all/n_all:>7.1f}%')
    add('')
    add(f'总体: 检索 top-1 质心距离 中位数={np.median(top1_d_all):.3f} '
        f'vs 随机={np.median(rand_d_all):.3f} '
        f'(改善 {np.median(rand_d_all)/max(np.median(top1_d_all),1e-6):.1f}x)')
    add(f'emb+layout 检索 中位数={np.median(top1L_d_all):.3f}')
    add(f'质心距离<0.1 (≈51px) 的比例 = {100*hit_all/n_all:.1f}%  '
        f'← 可对齐传递 mask 的样本比例')

    out_txt = os.path.join(args.out_dir, 'retrieval_quality.txt')
    with open(out_txt, 'w') as f:
        f.write('\n'.join(L))
    print(f'\n[report] {out_txt}')


if __name__ == '__main__':
    main()
