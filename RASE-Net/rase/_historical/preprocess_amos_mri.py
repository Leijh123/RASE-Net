#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AMOS22 MRI (60 例带标签) -> IMIS-Bench 数据集格式。

产物与 dataset/ACDC_full、WORD_full 完全一致:
    <out>/image/{case}_{z}.png                       # 512x512, 每层 min-max -> RGB
    <out>/label/x___{case}_{z}.(15, 512, 512, 1).npz # 稀疏 one-hot
    <out>/imask/x___{case}_{z}.npy                   # (512,512,1) 逐类连通域实例 id
    <out>/dataset.json
    <out>/dataset_split_amos.json                    # 40 train / 20 test (零样本跨模态)
    <out>/dataset_split_amos_ft.json                 # 32 / 8 / 20 (微调+早停)
    <out>/dataset_split_amos_{5,10,20}.json          # 少样本子集

类别顺序 = AMOS 官方 15 类, 通道 c-1 对应官方标签 c。
"""
import argparse
import json
import os
import random

import numpy as np
import nibabel as nib
from PIL import Image
from scipy import sparse
from scipy.ndimage import label as cc_label

NUM_CLASSES = 15
SHAPE_TUPLE = (NUM_CLASSES, 512, 512, 1)
TARGET = 512

LABELS = {
    "0": "background", "1": "spleen", "2": "kidney_right", "3": "kidney_left",
    "4": "gallbladder", "5": "esophagus", "6": "liver", "7": "stomach",
    "8": "aorta", "9": "inferior_vena_cava", "10": "pancreas",
    "11": "adrenal_gland_right", "12": "adrenal_gland_left",
    "13": "duodenum", "14": "bladder", "15": "prostate_uterus",
}


def save_png(raw_slice, out_path):
    lo, hi = float(raw_slice.min()), float(raw_slice.max())
    if hi - lo < 1e-6:
        scaled = np.zeros(raw_slice.shape, dtype=np.uint8)
    else:
        scaled = np.clip((raw_slice - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    img = Image.fromarray(scaled).resize((TARGET, TARGET), Image.BILINEAR)
    Image.fromarray(np.stack([np.array(img)] * 3, axis=-1)).save(out_path)


def to_onehot_sparse(sem, out_path):
    out = np.zeros(SHAPE_TUPLE, dtype=np.int32)
    for c in range(1, NUM_CLASSES + 1):
        out[c - 1, :, :, 0] = (sem == c).astype(np.int32)
    sparse.save_npz(out_path, sparse.csr_matrix(out.reshape(-1)))


def make_instance_mask(sem):
    inst = np.zeros(sem.shape, dtype=np.int64)
    nid = 1
    for c in range(1, NUM_CLASSES + 1):
        om = (sem == c)
        if om.sum() == 0:
            continue
        lab, n = cc_label(om)
        for i in range(1, n + 1):
            cm = (lab == i)
            if cm.sum() < 5:
                continue
            inst[cm] = nid
            nid += 1
    return inst[:, :, None].astype(np.float32)


def process_case(case, img_p, lbl_p, img_dir, lbl_dir, msk_dir, skip_existing):
    img_vol = np.asarray(nib.load(img_p).get_fdata(), dtype=np.float32)
    lbl_vol = np.asarray(nib.load(lbl_p).get_fdata(), dtype=np.int16)
    if img_vol.shape != lbl_vol.shape:
        print(f"  [warn] {case} 形状不一致 {img_vol.shape} vs {lbl_vol.shape}, 跳过")
        return []
    entries = []
    for z in range(lbl_vol.shape[2]):
        sem = lbl_vol[:, :, z]
        if np.count_nonzero(sem) == 0:
            continue
        img_f = f"{case}_{z}.png"
        lbl_f = f"x___{case}_{z}.{SHAPE_TUPLE}.npz"
        msk_f = f"x___{case}_{z}.npy"
        ip = os.path.join(img_dir, img_f)
        lp = os.path.join(lbl_dir, lbl_f)
        mp = os.path.join(msk_dir, msk_f)
        if not (skip_existing and os.path.exists(ip) and os.path.exists(lp) and os.path.exists(mp)):
            save_png(img_vol[:, :, z], ip)
            sem_r = np.array(Image.fromarray(sem.astype(np.uint8))
                             .resize((TARGET, TARGET), Image.NEAREST)).astype(np.int16)
            to_onehot_sparse(sem_r, lp)
            np.save(mp, make_instance_mask(sem_r))
        entries.append({"image": f"image/{img_f}", "label": f"label/{lbl_f}",
                        "imask": f"imask/{msk_f}"})
    print(f"  {case}: {len(entries)}/{lbl_vol.shape[2]} slices kept", flush=True)
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/amos22_mri")
    ap.add_argument("--out", default="data/AMOS_MRI_full")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ft-val", type=int, default=8, help="从 40 例训练里划多少例做验证")
    args = ap.parse_args()

    img_dir = os.path.join(args.out, "image")
    lbl_dir = os.path.join(args.out, "label")
    msk_dir = os.path.join(args.out, "imask")
    for d in (img_dir, lbl_dir, msk_dir):
        os.makedirs(d, exist_ok=True)

    train_cases = sorted(f[:-7] for f in os.listdir(os.path.join(args.root, "train/imagesTr"))
                         if f.endswith(".nii.gz"))
    test_cases = sorted(f[:-7] for f in os.listdir(os.path.join(args.root, "valid/imagesVa"))
                        if f.endswith(".nii.gz"))
    print(f"[scan] train={len(train_cases)} test={len(test_cases)}")

    def collect(cases, sub):
        out = {}
        for c in cases:
            ip = os.path.join(args.root, sub, "images" + ("Tr" if sub == "train" else "Va"), c + ".nii.gz")
            lp = os.path.join(args.root, sub, "labels" + ("Tr" if sub == "train" else "Va"), c + ".nii.gz")
            if not (os.path.isfile(ip) and os.path.isfile(lp)):
                print(f"  [warn] 缺文件 {c}, 跳过")
                continue
            out[c] = process_case(c, ip, lp, img_dir, lbl_dir, msk_dir, args.skip_existing)
        return out

    tr = collect(train_cases, "train")
    te = collect(test_cases, "valid")

    def flat(d, keys):
        return [e for k in keys for e in d.get(k, [])]

    all_tr_keys = sorted(tr)
    rng = random.Random(args.seed)
    shuffled = all_tr_keys[:]
    rng.shuffle(shuffled)
    val_keys = sorted(shuffled[:args.ft_val])
    ft_tr_keys = sorted(shuffled[args.ft_val:])
    te_keys = sorted(te)

    def dump(name, note, tr_keys, va_keys, te_keys_):
        obj = {
            "name": "AMOS_MRI_full",
            "description": f"AMOS22 MRI labeled cases ({len(all_tr_keys)+len(te_keys)}), "
                           f"fg>0 slices, {TARGET}x{TARGET}, 15 organs",
            "dimension": "2D", "modality": {"0": "mr"}, "labels": LABELS,
            "split_note": note,
            "numTraining": len(flat(tr, tr_keys)),
            "numValidation": len(flat(tr, va_keys)) if va_keys else 0,
            "numTest": len(flat(te, te_keys_)),
            "training": flat(tr, tr_keys),
            "validation": flat(tr, va_keys) if va_keys else [],
            "test": flat(te, te_keys_),
        }
        with open(os.path.join(args.out, name), "w") as f:
            json.dump(obj, f, indent=1)
        print(f"[saved] {name}: train={obj['numTraining']} val={obj['numValidation']} "
              f"test={obj['numTest']}")

    dump("dataset_split_amos.json",
         "official labeled split: 40 train cases (Tr), 20 held-out test cases (Va); zero-shot cross-modal",
         all_tr_keys, [], te_keys)
    dump("dataset_split_amos_ft.json",
         f"fine-tune split: {len(ft_tr_keys)} train / {len(val_keys)} val / {len(te_keys)} test (seed={args.seed})",
         ft_tr_keys, val_keys, te_keys)
    for k in (5, 10, 20):
        if k <= len(ft_tr_keys):
            sub = sorted(random.Random(args.seed + k).sample(ft_tr_keys, k))
            dump(f"dataset_split_amos_{k}.json",
                 f"{k}-shot MRI fine-tune, {len(val_keys)} MRI val, {len(te_keys)} MRI test",
                 sub, val_keys, te_keys)

    with open(os.path.join(args.out, "dataset.json"), "w") as f:
        json.dump({"name": "AMOS_MRI_full", "description": "AMOS22 MRI labeled, all fg slices",
                   "dimension": "2D", "modality": {"0": "mr"}, "labels": LABELS,
                   "numTraining": len(flat(tr, all_tr_keys)),
                   "training": flat(tr, all_tr_keys)}, f, indent=1)


if __name__ == "__main__":
    main()
