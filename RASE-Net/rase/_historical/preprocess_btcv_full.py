# -*- coding: utf-8 -*-
"""全量预处理：30例完整volume -> 2D切片（STEP=1，全切片，仅过滤无标注层）
输出: dataset/BTCV_full/{image,label,imask} + dataset.json + dataset_split_full_24_6.json
划分: 24 train / 6 val (val=ABD_035..040, 与社区标准一致)
"""
import os, json
import numpy as np
import nibabel as nib
from PIL import Image
from scipy import sparse
from scipy.ndimage import label as cc_label

ABD_BASE = ".//abdomen"
BTCV_BASE = ".//dataset/BTCV_full"
IMG_DIR = os.path.join(BTCV_BASE, "image")
LBL_DIR = os.path.join(BTCV_BASE, "label")
MSK_DIR = os.path.join(BTCV_BASE, "imask")

NUM_CLASSES = 13
SHAPE_TUPLE = (13, 512, 512, 1)
MIN_NONZERO_PIXELS = 100   # 过滤几乎无器官的层面

CASE_NUMS = list(range(1, 11)) + list(range(21, 41))   # 30例
VAL_CASES = {35, 36, 37, 38, 39, 40}

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LBL_DIR, exist_ok=True)
os.makedirs(MSK_DIR, exist_ok=True)

def window_and_save_png(raw_slice, out_path):
    lo, hi = raw_slice.min(), raw_slice.max()
    if hi - lo < 1e-6:
        scaled = np.zeros_like(raw_slice, dtype=np.uint8)
    else:
        scaled = np.clip((raw_slice - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    scaled = scaled.T
    rgb = np.stack([scaled]*3, axis=-1)
    Image.fromarray(rgb).save(out_path)

def label_to_onehot_sparse(label_slice, out_path):
    label_T = label_slice.T.astype(np.int32)
    out = np.zeros(SHAPE_TUPLE, dtype=np.int32)
    for c in range(1, NUM_CLASSES+1):
        out[c-1, :, :, 0] = (label_T == c).astype(np.int32)
    flat = out.reshape(-1)
    sp = sparse.csr_matrix(flat)
    sparse.save_npz(out_path, sp)

def make_instance_mask(label_slice):
    label_T = label_slice.T.astype(np.int32)
    inst = np.zeros(label_T.shape, dtype=np.int64)
    nid = 1
    for c in range(1, NUM_CLASSES+1):
        om = (label_T == c)
        if om.sum() == 0:
            continue
        lab, n = cc_label(om)
        for i in range(1, n+1):
            cm = (lab == i)
            if cm.sum() < 5:
                continue
            inst[cm] = nid
            nid += 1
    return inst[:, :, None]

def process_case(num):
    case = "ABD_%03d" % num
    ip = os.path.join(ABD_BASE, "imagesTr", "img%04d.nii.gz" % num)
    lp = os.path.join(ABD_BASE, "labelsTr", "label%04d.nii.gz" % num)
    if not (os.path.exists(ip) and os.path.exists(lp)):
        print("[SKIP]", case); return []
    img_vol = nib.load(ip).get_fdata()
    lbl_vol = nib.load(lp).get_fdata().astype(np.int32)
    D = img_vol.shape[2]
    entries = []
    for z in range(D):
        if np.count_nonzero(lbl_vol[:, :, z]) < MIN_NONZERO_PIXELS:
            continue
        img_f = "%s_%d.png" % (case, z)
        lbl_f = "x___%s_%d.%s.npz" % (case, z, str(SHAPE_TUPLE))
        msk_f = "x___%s_%d.npy" % (case, z)
        window_and_save_png(img_vol[:, :, z], os.path.join(IMG_DIR, img_f))
        label_to_onehot_sparse(lbl_vol[:, :, z], os.path.join(LBL_DIR, lbl_f))
        inst = make_instance_mask(lbl_vol[:, :, z])
        np.save(os.path.join(MSK_DIR, msk_f), inst.astype(np.float32))
        entries.append({"image": "image/"+img_f, "label": "label/"+lbl_f, "imask": "imask/"+msk_f})
    print("%s: %d/%d slices kept" % (case, len(entries), D), flush=True)
    return entries

if __name__ == "__main__":
    all_train, all_val = [], []
    for n in CASE_NUMS:
        e = process_case(n)
        if n in VAL_CASES:
            all_val.extend(e)
        else:
            all_train.extend(e)
    dj = {"training": all_train + all_val}
    with open(os.path.join(BTCV_BASE, "dataset.json"), "w") as f:
        json.dump(dj, f)
    sp = {
        "name": "BTCV_full",
        "description": "30 cases, ALL slices (step=1, >=100 labeled px), 24 train / 6 val",
        "split_note": "full-volume 24/6, val=ABD_035_040 (community standard cases)",
        "numTraining": len(all_train), "numValidation": len(all_val), "numTest": 0,
        "training": all_train, "validation": all_val, "test": [],
    }
    with open(os.path.join(BTCV_BASE, "dataset_split_full_24_6.json"), "w") as f:
        json.dump(sp, f)
    print("TOTAL: train=%d val=%d all=%d" % (len(all_train), len(all_val), len(all_train)+len(all_val)))
