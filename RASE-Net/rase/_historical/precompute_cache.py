"""EGRC 积木三: 预计算缓存.

对 train + val 每个 slice 跑 IMISNet → SACF → FieldGuidedRefiner 前向,
保存 (256×256 fp16):
  base:    IMISNet sigmoid 概率图
  field:   SACF fused 矛盾场
  refined: Refiner sigmoid 概率图
  gt:      二值 GT
  emb:     img embedding 768×64×64 (fp16)  [仅训练集, 省得重算]

用途: EGRC 校正器训练/评估的数据底座, 避免每 epoch 重跑冻结主干.
输出: work_dir/egrc/cache/{train,val}/%05d.pt + manifest.json
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
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda import amp

from segment_anything import sam_model_registry
from model import IMISNet
from dataloaders.data_utils import get_bboxes_from_mask
from scar.lib import (SemanticContradictionField, FieldGuidedRefiner, SACFDataset,
                      get_anatomy_prompt, encode_structured_text)


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
    p.add_argument('--model_type', type=str, default='vit_b')
    p.add_argument('--dim', type=int, default=192)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--min_organ_area', type=int, default=1)
    p.add_argument('--out_dir', type=str, default='work_dir/egrc/cache')
    p.add_argument('--device', type=str, default='cuda')
    return p.parse_args()


@torch.no_grad()
def run_split(args, split, imis, sacf, refiner, device):
    ds = SACFDataset(args.data_dir, args.split_json, split,
                     args.image_size, min_organ_area=args.min_organ_area)
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    C = ds.num_organs
    out_root = os.path.join(args.out_dir, split)
    os.makedirs(out_root, exist_ok=True)
    manifest = []

    for bi, batch in enumerate(loader):
        images = batch['image'].to(device)
        label = batch['label'].to(device)[0]
        organ_ids = batch['organ_ids'][0].tolist()
        if not organ_ids:
            manifest.append({'idx': bi, 'organs': []})
            continue
        out_path = os.path.join(out_root, f'{bi:05d}.pt')
        if os.path.exists(out_path):
            manifest.append({'idx': bi, 'organs': organ_ids})
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
            sacf_text = encode_structured_text(
                imis, [get_anatomy_prompt(n) for n in names])

            bases, fields, refineds = [], [], []
            for i, c in enumerate(organ_ids):
                context = pred_ctx.clone()
                context[0, c] = 0.0
                sacf_out = sacf(img_embed, pred_prob[i:i + 1], context,
                                sacf_text[i:i + 1])
                logit = refiner(img_embed, pred_prob[i:i + 1],
                                sacf_out['fused'].float(), sacf_text[i:i + 1])
                refined = torch.sigmoid(logit.float())
                bases.append(pred_prob[i, 0])
                fields.append(sacf_out['fused'].float()[0, 0])
                refineds.append(refined[0, 0])

        def _down(t):
            return F.interpolate(t[None, None], size=(256, 256),
                                 mode='bilinear', align_corners=False)[0, 0].half()

        gts = torch.stack([_down(label[c]) for c in organ_ids])
        data = {
            'organs': torch.tensor(organ_ids, dtype=torch.long),
            'base': torch.stack([_down(b) for b in bases]),
            'field': torch.stack([_down(f) for f in fields]),
            'refined': torch.stack([_down(r) for r in refineds]),
            'gt': gts,
        }
        # emb 不缓存 (磁盘紧张), 训练时 encoder 在线重算 (~2min/epoch)
        torch.save(data, out_path)
        manifest.append({'idx': bi, 'organs': organ_ids})
        if bi % 100 == 0:
            print(f'  [{split}] {bi}/{len(loader)}', flush=True)

    with open(os.path.join(out_root, 'manifest.json'), 'w') as f:
        json.dump(manifest, f)
    print(f'[{split}] done: {len(manifest)} slices')


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

    ckpt = torch.load(args.sacf_ckpt, map_location='cpu')
    # num_organs 由 SACF 权重形状直接推断 (context_encoder 首层输入通道), 避免硬编码
    _nv = ckpt['sacf_state_dict']['context_encoder.net.0.weight'].shape[1]
    sacf = SemanticContradictionField(num_organs=_nv, image_embed_dim=768,
                                      text_embed_dim=768, dim=args.dim,
                                      image_size=args.image_size).to(device)
    sacf.load_state_dict(ckpt['sacf_state_dict'])
    sacf.eval()

    rck = torch.load(args.refiner_ckpt, map_location='cpu')
    refiner = FieldGuidedRefiner(image_embed_dim=768, text_embed_dim=768,
                                 dim=args.dim).to(device)
    refiner.load_state_dict(rck['refiner_state_dict'])
    refiner.eval()
    print('[models] all loaded (frozen)')

    run_split(args, 'training', imis, sacf, refiner, device)
    run_split(args, 'validation', imis, sacf, refiner, device)
    # ACDC 需要独立 test 划分的缓存用于最终评估; WORD 的 split json 无 test 键, 自动跳过
    _sj = json.load(open(os.path.join(args.data_dir, args.split_json)))
    if _sj.get('test'):
        run_split(args, 'test', imis, sacf, refiner, device)


if __name__ == '__main__':
    main()
