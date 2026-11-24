"""FieldGuidedRefiner 训练: 闭环 "SACF 检测 -> 精修 mask".

训练流程 (IMISNet 与 SACF 全程冻结, 只训练 refiner):
  1. 冻结 IMISNet 用 GT bbox prompt 预测 mask (与基准评估同协议)
  2. 冻结 SACF 对预测计算矛盾场 C(x) (context=其他器官预测, GT-free)
  3. Refiner(image_embed, pred, C(x), text) -> 修正 logit
  4. L = BCE + Dice vs GT

消融: --no_field 训练同结构 refiner 但置零矛盾场 (证明场信息的贡献).

用法:
  python train_refiner.py                # 场指导精修
  python train_refiner.py --no_field     # 消融: 无场
"""

import argparse
import os
import time
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda import amp

from segment_anything import sam_model_registry
from model import IMISNet
from dataloaders.data_utils import get_bboxes_from_mask
from sacf import (SemanticContradictionField, FieldGuidedRefiner, SACFDataset,
                  sacf_collate_fn, get_anatomy_prompt, encode_structured_text)


def parse_args():
    p = argparse.ArgumentParser()
    # 与基准 (ft-IMISNet-full) 对齐
    p.add_argument('--data_dir', type=str, default='dataset/BTCV_full')
    p.add_argument('--split_json', type=str, default='dataset_split_full_24_6.json')
    p.add_argument('--image_size', type=int, default=512)
    p.add_argument('--sam_checkpoint', type=str, default='ckpt/IMISNet-B.pth')
    p.add_argument('--imis_ckpt', type=str,
                   default='work_dir/ft-IMISNet-full/IMIS_val_dice_best.pth')
    p.add_argument('--sacf_ckpt', type=str, default='work_dir/sacf/sacf_best.pth')
    p.add_argument('--model_type', type=str, default='vit_b')
    p.add_argument('--work_dir', type=str, default='work_dir/refiner')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--organs_per_slice', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--dim', type=int, default=192)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--no_field', action='store_true',
                   help='消融: 置零矛盾场, 同结构 refiner')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--patience', type=int, default=0,
                   help='早停: val refined_dice 连续 N 轮无提升则停止; 0 表示不早停')
    return p.parse_args()


def build_imis(args):
    sam = sam_model_registry[args.model_type](args).to(args.device)
    imis = IMISNet(sam, test_mode=True,
                   category_weights='dataloaders/categories_weight.pkl').to(args.device)
    if args.imis_ckpt and os.path.exists(args.imis_ckpt):
        sd = torch.load(args.imis_ckpt, map_location='cpu')
        if isinstance(sd, dict) and 'model_state_dict' in sd:
            sd = sd['model_state_dict']
        missing, unexpected = imis.load_state_dict(sd, strict=False)
        n_loaded = len(sd) - len(unexpected)
        print(f'[IMISNet] finetuned weights loaded (loaded={n_loaded}/{len(sd)})')
        assert n_loaded > 400
    imis.eval()
    for p_ in imis.parameters():
        p_.requires_grad = False
    return imis


def build_sacf(args):
    ckpt = torch.load(args.sacf_ckpt, map_location='cpu')
    # num_organs 由 SACF 权重形状直接推断 (context_encoder 首层输入通道), 避免硬编码
    _nv = ckpt['sacf_state_dict']['context_encoder.net.0.weight'].shape[1]
    sacf = SemanticContradictionField(num_organs=_nv, image_embed_dim=768,
                                      text_embed_dim=768, dim=args.dim,
                                      image_size=args.image_size).to(args.device)
    sacf.load_state_dict(ckpt['sacf_state_dict'])
    sacf.eval()
    for p_ in sacf.parameters():
        p_.requires_grad = False
    print(f'[SACF] loaded {args.sacf_ckpt} (epoch {ckpt.get("epoch", "?")})')
    return sacf


@torch.no_grad()
def predict_and_field(args, imis, sacf, images, labels, organ_ch, slice_idx, classes):
    """冻结前向: IMISNet 预测 + SACF 矛盾场.

    返回 (img_embed_per_organ, pred_prob, field, text_emb)
    """
    device = args.device
    img_embed = imis.image_encoder(images)                     # (B,768,h,w)
    N = organ_ch.shape[0]
    names = [classes[c] for c in organ_ch.tolist()]

    # GT bbox prompt (与基准评估同协议)
    bboxes = torch.cat([
        get_bboxes_from_mask(labels[bi, ch:ch + 1].unsqueeze(0), offset=0).to(device)
        for bi, ch in zip(slice_idx.tolist(), organ_ch.tolist())], dim=0)
    text = imis.process_text_prompt(names)
    prompts = {'bboxes': bboxes}
    prompts.update(text)

    embed_rep = img_embed[slice_idx]                           # (N,768,h,w)
    outputs = imis.forward_decoder(embed_rep, prompts)
    pred_prob = torch.sigmoid(outputs['masks'].float())        # (N,1,H,W)

    # context = 其他器官预测 (GT-free)
    B, C = labels.shape[0], labels.shape[1]
    pred_ctx = torch.zeros(B, C, *labels.shape[-2:], device=device)
    for i in range(N):
        pred_ctx[slice_idx[i], organ_ch[i]] = pred_prob[i, 0]
    context = pred_ctx[slice_idx].clone()
    context[torch.arange(N, device=device), organ_ch] = 0.0

    # SACF 评判文本: 结构化解剖描述 (必改1, 与 SACF 训练一致);
    # decoder 分割 prompt 仍为器官名 (基准协议)
    sacf_text = encode_structured_text(
        imis, [get_anatomy_prompt(n) for n in names])          # (N,768)
    sacf_out = sacf(embed_rep, pred_prob, context, sacf_text)
    field = sacf_out['fused'].float()                          # (N,1,H,W)
    return embed_rep, pred_prob, field, sacf_text


def dice_loss(pred_logit, target):
    p = torch.sigmoid(pred_logit)
    inter = (p * target).sum(dim=(1, 2, 3))
    union = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return 1.0 - (2 * inter / (union + 1e-6)).mean()


def batch_loss(logit, gt):
    bce = F.binary_cross_entropy_with_logits(logit, gt)
    return bce + dice_loss(logit, gt)


@torch.no_grad()
def run_val(args, imis, sacf, refiner, val_loader, device):
    refiner.eval()
    base_dices, ref_dices = [], []
    for batch in val_loader:
        images = batch['images'].to(device)
        labels = batch['labels'].to(device)
        organ_ch = batch['organ_ch'].to(device)
        slice_idx = batch['slice_idx'].to(device)
        masks = batch['masks'].to(device)
        if masks.shape[0] == 0:
            continue
        embed_rep, pred_prob, field, t_emb = predict_and_field(
            args, imis, sacf, images, labels, organ_ch, slice_idx,
            val_loader.dataset.classes)
        f_in = torch.zeros_like(field) if args.no_field else field
        with amp.autocast():
            logit = refiner(embed_rep, pred_prob, f_in, t_emb)
        ref = (torch.sigmoid(logit.float()) > 0.5).float()
        for i in range(masks.shape[0]):
            base_dices.append(_dice(pred_prob[i], masks[i]))
            ref_dices.append(_dice(ref[i], masks[i]))
    refiner.train()
    return float(np.mean(base_dices)), float(np.mean(ref_dices))


def _dice(a, b):
    inter = (a * b).sum()
    return (2 * inter / (a.sum() + b.sum() + 1e-6)).item()


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.work_dir, exist_ok=True)
    tag = 'no_field' if args.no_field else 'field'
    print(f'[{time.strftime("%H:%M:%S")}] args: {vars(args)}')

    imis = build_imis(args)
    sacf = build_sacf(args)
    refiner = FieldGuidedRefiner(image_embed_dim=768, text_embed_dim=768,
                                 dim=args.dim).to(device)
    n_params = sum(p_.numel() for p_ in refiner.parameters() if p_.requires_grad)
    print(f'[refiner] trainable params: {n_params / 1e6:.2f}M')

    min_area = int(16 * (args.image_size / 256.0) ** 2)
    train_ds = SACFDataset(args.data_dir, args.split_json, 'training',
                           args.image_size, min_organ_area=min_area)
    val_ds = SACFDataset(args.data_dir, args.split_json, 'validation',
                         args.image_size, min_organ_area=min_area)
    collate = partial(sacf_collate_fn, organs_per_slice=args.organs_per_slice)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              collate_fn=collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            collate_fn=collate)
    print(f'[data] train={len(train_ds)} slices, val={len(val_ds)} slices')

    opt = torch.optim.AdamW(refiner.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * len(train_loader))

    log_path = os.path.join(args.work_dir, f'train_refiner_{tag}.log')
    log_f = open(log_path, 'a')
    best_dice = -1.0
    bad = 0

    for ep in range(args.epochs):
        refiner.train()
        t0, losses = time.time(), []
        for it, batch in enumerate(train_loader):
            images = batch['images'].to(device)
            labels = batch['labels'].to(device)
            organ_ch = batch['organ_ch'].to(device)
            slice_idx = batch['slice_idx'].to(device)
            masks = batch['masks'].to(device)
            if masks.shape[0] == 0:
                continue

            with torch.no_grad():
                embed_rep, pred_prob, field, t_emb = predict_and_field(
                    args, imis, sacf, images, labels, organ_ch, slice_idx,
                    train_loader.dataset.classes)
            f_in = torch.zeros_like(field) if args.no_field else field

            with amp.autocast():
                logit = refiner(embed_rep, pred_prob, f_in, t_emb)
                loss = batch_loss(logit.float(), masks)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(refiner.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())

            if it % 50 == 0:
                print(f'[{time.strftime("%H:%M:%S")}] ep {ep} it {it}/{len(train_loader)} '
                      f'loss={loss.item():.4f}')

        base_d, ref_d = run_val(args, imis, sacf, refiner, val_loader, device)
        msg = (f'[{time.strftime("%H:%M:%S")}] epoch {ep}: loss={np.mean(losses):.4f} | '
               f'VAL base_dice={base_d:.4f} refined_dice={ref_d:.4f} '
               f'(delta={ref_d - base_d:+.4f})')
        print(msg)
        log_f.write(msg + '\n')
        log_f.flush()

        ck = {'refiner_state_dict': refiner.state_dict(),
              'epoch': ep, 'base_dice': base_d, 'refined_dice': ref_d,
              'no_field': args.no_field}
        torch.save(ck, os.path.join(args.work_dir, f'refiner_latest_{tag}.pth'))
        if ref_d > best_dice:
            best_dice = ref_d
            bad = 0
            torch.save(ck, os.path.join(args.work_dir, f'refiner_best_{tag}.pth'))
            print(f'  -> saved refiner_best_{tag}.pth (refined_dice={ref_d:.4f})')
        else:
            bad += 1
            if args.patience and bad >= args.patience:
                msg = (f'epoch {ep}: early stop, refined_dice 连续 {bad} 轮无提升 '
                       f'(best={best_dice:.4f})')
                print(msg)
                log_f.write(msg + '\n')
                log_f.flush()
                break

    log_f.close()


if __name__ == '__main__':
    main()
