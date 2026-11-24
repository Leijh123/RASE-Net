"""TRC 创新点三: 训练阶段稀少器官补偿微调.

两个可独立开关的机制 (消融用):
  --split_json dataset_split_trc.json   (C3a: exemplar copy-paste 增强数据)
  --rare_boost                          (C3b: 器官频率均衡区域选择,
                                          采样概率 ∝ 1/sqrt(切片频率))

流程: 从 Base 最优 checkpoint 初始化 (仅模型权重), 低 LR 短程补偿微调,
每个 epoch 后按器官跟踪 val Dice (FEWSHOT / COMMON / ALL), 保存
IMIS_val_dice_best.pth 与 IMIS_val_fewshot_best.pth 两份 checkpoint.
"""

import sys as _sys, os as _os
_ROOT = _os.path.abspath(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import argparse
import datetime
import json
import logging
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.cuda import amp
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import monai.transforms  # noqa: F401  (兼容保留)
from segment_anything import sam_model_registry
from model import IMISNet
from utils import FocalDice_MSELoss
from data_loader import UniversalDataset, train_collate_fn
from dataloaders.data_utils import (Resize, PermuteTransform, Normalization,
                                     get_points_from_mask,
                                     get_bboxes_from_mask)
from monai import transforms
parser = argparse.ArgumentParser()
parser.add_argument('--work_dir', type=str, default='work_dir')
parser.add_argument('--task_name', type=str, default='trc-full')
parser.add_argument('--data_dir', type=str, default='dataset/BTCV_full')
parser.add_argument('--split_json', type=str, default='dataset_split_trc.json')
parser.add_argument('--image_size', type=int, default=512)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--val_batch_size', type=int, default=None)
parser.add_argument('--model_type', type=str, default='vit_b')
parser.add_argument('--sam_checkpoint', type=str, default='ckpt/IMISNet-B.pth')
parser.add_argument('--init_from', type=str,
                    default='work_dir/ft-IMISNet-full/IMIS_val_dice_best.pth')
parser.add_argument('--mask_num', type=int, default=5)
parser.add_argument('--inter_num', type=int, default=4)
parser.add_argument('--num_epochs', type=int, default=15)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--rare_boost', action='store_true',
                    help='C3b: 器官区域选择概率 ∝ 1/sqrt(切片频率)')
parser.add_argument('--test_mode', type=bool, default=False)
parser.add_argument('--device', type=str, default='cuda')
args = parser.parse_args()

device = torch.device('cuda:0')
MODEL_SAVE_PATH = _os.path.join(args.work_dir, args.task_name)
_os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# AMOS22 MRI 15 器官训练集切片出现频率 (2253 slices 实测, 10-shot split):
FREQ = {0:0.283,1:0.327,2:0.310,3:0.119,4:0.403,5:0.491,6:0.264,7:0.965,8:0.694,9:0.245,10:0.071,11:0.095,12:0.250,13:0.009,14:0.017}
# AMOS 稀少器官 (平均面积最小, 排除 bladder/prostate_uterus 因 test 切片为 0)
FEWSHOT_NAMES = {'adrenal_gland_left','adrenal_gland_right','esophagus','duodenum','gallbladder','pancreas'}

logging.basicConfig(
    format='[%(asctime)s] - %(message)s', datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
    handlers=[logging.FileHandler(
        _os.path.join(MODEL_SAVE_PATH, 'train.log'), mode='a'),
        logging.StreamHandler()])
logger = logging.getLogger()


class TRCDataset(UniversalDataset):
    """C3b: preprocess_label 的区域选择改为 1/sqrt(freq) 加权."""

    def preprocess_label(self, gt_label, label_ids, select_labels):
        point_coords, point_labels, bboxes, categories = [], [], [], []
        if getattr(self, 'rare_boost', False) and len(label_ids) > self.mask_num:
            w = np.array([1.0 / np.sqrt(FREQ.get(int(r), 0.5))
                          for r in label_ids])
            w = w / w.sum()
            region_ids = np.random.choice(
                len(label_ids), size=self.mask_num, replace=False, p=w)
            label_region_ids = [label_ids[k] for k in region_ids]
        else:
            label_region_ids = (random.sample(list(label_ids), k=self.mask_num)
                                if len(label_ids) >= self.mask_num
                                else random.choices(list(label_ids),
                                                    k=self.mask_num))
        for idx, region_id in enumerate(label_region_ids):
            select_labels[idx][0] = gt_label[region_id]
            point_and_labels = get_points_from_mask(select_labels[idx], top_num=0.5)
            point_coords.append(torch.as_tensor(point_and_labels[0]))
            point_labels.append(torch.as_tensor(point_and_labels[1]))
            bboxes.append(torch.as_tensor(
                get_bboxes_from_mask(select_labels[idx], offset=5)))
            categories.append(self.target_list[region_id])
        point_coords = torch.stack(point_coords)
        point_labels = torch.stack(point_labels)
        bboxes = torch.stack(bboxes)
        return select_labels, point_coords, point_labels, bboxes, categories


def get_loader_trc():
    dataset_json = _os.path.join(args.data_dir, args.split_json)
    dataset_dict = json.load(open(dataset_json, 'r'))
    target_size = (args.image_size, args.image_size)
    classes_list = list(dataset_dict['labels'].values())

    train_transform = transforms.Compose([
        Resize(keys=["image", "label", "pseudo"], target_size=target_size),
        PermuteTransform(keys=["image"], dims=(2, 0, 1)),
        transforms.ToTensord(keys=["image", "label", "pseudo"]),
        Normalization(keys=["image"]),
        transforms.RandScaleIntensityd(keys="image", factors=0.2, prob=0.2),
        transforms.RandShiftIntensityd(keys="image", offsets=0.2, prob=0.2),
    ])
    val_transform = transforms.Compose([
        Resize(keys=["image", "label", "pseudo"], target_size=target_size),
        PermuteTransform(keys=["image"], dims=(2, 0, 1)),
        transforms.ToTensord(keys=["image", "label", "pseudo"]),
        Normalization(keys=["image"]),
    ])

    train_ds = TRCDataset(args=args, datalist=dataset_dict['training'],
                          classes_list=classes_list, transform=train_transform)
    val_ds = TRCDataset(args=args, datalist=dataset_dict['validation'],
                        classes_list=classes_list, transform=val_transform)
    train_ds.rare_boost = args.rare_boost
    val_ds.rare_boost = False  # val 统一均匀采样协议, 各变体可比

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=(args.num_workers > 0),
                              collate_fn=train_collate_fn)
    val_batch = args.val_batch_size or args.batch_size
    val_loader = DataLoader(val_ds, batch_size=val_batch, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            persistent_workers=(args.num_workers > 0),
                            collate_fn=train_collate_fn)
    return train_loader, val_loader


def build_model():
    sam = sam_model_registry[args.model_type](args).to(device)
    imis = IMISNet(sam, test_mode=False, select_mask_num=args.mask_num,
                   category_weights='dataloaders/categories_weight.pkl').to(device)
    ck = torch.load(args.init_from, map_location='cpu')
    sd = ck.get('model_state_dict', ck)
    missing, unexpected = imis.load_state_dict(sd, strict=False)
    logger.info(f'init from {args.init_from} (epoch {ck.get("epoch", "?")} '
                f'val_dice {ck.get("best_val_dice", "?")}), '
                f'missing={len(missing)} unexpected={len(unexpected)}')
    return imis


class Trainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.seg_loss = FocalDice_MSELoss()
        self.ce_loss = CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, 0.1)
        self.scaler = amp.GradScaler()
        self.best_val_dice = 0.0
        self.best_val_fewshot = 0.0
        self.history = []

    def train_epoch(self, epoch):
        self.model.train()
        step_loss = step_dice = 0.0
        l = len(self.train_loader)
        for step, batch in enumerate(self.train_loader):
            images = batch['image'].to(device)
            labels = batch['label'].to(device).type(torch.long)
            pseudos = batch['pseudo'].to(device)
            target_list = batch['target_list']
            gt_prompt = batch['gt_prompt']
            pseudo_prompt = batch['pseudo_prompt']
            if torch.sum(labels) == 0 or torch.sum(pseudos) == 0:
                continue
            text_prompt = self.model.process_text_prompt(target_list)
            image_embedding = self.model.image_forward(images)

            gt_prompts, pseudo_prompts = {}, {}
            gt_prm = random.choices(['bboxes', 'points', 'text'], [0.4, 0.3, 0.3])[0]
            pse_prm = random.choices(['bboxes', 'points'], [0.5, 0.5])[0]
            if gt_prm == 'bboxes':
                gt_prompts['bboxes'] = gt_prompt['bboxes'].to(device)
            elif gt_prm == 'points':
                gt_prompts['point_coords'] = gt_prompt['point_coords'].to(device)
                gt_prompts['point_labels'] = gt_prompt['point_labels'].to(device)
            else:
                gt_prompts.update(text_prompt)
            if pse_prm == 'bboxes':
                pseudo_prompts['bboxes'] = pseudo_prompt['bboxes'].to(device)
            else:
                pseudo_prompts['point_coords'] = pseudo_prompt['point_coords'].to(device)
                pseudo_prompts['point_labels'] = pseudo_prompt['point_labels'].to(device)

            with amp.autocast():
                gt_outputs = self.model.forward_decoder(image_embedding, gt_prompts)
                gt_loss = self.seg_loss(gt_outputs['masks'], labels.float(),
                                       gt_outputs['iou_pred'])
                pseudo_outputs = self.model.forward_decoder(image_embedding, pseudo_prompts)
                pseudo_loss = self.seg_loss(pseudo_outputs['masks'], pseudos.float(),
                                            pseudo_outputs['iou_pred'])
                loss = gt_loss + pseudo_loss
                if torch.isnan(loss).any():
                    continue
                self.scaler.scale(loss).backward()

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            gt_preds, gt_low = gt_outputs['masks'], gt_outputs['low_res_masks']
            pseudo_preds, pseudo_low = (pseudo_outputs['masks'],
                                        pseudo_outputs['low_res_masks'])
            image_embedding = image_embedding.detach().clone()
            text_prompt['text_inputs'] = text_prompt['text_inputs'].detach().clone()
            loss, gt_preds, _ = self.interaction(
                image_embedding, gt_low, pseudo_low, gt_preds, pseudo_preds,
                labels, pseudos, text_prompt)
            _, dice = self.iou_dice(gt_preds, labels)
            step_loss += float(loss)
            step_dice += dice
            if (step + 1) % 100 == 0:
                logger.info(f'  E{epoch} step {step + 1}/{l}: loss={loss:.4f} dice={dice:.4f}')
        return step_loss / max(l, 1), step_dice / max(l, 1)

    def interaction(self, image_embedding, gt_low_masks, pseudo_low_masks,
                    gt_preds, pseudo_preds, labels, pseudos, text_prompt):
        total_loss = 0
        text_and_mask_inter = np.random.randint(0, args.inter_num - 1)
        with amp.autocast():
            for inter in range(args.inter_num):
                if inter == text_and_mask_inter or inter == args.inter_num - 1:
                    gt_prompts = self.model.process_mask_prompt(gt_low_masks)
                    gt_prompts.update(text_prompt)
                    gt_outputs = self.model.forward_decoder(image_embedding, gt_prompts)
                    gt_preds, gt_low_masks = gt_outputs['masks'], gt_outputs['low_res_masks']
                    gt_loss = self.seg_loss(gt_preds, labels.float(), gt_outputs['iou_pred'])
                    pseudo_prompts = self.model.process_mask_prompt(pseudo_low_masks)
                else:
                    gt_prompts = self.model.supervised_prompts(
                        None, labels, gt_preds, gt_low_masks, 'points')
                    if random.random() > 0.6:
                        gt_prompts.update(text_prompt)
                        del gt_prompts['mask_inputs']
                    gt_outputs = self.model.forward_decoder(image_embedding, gt_prompts)
                    gt_preds, gt_low_masks = gt_outputs['masks'], gt_outputs['low_res_masks']
                    gt_loss = self.seg_loss(gt_preds, labels.float(), gt_outputs['iou_pred'])
                    pseudo_prompts = self.model.unsupervised_prompts(
                        pseudos, pseudo_preds, pseudo_low_masks, 'points')
                pseudo_outputs = self.model.forward_decoder(image_embedding, pseudo_prompts)
                pseudo_preds, pseudo_low_masks = (pseudo_outputs['masks'],
                                                 pseudo_outputs['low_res_masks'])
                loss = gt_loss + self.seg_loss(pseudo_preds, pseudos.float(),
                                              pseudo_outputs['iou_pred'])
                if torch.isnan(loss).any():
                    continue
                total_loss += loss.item()
                self.scaler.scale(loss).backward(retain_graph=True)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        return total_loss / max(args.inter_num, 1), gt_preds, pseudo_preds

    @staticmethod
    @torch.no_grad()
    def iou_dice(pred, label):
        pred = (torch.sigmoid(pred) > 0.5)
        label = (label > 0)
        inter = torch.logical_and(pred, label).sum(dim=(1, 2, 3)).float()
        union = torch.logical_or(pred, label).sum(dim=(1, 2, 3)).float()
        iou = inter / (union + 1e-8)
        dice = 2 * inter / (pred.sum(dim=(1, 2, 3)) + label.sum(dim=(1, 2, 3)) + 1e-8)
        return iou.mean().item(), dice.mean().item()

    @torch.no_grad()
    def val_epoch(self, epoch):
        """按器官跟踪 val Dice (与 train.py 同协议: bboxes + text prompt)."""
        self.model.eval()
        organ_dice = defaultdict(list)
        loss_sum = n_batch = 0
        for batch in self.val_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device).type(torch.long)
            target_list = batch['target_list']
            gt_prompt = batch['gt_prompt']
            if torch.sum(labels) == 0:
                continue
            text_prompt = self.model.process_text_prompt(target_list)
            image_embedding = self.model.image_forward(images)
            gt_prompts = {}
            if gt_prompt.get('bboxes') is not None:
                gt_prompts['bboxes'] = gt_prompt['bboxes'].to(device)
            gt_prompts.update(text_prompt)
            with amp.autocast():
                outputs = self.model.forward_decoder(image_embedding, gt_prompts)
                loss = self.seg_loss(outputs['masks'], labels.float(),
                                     outputs['iou_pred'])
            if torch.isnan(loss).any():
                continue
            loss_sum += float(loss)
            n_batch += 1
            preds = (torch.sigmoid(outputs['masks']) > 0.5)
            gt = (labels > 0)
            inter = torch.logical_and(preds, gt).sum(dim=(1, 2, 3)).float()
            denom = (preds.sum(dim=(1, 2, 3)) + gt.sum(dim=(1, 2, 3))).float() + 1e-8
            dices = 2 * inter / denom
            for j, name in enumerate(target_list):
                organ_dice[name].append(float(dices[j]))
        if n_batch == 0:
            return None
        all_d = [v for vs in organ_dice.values() for v in vs]
        few_d = [v for n, vs in organ_dice.items() if n in FEWSHOT_NAMES for v in vs]
        com_d = [v for n, vs in organ_dice.items() if n not in FEWSHOT_NAMES for v in vs]
        res = {'loss': loss_sum / n_batch,
               'dice_all': float(np.mean(all_d)),
               'dice_fewshot': float(np.mean(few_d)) if few_d else 0.0,
               'dice_common': float(np.mean(com_d)) if com_d else 0.0,
               'per_organ': {n: float(np.mean(vs)) for n, vs in organ_dice.items()}}
        return res

    def save(self, name):
        torch.save({'model_state_dict': self.model.state_dict(),
                    'args': vars(args)}, _os.path.join(MODEL_SAVE_PATH, name))

    def train(self):
        for epoch in range(args.num_epochs):
            logger.info(f'Epoch {epoch + 1}/{args.num_epochs}')
            tr_loss, tr_dice = self.train_epoch(epoch + 1)
            res = self.val_epoch(epoch + 1)
            self.lr_scheduler.step()
            if res is None:
                continue
            self.history.append({'epoch': epoch + 1, 'train_loss': tr_loss,
                                 'train_dice': tr_dice, **res})
            logger.info(
                f'E{epoch + 1}: train_loss={tr_loss:.4f} train_dice={tr_dice:.4f} | '
                f'val dice_all={res["dice_all"]:.4f} '
                f'FEWSHOT={res["dice_fewshot"]:.4f} COMMON={res["dice_common"]:.4f} '
                f'[best: all={self.best_val_dice:.4f} few={self.best_val_fewshot:.4f}]')
            few_sorted = sorted(
                ((n, d) for n, d in res['per_organ'].items()
                 if n in FEWSHOT_NAMES), key=lambda x: x[1])
            logger.info('  worst fewshot: ' + ', '.join(
                f'{n}={d:.3f}' for n, d in few_sorted[:4]))
            if res['dice_all'] > self.best_val_dice:
                self.best_val_dice = res['dice_all']
                self.save('IMIS_val_dice_best.pth')
                logger.info(f'  [CKPT] IMIS_val_dice_best.pth @E{epoch + 1}')
            if res['dice_fewshot'] > self.best_val_fewshot:
                self.best_val_fewshot = res['dice_fewshot']
                self.save('IMIS_val_fewshot_best.pth')
                logger.info(f'  [CKPT] IMIS_val_fewshot_best.pth @E{epoch + 1}')
            self.save(f'epoch{epoch + 1:02d}.pth')
            # 磁盘优化: 仅保留最近 2 个 epoch checkpoint, 旧的删除
            import glob as _glob
            _ep_files = sorted(_glob.glob(_os.path.join(MODEL_SAVE_PATH, 'epoch*.pth')))
            for _oldf in _ep_files[:-2]:
                _os.remove(_oldf)
            json.dump(self.history, open(
                _os.path.join(MODEL_SAVE_PATH, 'history.json'), 'w'), indent=1)
        logger.info(f'TRAIN DONE. best all={self.best_val_dice:.4f} '
                    f'fewshot={self.best_val_fewshot:.4f}')


def main():
    logger.info('ARGS: ' + json.dumps({k: str(v) for k, v in vars(args).items()}))
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    cudnn.benchmark = True
    train_loader, val_loader = get_loader_trc()
    logger.info(f'train={len(train_loader.dataset)} val={len(val_loader.dataset)}')
    model = build_model()
    Trainer(model, train_loader, val_loader).train()


if __name__ == '__main__':
    main()
