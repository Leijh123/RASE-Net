"""SACF 数据集: 返回完整多器官标注以支持 relational evidence.

与主训练管线 (UniversalDataset) 不同, 这里需要:
  - image (3,H,W)
  - full label (C,H,W)  全部器官通道 (13 类)
  - 该 slice 上出现的器官 id 列表
collate 时对 (slice, organ) 展开成 organ-level 样本.
"""

import json
import os
import ast
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize

PIXEL_MEAN = (123.675, 116.28, 103.53)
PIXEL_STD = (58.395, 57.12, 57.375)


class SACFDataset(Dataset):
    def __init__(self, data_dir, split_json='dataset.json', split='training',
                 image_size=256, min_organ_area=16):
        self.data_dir = data_dir
        self.image_size = image_size
        self.min_organ_area = min_organ_area

        ds = json.load(open(os.path.join(data_dir, split_json)))
        self.classes = [c for c in ds['labels'].values() if c != 'background']
        self.num_organs = len(self.classes)
        self.datalist = ds.get(split) or ds.get('test') or ds.get('validation') or ds['training']

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        item = self.datalist[idx]
        img = Image.open(os.path.join(self.data_dir, item['image'])).convert('RGB')
        img = np.array(resize(img, (self.image_size, self.image_size),
                              interpolation=InterpolationMode.NEAREST))
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        for c in range(3):
            img[c] = (img[c] - PIXEL_MEAN[c]) / PIXEL_STD[c]

        label_path = os.path.join(self.data_dir, item['label'])
        shape = ast.literal_eval(label_path.split('.')[-2])
        import scipy.sparse as sparse
        lab = sparse.load_npz(label_path).toarray().reshape(shape)  # (C,512,512,1)
        lab = torch.from_numpy(lab[:, :, :, 0]).float() if lab.ndim == 4 else torch.from_numpy(lab).float()
        # resize 到 image_size (最近邻, 保持二值)
        lab_r = F.interpolate(lab.unsqueeze(1), size=(self.image_size, self.image_size),
                              mode='nearest').squeeze(1)  # (C,S,S)

        organ_ids = [int(c) for c in range(lab_r.shape[0])
                     if lab_r[c].sum() >= self.min_organ_area]
        if len(organ_ids) == 0:  # 纯背景 slice, 重采样
            return self.__getitem__(np.random.randint(self.__len__()))

        return {
            'image': img,
            'label': lab_r,            # (C,H,W) 全器官
            'organ_ids': torch.tensor(organ_ids, dtype=torch.long),
        }


def sacf_collate_fn(batch, organs_per_slice=2, min_other_area=16):
    """展开 (slice, organ) 样本.

    返回:
        images   (B,3,H,W)
        labels   (B,C,H,W)
        masks    (N,1,H,W)   选中器官的 GT
        organ_ch (N,)        器官通道 id
        slice_idx (N,)       所属 slice
        names    list[str]   器官名
    """
    images = torch.stack([b['image'] for b in batch])
    labels = torch.stack([b['label'] for b in batch])
    C = labels.shape[1]

    masks, organ_ch, slice_idx, names = [], [], [], []
    for bi, b in enumerate(batch):
        ids = b['organ_ids'].tolist()
        k = min(organs_per_slice, len(ids))
        sel = np.random.choice(ids, size=k, replace=False) if len(ids) >= k else ids
        for c in sel:
            masks.append(labels[bi, c:c + 1])
            organ_ch.append(c)
            slice_idx.append(bi)
            # organ name 由外部 classes 提供, 这里先存通道 id
            names.append(int(c))

    masks = torch.stack(masks) if masks else torch.zeros(0, 1, labels.shape[-2], labels.shape[-1])
    return {
        'images': images,
        'labels': labels,
        'masks': masks,
        'organ_ch': torch.tensor(organ_ch, dtype=torch.long),
        'slice_idx': torch.tensor(slice_idx, dtype=torch.long),
        'organ_names': names,   # 通道 id, 训练脚本里映射为器官名
        'num_organs': C,
    }
