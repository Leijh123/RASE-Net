"""scar.lib — 公共符号导出层.

所有脚本 (build_bank / precompute_cache / train_egrc / eval_egrc_compare /
eval_egrc_256 等) 都通过 `from scar.lib import ...` 拿 SACF 家族的符号.
这里统一从 sacf_dev.sacf 重导出, 保证无论 scar 目录物理上在哪里都能 import.
"""
import os as _os
import sys as _sys

# 确保 sacf_dev 子包可被 import (scar/ 可能在任何 cwd 下被调用).
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
_SACF_DEV = _os.path.join(_ROOT, 'sacf_dev')
if _SACF_DEV not in _sys.path:
    _sys.path.insert(0, _SACF_DEV)

from sacf import (  # noqa: E402
    SemanticContradictionField,
    FieldGuidedRefiner,
    CounterfactualCorruptor,
    SACFLoss,
    SACFDataset,
    sacf_collate_fn,
    field_iou,
    field_iou_oracle,
    pixel_auroc,
    CORRUPTION_TYPES,
    TYPE_TO_INDEX,
    ORGAN_ANATOMY,
    get_anatomy_prompt,
    encode_structured_text,
    EVIDENCE_NAMES,
)

__all__ = [
    'SemanticContradictionField', 'FieldGuidedRefiner',
    'CounterfactualCorruptor', 'SACFLoss',
    'SACFDataset', 'sacf_collate_fn',
    'field_iou', 'field_iou_oracle', 'pixel_auroc',
    'CORRUPTION_TYPES', 'TYPE_TO_INDEX',
    'ORGAN_ANATOMY', 'get_anatomy_prompt', 'encode_structured_text',
    'EVIDENCE_NAMES',
]
