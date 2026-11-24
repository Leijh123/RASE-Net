"""SACF: Semantic Anatomical Contradiction Field.

第一创新点: Counterfactual Anatomical Semantic Criterion (CASC)
  - SemanticContradictionField  矛盾场网络 (model)
  - CounterfactualCorruptor     反事实解剖扰动 (corruptions)
  - SACFLoss                    contradiction ranking 目标 (losses)
"""

from .model import (SemanticContradictionField, FieldGuidedRefiner,
                    AnatomicalComparator, EVIDENCE_NAMES)
from .corruptions import CounterfactualCorruptor, CORRUPTION_TYPES, TYPE_TO_INDEX
from .losses import SACFLoss, field_iou, field_iou_oracle, pixel_auroc
from .dataset import SACFDataset, sacf_collate_fn
from .anatomy_knowledge import (ORGAN_ANATOMY, get_anatomy_prompt,
                                encode_structured_text)
