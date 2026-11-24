"""Structured Anatomical Semantics (结构化医学语义知识库).

第一创新点 CASC 的 Expected Anatomy 来源: 每个器官的文本约束不再是
裸器官名 ("liver"), 而是 Identity / Location / Shape / Relation 四方面
结构化解剖描述 —— 与四类 evidence 一一对应:

    T_o = [T_identity, T_spatial, T_shape, T_relational]

这样 TextConstraintEncoder 的 FiLM 与 AnatomicalComparator 的期望向量
才真正来自 "医学语义约束", 而非类别身份 (可被 one-hot 替代).

同时提供 encode_structured_text(): 绕过 IMISNet text_tokenizer 内部的
categories_map 类名查表, 直接用冻结 CLIP text encoder 自由编码长文本.
注意: IMISNet decoder 的分割 prompt 仍走 process_text_prompt(organ_name)
(基准协议, 不可动); 只有 SACF/refiner 的评判文本走本模块.
"""

import torch

# 13 器官 (BTCV) 结构化解剖描述, 每条四句分别对应四类 evidence
ORGAN_ANATOMY = {
    'spleen': (
        "The spleen is a homogeneous soft-tissue organ with smooth borders. "
        "It is located in the left upper abdomen. "
        "It has a crescent-like morphology. "
        "It is adjacent to the stomach, left kidney and pancreas tail."
    ),
    'kidney_right': (
        "The right kidney is a bean-shaped paired organ with homogeneous cortex. "
        "It is located in the right retroperitoneal space. "
        "It has a bean-like morphology with smooth borders. "
        "It is adjacent to the liver, right adrenal gland and psoas muscle."
    ),
    'kidney_left': (
        "The left kidney is a bean-shaped paired organ with homogeneous cortex. "
        "It is located in the left retroperitoneal space. "
        "It has a bean-like morphology with smooth borders. "
        "It is adjacent to the spleen, left adrenal gland and psoas muscle."
    ),
    'gallbladder': (
        "The gallbladder is a small fluid-filled sac. "
        "It is located inferior to the liver in the right upper abdomen. "
        "It has a round or oval morphology. "
        "It is adjacent to the liver and duodenum."
    ),
    'esophagus': (
        "The esophagus is a muscular tube. "
        "It is located in the posterior mediastinum. "
        "It has a small circular morphology. "
        "It is adjacent to the aorta and stomach."
    ),
    'liver': (
        "The liver is the largest abdominal organ with homogeneous parenchyma. "
        "It is located in the right upper abdomen. "
        "It has a wedge-like morphology. "
        "It is adjacent to the gallbladder, stomach, right kidney and inferior vena cava."
    ),
    'stomach': (
        "The stomach is a hollow organ with foldable walls. "
        "It is located in the left upper abdomen. "
        "It has a variable bag-like morphology. "
        "It is adjacent to the liver, spleen and pancreas."
    ),
    'aorta': (
        "The abdominal aorta is a large artery. "
        "It is located anterior to the spine in the retroperitoneum. "
        "It has a circular tubular morphology. "
        "It is adjacent to the inferior vena cava and pancreas."
    ),
    'inferior_vena_cava': (
        "The inferior vena cava is a large vein. "
        "It is located to the right of the aorta. "
        "It has a circular tubular morphology. "
        "It is adjacent to the aorta, liver and right kidney."
    ),
    'portal_vein_and_splenic_vein': (
        "The portal and splenic veins are vessels draining into the liver. "
        "They are located between the pancreas and the liver. "
        "They have a small tubular morphology. "
        "They are adjacent to the pancreas and liver."
    ),
    'pancreas': (
        "The pancreas is an elongated soft-tissue organ. "
        "It is located in the retroperitoneum behind the stomach. "
        "It has an elongated morphology with head, body and tail. "
        "It is adjacent to the duodenum, spleen and stomach."
    ),
    'adrenal_gland_right': (
        "The right adrenal gland is a small endocrine gland. "
        "It is located above the right kidney. "
        "It has a small triangular morphology. "
        "It is adjacent to the right kidney and liver."
    ),
    'adrenal_gland_left': (
        "The left adrenal gland is a small endocrine gland. "
        "It is located above the left kidney. "
        "It has a small triangular morphology. "
        "It is adjacent to the left kidney and spleen."
    ),
    # ---- ACDC 心脏短轴 3 类 (Identity / Location / Shape / Relation) ----
    'left_ventricle': (
        "The left ventricle is the thick-walled cardiac blood-pool chamber. "
        "It is located centrally, forming the core of the left heart. "
        "It has a circular blood-pool morphology with smooth borders. "
        "It is surrounded by the myocardium and separated from the right ventricle by the septum."
    ),
    'myocardium': (
        "The myocardium is the thick muscular wall of the left ventricle. "
        "It surrounds the left ventricular blood pool. "
        "It forms a ring-like annular band of uniform thickness. "
        "It encloses the left ventricle and borders the right ventricle at the septum."
    ),
    'right_ventricle': (
        "The right ventricle is the thin-walled cardiac blood-pool chamber. "
        "It is located anterior and to the right of the left ventricle. "
        "It has a crescent or half-moon shaped blood-pool morphology. "
        "It is separated from the left ventricle by the interventricular septum."
    ),
}


def get_anatomy_prompt(organ_name):
    """器官类名 -> 结构化解剖描述; 未知器官回退为原名."""
    return ORGAN_ANATOMY.get(organ_name, organ_name)


@torch.no_grad()
def encode_structured_text(imis, descriptions):
    """自由文本 -> 768d embedding, 绕过 categories_map 类名查表.

    imis.text_tokenizer 内部强制 self.categories_map[t][0] 查表 + 分割模板,
    只接受器官类名; 本函数直接调用冻结 CLIP tokenizer/text_model,
    使结构化长描述可以被编码为 SACF 的 Expected Anatomy 语义向量.

    Args:
        imis: 冻结 IMISNet (提供 .tokenizer / .text_model / .text_out_dim)
        descriptions: list[str] 结构化解剖描述
    Returns:
        (N, 768) text embedding, 与 process_text_prompt 输出同分布通路
    """
    tokens = imis.tokenizer(descriptions, padding=True, truncation=True,
                            max_length=77, return_tensors='pt')
    tokens = {k: v.to(imis.device) for k, v in tokens.items()}
    out = imis.text_model(**tokens)
    return imis.text_out_dim(out.pooler_output)          # (N, 768)
