# RASE-Net 项目说明

本项目将训练监督补偿与预测阶段精修分开组织。TRC包含解剖锚定示例复制粘贴增强AAECA与逆平方根频率器官采样ISRFOS；SCAR由语义解剖矛盾场SACF及场引导精修器组成；EGRC通过同类实例检索、质心对齐与门控残差修正补充形态信息。

## 从哪里开始

1. 阅读 `data.md`，准备数据、固定划分和本地模型文件，不重新随机划分。
2. 阅读 `versions.md`，根据现有权重选择历史流程或状态门控实现。
3. 修改 `configs/*.example.json` 中的路径与阶段参数。
4. 使用 `python -m rase <stage> --config <file> --dry-run` 检查实际命令，再运行训练。
5. 按 `evaluation.md` 明确评价集合、网格、阈值及空预测处理。

## 代码定位

| 论文部分 | 代码 |
|---|---|
| TRC实例增强 | `rase/trc/augmentation_btcv.py` |
| TRC频率权重 | `rase/trc/sampling.py` |
| SACF与场引导精修器 | `rase/models/scar.py` |
| SACF损失 | `rase/losses/scar.py` |
| 当前EGRC | `rase/models/egrc.py` |
| 检索与对齐 | `rase/retrieval/core.py` |
| 历史训练、推理完整调用链 | `rase/_historical/` |
| 后补内部消融参考 | `experiments/internal_reference/` |

`_historical`是兼容原实验权重的完整后端，不是需要读者逐个修改的服务器脚本集合。对外使用统一入口。当前EGRC与历史EGRC分开放置，避免将不同结构的权重误认为同一个版本。

包内不含数据、权重、密钥、服务器登录信息及患者名单。没有自动上传或公开GitHub仓库。
