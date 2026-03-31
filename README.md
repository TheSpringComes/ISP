# Unified ISP for Downstream Detection

统一对比 **AdaptiveISP** (NeurIPS 2024) 和 **RAW-Adapter** (ECCV 2024) 的代码框架，
当前聚焦 **LOD 数据集** (低光 RAW 目标检测)。

## 架构概览

```
RAW Image (4ch RGGB)
       │
       ▼
┌─────────────────────┐
│    ISP Module        │ ← 可插拔: AdaptiveISP / RAW-Adapter / 自定义
│  (BaseISP 接口)      │
└────────┬────────────┘
         │  RGB Image (3ch)
         ▼
┌─────────────────────┐
│    Detector          │ ← AdaptiveISP→YOLOv3(frozen) / RAW-Adapter→Sparse-RCNN(E2E)
│  (各自原始 detector)  │
└────────┬────────────┘
         │
         ▼
    Detection Output (boxes, labels, scores)
```

**RAW-Adapter 额外路径**: ISP 中间特征 → Model-level Adapter → 注入 Backbone

## 关键设计

| 维度 | AdaptiveISP | RAW-Adapter |
|------|------------|-------------|
| ISP | RL Agent 动态选择操作 | 固定 5 阶段 learnable ISP |
| Detector | YOLOv3 (冻结) | Sparse-RCNN (联合训练) |
| 训练 | RL + gradient | 端到端 |
| Model Adapter | 无 | ISP 中间特征注入 backbone |
| 配置 | `configs/adaptive_isp/` | `configs/raw_adapter/` |

## 快速开始

### 1. 安装

```bash
pip install -r requirements.txt
```

### 2. 准备 LOD 数据

下载 LOD 数据集 (参考 https://github.com/ying-fu/LODDataset)，组织为:

```
data/LOD/
├── RAW_dark/              # 低光 RAW 图像 (PNG)
│   ├── 2.png
│   └── ...
├── xml_annotations/       # VOC XML 标注
│   ├── 2.xml
│   └── ...
```

转换为 COCO JSON 格式:

```bash
python tools/prepare_lod.py \
    --img-dir data/LOD/RAW_dark \
    --xml-dir data/LOD/xml_annotations \
    --output-dir data/LOD/annotations
```

如果你已有 COCO JSON 标注 (如 SimROD 提供的)，直接放入 `data/LOD/annotations/` 即可。

### 3. 训练

```bash
# AdaptiveISP (RL 训练, detector 冻结)
python tools/train.py --config configs/adaptive_isp/adaptive_isp_lod.yaml

# RAW-Adapter (E2E 联合训练)
python tools/train.py --config configs/raw_adapter/raw_adapter_lod.yaml
```

覆盖配置参数:
```bash
python tools/train.py --config configs/adaptive_isp/adaptive_isp_lod.yaml \
    --override training.batch_size=8 training.max_epochs=12
```

### 4. 测试

```bash
python tools/test.py \
    --config configs/adaptive_isp/adaptive_isp_lod.yaml \
    --checkpoint work_dirs/adaptive_isp_lod/best.pth \
    --visualize
```

## 项目结构

```
unified_isp_det/
├── configs/                     YAML 实验配置
│   ├── adaptive_isp/
│   │   └── adaptive_isp_lod.yaml
│   └── raw_adapter/
│       └── raw_adapter_lod.yaml
├── src/
│   ├── datasets/
│   │   └── lod.py              LOD 数据集 (COCO JSON + VOC XML)
│   ├── isp/
│   │   ├── base_isp.py          BaseISP 抽象接口
│   │   ├── adaptive_isp/
│   │   │   ├── isp_ops.py       可微 ISP 操作 (WB/CCM/Gamma/...)
│   │   │   ├── rl_agent.py      RL 策略网络
│   │   │   └── adaptive_isp.py  AdaptiveISP 主模块
│   │   └── raw_adapter/
│   │       └── raw_adapter.py   RAW-Adapter (Input+Model adapters)
│   ├── detectors/
│   │   ├── yolov3_detector.py   YOLOv3 (AdaptiveISP 用)
│   │   └── sparse_rcnn_detector.py  Sparse-RCNN (RAW-Adapter 用)
│   ├── models/
│   │   └── isp_detector.py      ISP + Detector 组合模型
│   └── engine/
│       ├── trainer.py           统一训练器 (E2E / RL)
│       └── evaluator.py         mAP 评估器
├── tools/
│   ├── train.py                 训练入口
│   ├── test.py                  测试入口
│   └── prepare_lod.py           数据准备 (XML→JSON)
└── scripts/
    ├── train_adaptive_isp.sh
    └── train_raw_adapter.sh
```

## 扩展指南

### 添加新的 ISP 方法

1. 继承 `BaseISP`，实现 `forward(raw) -> rgb`
2. 如果有中间特征需要注入 backbone，实现 `adapt_backbone_features()`
3. 在 `tools/train.py` 的 `build_isp()` 中注册
4. 写对应的 YAML config

### 添加新的 Detector

1. 实现统一接口: `forward(images, targets)` → 训练返回 loss dict，推理返回 predictions
2. 如果要支持 model-level adapter，实现 `extract_backbone_features()` 和 `forward_with_features()`
3. 在 `build_detector()` 中注册

### 添加新的数据集

参考 `LODDataset` 的实现，保持输出格式一致:
- `__getitem__` 返回 `(image_tensor, target_dict)`
- `target_dict` 包含 `boxes` (xyxy), `labels`, `image_id`

## 注意事项

- Detector 当前使用 torchvision Faster R-CNN 作为 placeholder。如需使用原始的 YOLOv3 或 Sparse-RCNN，替换对应 detector 文件即可，接口保持一致。
- LOD 图像如果是原始 .CR2 格式，需要安装 `rawpy`；如果已转为 PNG 则无需。
- RAW-Adapter 的 `forward_with_features` 路径需要 detector 支持拆分 backbone/head，当前实现基于 torchvision 内部 API，换用 mmdetection 时需要适配。

## 引用

```bibtex
@inproceedings{wangadaptiveisp,
  title={AdaptiveISP: Learning an Adaptive Image Signal Processor for Object Detection},
  author={Wang, Yujin and Fan, Zhang and Xue, Tianfan and Gu, Jinwei and others},
  booktitle={NeurIPS},
  year={2024}
}

@inproceedings{raw_adapter,
  title={RAW-Adapter: Adapting Pretrained Visual Model to Camera RAW Images},
  author={Cui, Ziteng and Harada, Tatsuya},
  booktitle={ECCV},
  year={2024}
}

@inproceedings{Hong2021Crafting,
  title={Crafting Object Detection in Very Low Light},
  author={Yang Hong, Kaixuan Wei, Linwei Chen, Ying Fu},
  booktitle={BMVC},
  year={2021}
}
```
