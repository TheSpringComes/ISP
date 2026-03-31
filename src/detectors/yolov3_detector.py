"""YOLOv3 检测器 — AdaptiveISP 原始使用的 detector.

简化的 YOLOv3 实现, 使用 torchvision 预训练模型或自定义 Darknet-53.
在 AdaptiveISP 框架中 detector 权重冻结, 只有 ISP 部分被训练.

为了代码完整性和可运行性, 这里使用 torchvision 的 Faster R-CNN
作为"YOLOv3 placeholder". 实际使用时你可以:
  1. 替换为真正的 YOLOv3 (来自 ultralytics 或 AdaptiveISP 原始代码)
  2. 使用 mmdetection 的 YOLOv3
  3. 保持当前的 Faster R-CNN 做对比实验
"""

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from typing import Dict, List, Optional


class YOLOv3Detector(nn.Module):
    """YOLOv3 Detector wrapper.

    当前实现: 基于 torchvision Faster R-CNN 的 placeholder.
    接口保持一致, 方便替换为真正的 YOLOv3.

    Args:
        num_classes: 类别数 (不含背景)
        pretrained: 是否使用预训练权重
        freeze: 是否冻结所有参数 (AdaptiveISP 模式)
        input_channels: 输入通道数 (3=RGB, 来自 ISP 输出)
    """

    def __init__(
        self,
        num_classes: int = 8,
        pretrained: bool = True,
        freeze: bool = True,
        input_channels: int = 3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.frozen = freeze

        # 使用 torchvision Faster R-CNN 作为 placeholder
        if pretrained:
            weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            self.model = fasterrcnn_resnet50_fpn(weights=weights)
        else:
            self.model = fasterrcnn_resnet50_fpn(weights=None)

        # 替换分类头 (num_classes + 1 for background)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes + 1
        )

        if freeze:
            self._freeze()

    def _freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(
        self,
        images: torch.Tensor,
        targets: List[Dict] = None,
    ) -> Dict:
        """
        训练时: 返回 loss dict
        推理时: 返回 predictions list

        Args:
            images: (B, 3, H, W) — 来自 ISP 的 RGB 输出
            targets: list of dicts with 'boxes' (xyxy) and 'labels'
        """
        # torchvision 的 detection model 需要 list of tensors
        image_list = [img for img in images]

        if self.training and targets is not None:
            # 训练模式: 需要 target 格式对齐
            formatted_targets = []
            for t in targets:
                formatted_targets.append({
                    'boxes': t['boxes'],
                    'labels': t['labels'],
                })
            losses = self.model(image_list, formatted_targets)
            return losses  # dict: loss_classifier, loss_box_reg, ...
        else:
            # 推理模式
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(image_list)
            return predictions  # list of dicts with boxes, labels, scores

    def get_backbone_features(self, images: torch.Tensor) -> List[torch.Tensor]:
        """提取 backbone 特征 (用于 model-level adapter).

        注意: 在 AdaptiveISP 中不需要这个, 但保留接口兼容性.
        """
        features = self.model.backbone(images)
        return [features[k] for k in sorted(features.keys())]
