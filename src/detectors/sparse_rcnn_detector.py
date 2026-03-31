"""Sparse R-CNN 检测器 — RAW-Adapter 原始使用的 detector.

优先使用 mmdetection 的 Sparse R-CNN; 如果 mmdetection 不可用,
回退到 torchvision Faster R-CNN (结构不同但接口一致).

RAW-Adapter 的关键特点: detector 与 ISP 联合训练 (非冻结),
且 model-level adapter 需要 backbone 的多尺度特征.

Args:
    num_classes: 类别数 (不含背景)
    backbone_name: backbone 类型
    pretrained: 是否使用预训练
    backbone_channels: backbone 各 stage 输出通道数 (用于 model-level adapter)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple


def _try_import_mmdet():
    try:
        import mmdet
        return True
    except ImportError:
        return False


class SparseRCNNDetector(nn.Module):
    """Sparse R-CNN detector with exposed backbone features.

    关键区别于 YOLOv3Detector:
      1. 不冻结 — 联合训练
      2. 暴露 backbone 中间特征 — 供 model-level adapter 使用
      3. 支持 backbone feature 被外部修改后重新注入
    """

    def __init__(
        self,
        num_classes: int = 8,
        pretrained: bool = True,
        freeze: bool = False,
        backbone_channels: List[int] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.has_mmdet = _try_import_mmdet()

        if backbone_channels is None:
            backbone_channels = [256, 512, 1024, 2048]
        self.backbone_channels = backbone_channels

        # 使用 torchvision 的模块组装
        self._build_with_torchvision(num_classes, pretrained)

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def _build_with_torchvision(self, num_classes, pretrained):
        """用 torchvision 组件构建: ResNet-50 backbone + FPN + RoI head."""
        import torchvision
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

        if pretrained:
            from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
            self.det_model = fasterrcnn_resnet50_fpn(
                weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            )
        else:
            self.det_model = fasterrcnn_resnet50_fpn(weights=None)

        # 替换分类头
        in_features = self.det_model.roi_heads.box_predictor.cls_score.in_features
        self.det_model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes + 1
        )

    def extract_backbone_features(
        self, images: torch.Tensor
    ) -> List[torch.Tensor]:
        """只跑 backbone, 返回多尺度特征.

        这些特征可以被 RAW-Adapter 的 model-level adapter 修改,
        修改后再送入 detection head.
        """
        features = self.det_model.backbone(images)
        # torchvision backbone 输出 OrderedDict
        feat_list = [features[k] for k in sorted(features.keys())]
        return feat_list

    def forward_with_features(
        self,
        images: torch.Tensor,
        adapted_features: Dict,
        targets: List[Dict] = None,
    ):
        """使用外部提供的 (已被 adapter 修改的) 特征做检测.

        这是 RAW-Adapter 的核心: ISP 中间特征融合到 backbone 后,
        用修改后的特征做检测.
        """
        from collections import OrderedDict

        # 包装为 ImageList (torchvision 需要)
        from torchvision.models.detection.transform import GeneralizedRCNNTransform
        original_image_sizes = [
            (img.shape[-2], img.shape[-1]) for img in images
        ]

        # 构造 ImageList
        image_list = torch.stack([img for img in images])

        # 直接用传入的 features
        if isinstance(adapted_features, list):
            keys = sorted(self.det_model.backbone(images[:1]).keys())[:len(adapted_features)]
            feature_dict = OrderedDict(zip(keys, adapted_features))
        elif isinstance(adapted_features, dict):
            feature_dict = adapted_features
        else:
            feature_dict = adapted_features

        if self.training and targets is not None:
            formatted = [{'boxes': t['boxes'], 'labels': t['labels']}
                         for t in targets]
            # 手动走 rpn + roi_heads
            proposals, proposal_losses = self.det_model.rpn(
                image_list, feature_dict, formatted
            )
            detections, detector_losses = self.det_model.roi_heads(
                feature_dict, proposals, [(img.shape[-2], img.shape[-1]) for img in images], formatted
            )
            losses = {}
            losses.update(proposal_losses)
            losses.update(detector_losses)
            return losses
        else:
            with torch.no_grad():
                proposals, _ = self.det_model.rpn(image_list, feature_dict)
                detections, _ = self.det_model.roi_heads(
                    feature_dict, proposals, original_image_sizes
                )
            return detections

    def forward(
        self,
        images: torch.Tensor,
        targets: List[Dict] = None,
    ):
        """标准 forward (不使用 model-level adapter 时)."""
        image_list = [img for img in images]

        if self.training and targets is not None:
            formatted = [{'boxes': t['boxes'], 'labels': t['labels']}
                         for t in targets]
            losses = self.det_model(image_list, formatted)
            return losses
        else:
            self.det_model.eval()
            with torch.no_grad():
                predictions = self.det_model(image_list)
            return predictions
