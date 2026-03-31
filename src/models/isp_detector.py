"""ISPDetector — 统一的 ISP + Detector 组合模型.

Pipeline:
  raw -> ISP -> rgb -> Detector -> predictions
  (RAW-Adapter 额外: ISP 中间特征 -> model-level adapter -> 注入 backbone)

支持两种训练策略:
  1. E2E: ISP + Detector 联合训练 (RAW-Adapter)
  2. RL + frozen det: ISP 用 RL + det gradient 训练, Detector 冻结 (AdaptiveISP)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional


class ISPDetector(nn.Module):

    def __init__(
        self,
        isp: nn.Module,
        detector: nn.Module,
        use_model_adapter: bool = False,
    ):
        super().__init__()
        self.isp = isp
        self.detector = detector
        self.use_model_adapter = use_model_adapter

    def forward(
        self,
        raw_images: torch.Tensor,
        targets: List[Dict] = None,
    ):
        """
        Args:
            raw_images: (B, C_in, H, W) RAW tensor
            targets: list of dicts, each with 'boxes' and 'labels'
        """
        # Step 1: ISP
        rgb_images = self.isp(raw_images)

        # Step 2: Detection (with or without model-level adapter)
        if (self.use_model_adapter
                and hasattr(self.isp, 'adapt_backbone_features')
                and hasattr(self.detector, 'extract_backbone_features')):
            # RAW-Adapter 路径: 提取 -> 适配 -> 检测
            bb_features = self.detector.extract_backbone_features(rgb_images)
            adapted_features = self.isp.adapt_backbone_features(bb_features)
            output = self.detector.forward_with_features(
                rgb_images, adapted_features, targets
            )
        else:
            # 标准路径: 直接检测
            output = self.detector(rgb_images, targets)

        return output

    def get_isp_output(self, raw_images: torch.Tensor) -> torch.Tensor:
        """只跑 ISP, 返回 RGB 图像 (用于可视化)."""
        with torch.no_grad():
            return self.isp(raw_images)
