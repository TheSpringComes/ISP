"""RAW-Adapter — Learnable ISP + Model-level Adapter.

Input-level adapter: 5 阶段固定 ISP pipeline (demosaic -> WB -> CC -> gamma -> tonemap)
Model-level adapter: 将 ISP 中间特征注入 backbone 各 stage

参考: RAW-Adapter: Adapting Pre-trained Visual Model to Camera RAW Images (ECCV 2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from ..base_isp import BaseISP


class QueryAdaptiveLearning(nn.Module):
    """QAL — 用轻量网络从 RAW 图像预测 ISP 参数.

    根据输入内容自适应生成 ISP 超参数, 而非全局固定参数.
    """

    def __init__(self, in_ch: int = 4, param_dim: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(8)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch * 8 * 8, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, param_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(x))


class LearnableDemosaic(nn.Module):
    """4ch RGGB -> 3ch RGB, 可学习."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 12, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(12, 3, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).clamp(0, 1)


class LearnableWB(nn.Module):
    """内容自适应白平衡."""

    def __init__(self, use_qal: bool = True):
        super().__init__()
        self.use_qal = use_qal
        if use_qal:
            self.qal = QueryAdaptiveLearning(3, 3)
        else:
            self.gains = nn.Parameter(torch.ones(3))

    def forward(self, x: torch.Tensor, raw_ref: torch.Tensor = None) -> torch.Tensor:
        if self.use_qal and raw_ref is not None:
            gains = self.qal(raw_ref).abs() + 0.5  # (B, 3)
            gains = gains.view(-1, 3, 1, 1)
        else:
            gains = self.gains.abs().view(1, 3, 1, 1) + 0.5
        return (x * gains).clamp(0, 1)


class LearnableCCM(nn.Module):
    """内容自适应 Color Correction Matrix."""

    def __init__(self, use_qal: bool = True):
        super().__init__()
        self.use_qal = use_qal
        if use_qal:
            self.qal = QueryAdaptiveLearning(3, 9)
        else:
            self.ccm = nn.Parameter(torch.eye(3).flatten())

    def forward(self, x: torch.Tensor, raw_ref: torch.Tensor = None) -> torch.Tensor:
        B, C, H, W = x.shape
        if self.use_qal and raw_ref is not None:
            ccm = self.qal(raw_ref).view(B, 3, 3)
            # 加残差连接: CCM = I + predicted
            ccm = ccm + torch.eye(3, device=x.device).unsqueeze(0)
        else:
            ccm = self.ccm.view(1, 3, 3).expand(B, -1, -1)

        pixels = x.permute(0, 2, 3, 1).reshape(B, -1, 3)
        corrected = torch.bmm(pixels, ccm.transpose(1, 2))
        return corrected.reshape(B, H, W, 3).permute(0, 3, 1, 2).clamp(0, 1)


class LearnableGamma(nn.Module):
    """内容自适应 Gamma."""

    def __init__(self, use_qal: bool = True):
        super().__init__()
        self.use_qal = use_qal
        if use_qal:
            self.qal = QueryAdaptiveLearning(3, 1)
        else:
            self.gamma = nn.Parameter(torch.tensor(1.0 / 2.2))

    def forward(self, x: torch.Tensor, raw_ref: torch.Tensor = None) -> torch.Tensor:
        if self.use_qal and raw_ref is not None:
            gamma = torch.sigmoid(self.qal(raw_ref)) * 2.0 + 0.1  # (B, 1)
            gamma = gamma.view(-1, 1, 1, 1)
        else:
            gamma = self.gamma.abs().clamp(0.1, 3.0)
        return x.clamp(1e-8, 1.0).pow(gamma)


class LearnableToneMap(nn.Module):
    """轻量级 tone mapping."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(8, 3, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, raw_ref: torch.Tensor = None) -> torch.Tensor:
        return self.net(x)


class ModelLevelAdapter(nn.Module):
    """Model-level adapter: 融合 ISP 中间特征到 backbone feature map.

    将 ISP 阶段 i 的 3ch 输出 downsample 后, 通过轻量网络
    投影到 backbone stage i 的通道维度, 加到 backbone 特征上.
    """

    def __init__(self, isp_channels: int = 3, backbone_channels: int = 256,
                 adapter_dim: int = 64):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(isp_channels, adapter_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(adapter_dim, backbone_channels, 1),
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(backbone_channels, backbone_channels),
            nn.Sigmoid(),
        )

    def forward(
        self, bb_feat: torch.Tensor, isp_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            bb_feat: (B, C_bb, H_bb, W_bb)  backbone feature
            isp_feat: (B, 3, H_isp, W_isp)  ISP 中间输出
        Returns:
            adapted: (B, C_bb, H_bb, W_bb)
        """
        # Downsample ISP feature 到 backbone 分辨率
        isp_down = F.interpolate(
            isp_feat, size=bb_feat.shape[2:],
            mode='bilinear', align_corners=False
        )
        # 投影到 backbone 通道维度
        projected = self.adapter(isp_down)
        # 门控融合
        gate = self.gate(bb_feat).view(bb_feat.shape[0], -1, 1, 1)
        return bb_feat + gate * projected


class RAWAdapterModule(BaseISP):
    """RAW-Adapter: Input-level + Model-level adapters."""

    def __init__(
        self,
        in_channels: int = 4,
        backbone_channels: List[int] = None,
        adapter_dim: int = 64,
        use_qal: bool = True,
    ):
        super().__init__()

        if backbone_channels is None:
            backbone_channels = [256, 512, 1024, 2048]  # ResNet-50 default

        self.use_qal = use_qal

        # ===== Input-level adapter: 5-stage learnable ISP =====
        self.stage1_demosaic = LearnableDemosaic()
        self.stage2_wb = LearnableWB(use_qal=use_qal)
        self.stage3_ccm = LearnableCCM(use_qal=use_qal)
        self.stage4_gamma = LearnableGamma(use_qal=use_qal)
        self.stage5_tonemap = LearnableToneMap()

        # ===== Model-level adapters =====
        self.model_adapters = nn.ModuleList([
            ModelLevelAdapter(3, ch, adapter_dim)
            for ch in backbone_channels
        ])

        self._raw_ref = None

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        """(B, 4, H, W) -> (B, 3, H, W)"""
        self._intermediate_features = []
        self._raw_ref = raw[:, :3]  # 取前 3 通道作为 QAL 参考

        # Stage 1: Demosaic
        x1 = self.stage1_demosaic(raw)
        self._intermediate_features.append(x1)

        # Stage 2: White Balance
        x2 = self.stage2_wb(x1, self._raw_ref)
        self._intermediate_features.append(x2)

        # Stage 3: Color Correction
        x3 = self.stage3_ccm(x2, self._raw_ref)
        self._intermediate_features.append(x3)

        # Stage 4: Gamma
        x4 = self.stage4_gamma(x3, self._raw_ref)
        self._intermediate_features.append(x4)

        # Stage 5: Tone Mapping
        x5 = self.stage5_tonemap(x4, self._raw_ref)
        self._intermediate_features.append(x5)

        return x5

    def adapt_backbone_features(
        self, backbone_features: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """将 ISP 中间特征注入 backbone.

        backbone 通常有 4 个 stage (C2, C3, C4, C5 for ResNet),
        ISP 有 5 个 stage. 取后 4 个 ISP stage 特征对应 4 个 backbone stage.
        """
        feats = self._intermediate_features
        num_bb = len(backbone_features)
        # 对齐: 取最后 num_bb 个 ISP 特征
        isp_feats = feats[-num_bb:] if len(feats) >= num_bb else feats

        adapted = []
        for i, bb_feat in enumerate(backbone_features):
            if i < len(isp_feats) and i < len(self.model_adapters):
                adapted.append(
                    self.model_adapters[i](bb_feat, isp_feats[i])
                )
            else:
                adapted.append(bb_feat)

        return adapted

    def get_isp_config(self):
        return {
            'pipeline': ['demosaic', 'wb', 'ccm', 'gamma', 'tonemap'],
            'num_stages': 5,
            'use_qal': self.use_qal,
        }
