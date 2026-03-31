"""可微分的 ISP 操作模块.

每个操作接受 (B, 3, H, W) -> (B, 3, H, W), 附带可学习参数.
AdaptiveISP 的 RL agent 在这些操作中选择和组合.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DemosaicOp(nn.Module):
    """Bayer RGGB (4ch) -> RGB (3ch). 使用 learnable 1x1 conv."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 3, kernel_size=3, padding=1, bias=True)
        # 初始化为近似 bilinear demosaic
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 4, H, W) -> (B, 3, H, W)"""
        return self.conv(x).clamp(0, 1)


class WhiteBalanceOp(nn.Module):
    """可学习白平衡: 对 R/G/B 通道分别缩放."""

    def __init__(self):
        super().__init__()
        # 初始化为 [1, 1, 1]
        self.gains = nn.Parameter(torch.ones(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gains = self.gains.abs().view(1, 3, 1, 1)
        return (x * gains).clamp(0, 1)


class ColorCorrectionOp(nn.Module):
    """3x3 Color Correction Matrix (CCM)."""

    def __init__(self):
        super().__init__()
        self.ccm = nn.Parameter(torch.eye(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # x: (B, 3, H, W) -> (B, H*W, 3)
        pixels = x.permute(0, 2, 3, 1).reshape(B, -1, 3)
        corrected = torch.matmul(pixels, self.ccm.T)
        return corrected.reshape(B, H, W, 3).permute(0, 3, 1, 2).clamp(0, 1)


class GammaOp(nn.Module):
    """可学习 Gamma 校正."""

    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(1.0 / 2.2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma.abs().clamp(0.1, 5.0)
        return x.clamp(1e-8, 1.0).pow(gamma)


class DenoiseOp(nn.Module):
    """轻量级 3x3 Conv 去噪."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x + self.net(x)).clamp(0, 1)  # residual


class DesaturationOp(nn.Module):
    """可学习去饱和度 (低光场景下有助于检测)."""

    def __init__(self):
        super().__init__()
        self.factor = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        factor = torch.sigmoid(self.factor)
        gray = x.mean(dim=1, keepdim=True)
        return (factor * gray + (1 - factor) * x).clamp(0, 1)


class SharpenOp(nn.Module):
    """Unsharp masking 锐化."""

    def __init__(self):
        super().__init__()
        self.strength = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        blurred = F.avg_pool2d(x, 3, stride=1, padding=1)
        strength = torch.sigmoid(self.strength)
        return (x + strength * (x - blurred)).clamp(0, 1)


class ToneMappingOp(nn.Module):
    """简单的 S 型 tone mapping (可学习)."""

    def __init__(self):
        super().__init__()
        self.contrast = nn.Parameter(torch.tensor(1.0))
        self.brightness = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = self.contrast.abs().clamp(0.5, 3.0)
        b = self.brightness.clamp(-0.5, 0.5)
        return torch.sigmoid(c * (x - 0.5 + b))


class IdentityOp(nn.Module):
    """恒等操作 — 在 AdaptiveISP 中表示 'stop' action."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


# 操作注册表
ISP_OPS_REGISTRY = {
    'wb': WhiteBalanceOp,
    'ccm': ColorCorrectionOp,
    'gamma': GammaOp,
    'denoise': DenoiseOp,
    'desaturate': DesaturationOp,
    'sharpen': SharpenOp,
    'tonemap': ToneMappingOp,
    'identity': IdentityOp,
}
