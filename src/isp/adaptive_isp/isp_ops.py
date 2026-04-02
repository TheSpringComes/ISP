"""
AdaptiveISP 完整 ISP 操作库 (10 modules + Identity).

论文 Figure 1 中的 ISP Module Pool:
  White Balance / CCM / Gamma / Exposure / Denoise /
  Sharpen / Contrast / Tone Mapping / Saturation / Desaturation / Identity(stop)

每个 module 接口:
  forward(x, params) -> y
    x      : (B, 3, H, W) 当前 stage 的图像
    params : (B, num_params) 由 RL agent 的参数预测网络生成
    y      : (B, 3, H, W)  处理后的图像
  num_params : int, 该模块需要的参数数量
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
    """ITU-R BT.601 luma: (B,3,H,W) -> (B,1,H,W)."""
    w = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return (x * w).sum(1, keepdim=True)


# ------------------------------------------------------------------
# Demosaic (固定首步, 不在 RL 选择范围内)
# ------------------------------------------------------------------
class DemosaicNet(nn.Module):
    """Learned demosaic: 4ch RGGB -> 3ch RGB."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, 3, 1),
        )
    def forward(self, raw):
        return self.net(raw).clamp(0, 1)


# ------------------------------------------------------------------
# 10 ISP Modules
# ------------------------------------------------------------------
class WhiteBalanceOp(nn.Module):
    """R/G/B gain.  params: (B,3) -> gains in [0.5, 2.5]."""
    num_params = 3
    def forward(self, x, params):
        g = (torch.sigmoid(params) * 2.0 + 0.5).view(-1, 3, 1, 1)
        return (x * g).clamp(0, 1)


class CCMOp(nn.Module):
    """3x3 色彩校正矩阵.  params: (B,9) residual matrix."""
    num_params = 9
    def forward(self, x, params):
        B, C, H, W = x.shape
        ccm = params.view(B, 3, 3) * 0.3 + torch.eye(3, device=x.device)
        pix = x.permute(0, 2, 3, 1).reshape(B, -1, 3)
        out = torch.bmm(pix, ccm.transpose(1, 2))
        return out.reshape(B, H, W, 3).permute(0, 3, 1, 2).clamp(0, 1)


class GammaOp(nn.Module):
    """Gamma 校正.  params: (B,1) -> gamma in [0.2, 5.0]."""
    num_params = 1
    def forward(self, x, params):
        gamma = (torch.sigmoid(params) * 4.8 + 0.2).view(-1, 1, 1, 1)
        return x.clamp(1e-8, 1.0).pow(gamma)


class ExposureOp(nn.Module):
    """曝光增益.  params: (B,1) -> gain in [0.1, 10]."""
    num_params = 1
    def forward(self, x, params):
        gain = (torch.sigmoid(params) * 9.9 + 0.1).view(-1, 1, 1, 1)
        return (x * gain).clamp(0, 1)


class DenoiseOp(nn.Module):
    """轻量 CNN 去噪 + strength 混合.  params: (B,1)."""
    num_params = 1
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, 16, 3, padding=1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, 3, 3, padding=1),
        )
    def forward(self, x, params):
        s = torch.sigmoid(params).view(-1, 1, 1, 1)
        return (x + s * self.body(x)).clamp(0, 1)


class SharpenOp(nn.Module):
    """Unsharp mask.  params: (B,1) -> strength in [0, 3]."""
    num_params = 1
    def forward(self, x, params):
        s = (torch.sigmoid(params) * 3.0).view(-1, 1, 1, 1)
        blur = F.avg_pool2d(F.pad(x, [1]*4, 'reflect'), 3, 1)
        return (x + s * (x - blur)).clamp(0, 1)


class ContrastOp(nn.Module):
    """对比度.  params: (B,1) -> factor in [0.5, 3]."""
    num_params = 1
    def forward(self, x, params):
        f = (torch.sigmoid(params) * 2.5 + 0.5).view(-1, 1, 1, 1)
        mu = x.mean(dim=[2, 3], keepdim=True)
        return ((x - mu) * f + mu).clamp(0, 1)


class ToneMappingOp(nn.Module):
    """S-curve tone map.  params: (B,2) contrast + brightness."""
    num_params = 2
    def forward(self, x, params):
        c = (torch.sigmoid(params[:, 0:1]) * 4 + 0.5).view(-1, 1, 1, 1)
        b = (torch.tanh(params[:, 1:2]) * 0.5).view(-1, 1, 1, 1)
        return torch.sigmoid(c * (x - 0.5 + b))


class SaturationOp(nn.Module):
    """增饱和.  params: (B,1) -> factor in [1, 3]."""
    num_params = 1
    def forward(self, x, params):
        f = (torch.sigmoid(params) * 2.0 + 1.0).view(-1, 1, 1, 1)
        gray = rgb_to_gray(x)
        return (gray + f * (x - gray)).clamp(0, 1)


class DesaturationOp(nn.Module):
    """去饱和 (低光场景核心模块).  params: (B,1) -> strength [0,1]."""
    num_params = 1
    def forward(self, x, params):
        s = torch.sigmoid(params).view(-1, 1, 1, 1)
        gray = rgb_to_gray(x).expand_as(x)
        return (s * gray + (1 - s) * x).clamp(0, 1)


class IdentityOp(nn.Module):
    """Stop / 不处理."""
    num_params = 0
    def forward(self, x, params=None):
        return x


# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------
ISP_MODULES = [
    WhiteBalanceOp,   # 0
    CCMOp,            # 1
    GammaOp,          # 2
    ExposureOp,       # 3
    DenoiseOp,        # 4
    SharpenOp,        # 5
    ContrastOp,       # 6
    ToneMappingOp,    # 7
    SaturationOp,     # 8
    DesaturationOp,   # 9
    IdentityOp,       # 10  = STOP
]

ISP_NAMES = [
    'wb', 'ccm', 'gamma', 'exposure', 'denoise',
    'sharpen', 'contrast', 'tonemap', 'saturation',
    'desaturation', 'identity',
]

STOP_IDX  = 10
NUM_OPS   = len(ISP_MODULES)         # 11
MAX_PARAMS = max(c.num_params for c in ISP_MODULES)  # 9 (CCM)
