"""BaseISP — 所有 ISP 模块的抽象基类.

统一接口:
  forward(raw) -> rgb           主处理: (B, C_in, H, W) -> (B, 3, H, W)
  get_intermediate_features()   返回各阶段中间特征 (RAW-Adapter 的 model-level adapter 需要)
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import List, Dict


class BaseISP(ABC, nn.Module):

    def __init__(self):
        super().__init__()
        self._intermediate_features: List[torch.Tensor] = []

    @abstractmethod
    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        """(B, C_in, H, W) -> (B, 3, H, W), float32, [0, 1]."""
        ...

    def get_intermediate_features(self) -> List[torch.Tensor]:
        return self._intermediate_features

    def get_isp_config(self) -> Dict:
        return {}

    @property
    def num_stages(self) -> int:
        return len(self._intermediate_features)
