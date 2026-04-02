"""
AdaptiveISP RL Agent.

论文架构 (Figure 11 / Section 3.2):
  1. 共享 Feature Backbone (轻量 CNN, 从当前 stage 图像提取 state)
  2. Module Selection Network (softmax → 选择下一个 ISP 模块)
  3. Parameter Prediction Networks (每个模块一个小 head, 预测该模块参数)
  4. Value Network (PPO baseline)

训练: PPO / REINFORCE
  - Reward = detection mAP (或 -detection_loss)
  - Cost  = stage_count * cost_penalty (鼓励短 pipeline)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Tuple, List, Dict

from .isp_ops import ISP_MODULES, NUM_OPS, MAX_PARAMS, STOP_IDX


class SharedBackbone(nn.Module):
    """轻量 CNN 从当前 stage 图像提取 state vector.

    论文: "lightweight RL agent takes the processing output from the
    previous stage as input" — 用全局池化得到紧凑特征.
    """

    def __init__(self, state_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),   # /2
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # /4
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # /8
            nn.LeakyReLU(0.2, True),
            nn.AdaptiveAvgPool2d(4),                     # 128x4x4
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, state_dim),
            nn.LeakyReLU(0.2, True),
        )
        self.state_dim = state_dim

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) -> (B, state_dim)."""
        return self.fc(self.conv(img))


class ModuleSelector(nn.Module):
    """Module Selection Network — 输出 softmax 概率分布.

    论文: "The activation function for the module selection network
    is softmax, with the number of outputs corresponding to the
    number of ISP modules."
    """

    def __init__(self, state_dim: int, num_modules: int = NUM_OPS):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, num_modules),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """返回 logits (B, num_modules)."""
        return self.head(state)


class ParameterPredictor(nn.Module):
    """Per-module 参数预测 head.

    论文: "Parameter prediction networks share a common feature
    extraction backbone. The activation function and the number
    of outputs from parameter prediction networks are specific
    to each module."

    每个 ISP module 有自己的小 MLP head, 预测该模块所需参数.
    """

    def __init__(self, state_dim: int, num_params: int):
        super().__init__()
        if num_params == 0:
            self.head = None
        else:
            self.head = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ReLU(True),
                nn.Linear(64, num_params),
            )
        self.num_params = num_params

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """(B, state_dim) -> (B, num_params) or None."""
        if self.head is None:
            return None
        return self.head(state)


class ValueHead(nn.Module):
    """State value estimator for PPO advantage computation."""

    def __init__(self, state_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.head(state).squeeze(-1)


class RLAgent(nn.Module):
    """AdaptiveISP 的完整 RL Agent.

    每个 stage:
      1. backbone(current_image) -> state
      2. selector(state) -> module_logits
      3. sample module from Categorical(logits)
      4. param_heads[module](state) -> params
      5. 返回 (action, params, log_prob, value)
    """

    def __init__(self, state_dim: int = 256):
        super().__init__()

        self.backbone = SharedBackbone(state_dim)
        self.selector = ModuleSelector(state_dim, NUM_OPS)
        self.value_head = ValueHead(state_dim)

        # 每个 ISP module 的参数预测 head
        self.param_heads = nn.ModuleList([
            ParameterPredictor(state_dim, cls.num_params)
            for cls in ISP_MODULES
        ])

    def forward(
        self,
        img: torch.Tensor,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            img: (B,3,H,W)
            deterministic: True=推理 (argmax), False=训练 (sample)

        Returns dict:
            action   : (B,) int, 选择的模块 index
            params   : (B, MAX_PARAMS) 对应模块的参数 (多余位填 0)
            log_prob : (B,) 动作 log 概率
            value    : (B,) state value
            logits   : (B, NUM_OPS) 原始 logits
            state    : (B, state_dim)
        """
        state = self.backbone(img)

        # Module selection
        logits = self.selector(state)
        dist = Categorical(logits=logits)

        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        value = self.value_head(state)

        # Parameter prediction for selected modules
        B = img.shape[0]
        params = img.new_zeros(B, MAX_PARAMS)
        for op_idx in range(NUM_OPS):
            mask = (action == op_idx)
            if mask.any() and self.param_heads[op_idx].head is not None:
                pred = self.param_heads[op_idx](state[mask])   # (n, np)
                np_ = pred.shape[1]
                params[mask, :np_] = pred

        return {
            'action': action,
            'params': params,
            'log_prob': log_prob,
            'value': value,
            'logits': logits,
            'state': state,
        }

    def evaluate_action(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """PPO 需要: 根据存储的 state 和 action 重新算 log_prob, entropy, value."""
        logits = self.selector(state)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.value_head(state)
        return log_prob, entropy, value
