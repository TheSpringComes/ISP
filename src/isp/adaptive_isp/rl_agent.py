"""RL Agent for AdaptiveISP.

轻量级策略网络, 在每个 stage 根据当前处理结果选择下一步 ISP 操作.
使用 REINFORCE / PPO 训练.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class StateEncoder(nn.Module):
    """从当前图像提取紧凑的状态向量 (用于 RL agent 决策)."""

    def __init__(self, in_channels: int = 3, state_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(16),
            nn.Flatten(),
            nn.Linear(in_channels * 16 * 16, state_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.net(img)


class RLAgent(nn.Module):
    """ISP Pipeline 选择的 RL Agent.

    输入: 当前 stage 的图像特征 (state)
    输出: 选择哪个 ISP 操作 (action)
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_actions: int = 8,
        state_dim: int = 128,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.state_encoder = StateEncoder(in_channels, state_dim)
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_actions),
        )
        self.value = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def extract_state(self, img: torch.Tensor) -> torch.Tensor:
        """从批次图像提取 state. (B, 3, H, W) -> (B, state_dim)"""
        return self.state_encoder(img)

    def select_action(
        self, state: torch.Tensor, deterministic: bool = False
    ):
        """选择动作.

        Returns:
            actions: (B,) int tensor
            log_probs: (B,) log probability
            values: (B,) state value (for PPO)
        """
        logits = self.policy(state)
        dist = Categorical(logits=logits)

        if deterministic:
            actions = logits.argmax(dim=-1)
        else:
            actions = dist.sample()

        log_probs = dist.log_prob(actions)
        values = self.value(state).squeeze(-1)

        return actions, log_probs, values

    def evaluate_actions(self, state: torch.Tensor, actions: torch.Tensor):
        """评估已有 actions 的 log_prob 和 entropy (PPO 需要)."""
        logits = self.policy(state)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.value(state).squeeze(-1)
        return log_probs, entropy, values
