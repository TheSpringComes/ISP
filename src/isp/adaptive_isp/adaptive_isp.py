"""AdaptiveISP — RL-driven 动态 ISP Pipeline.

核心思路:
  1. Demosaic (固定): RGGB 4ch -> RGB 3ch
  2. RL Agent 逐 stage 选择操作: WB / CCM / Gamma / Denoise / Desaturate / Sharpen / ToneMap / Stop
  3. 每个 stage 选一个操作, "Stop" 表示 pipeline 终止
  4. Detection loss 反传优化 ISP 参数, RL 优化 pipeline 选择

参考: AdaptiveISP: Learning an Adaptive Image Signal Processor for Object Detection (NeurIPS 2024)
"""

import torch
import torch.nn as nn
from typing import Dict, List

from ..base_isp import BaseISP
from .isp_ops import (
    DemosaicOp, WhiteBalanceOp, ColorCorrectionOp, GammaOp,
    DenoiseOp, DesaturationOp, SharpenOp, ToneMappingOp, IdentityOp,
)
from .rl_agent import RLAgent


class AdaptiveISPModule(BaseISP):

    def __init__(
        self,
        in_channels: int = 4,
        max_stages: int = 5,
        cost_penalty: float = 0.01,
        agent_hidden_dim: int = 256,
    ):
        super().__init__()
        self.max_stages = max_stages
        self.cost_penalty = cost_penalty

        # 固定的 demosaic: 4ch -> 3ch
        self.demosaic = DemosaicOp()

        # ISP 操作库 (不含 demosaic 和 identity)
        self.op_names = ['wb', 'ccm', 'gamma', 'denoise',
                         'desaturate', 'sharpen', 'tonemap', 'identity']
        self.ops = nn.ModuleList([
            WhiteBalanceOp(),
            ColorCorrectionOp(),
            GammaOp(),
            DenoiseOp(),
            DesaturationOp(),
            SharpenOp(),
            ToneMappingOp(),
            IdentityOp(),  # index 7 = stop
        ])
        self.stop_idx = len(self.ops) - 1

        # RL Agent
        self.agent = RLAgent(
            in_channels=3,
            num_actions=len(self.ops),
            hidden_dim=agent_hidden_dim,
        )

        # RL 训练中间数据
        self._rl_data = []

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        """(B, 4, H, W) -> (B, 3, H, W)"""
        self._intermediate_features = []
        self._rl_data = []

        # Step 1: Demosaic
        x = self.demosaic(raw)
        self._intermediate_features.append(x)

        # Step 2: RL-driven pipeline
        for stage in range(self.max_stages):
            state = self.agent.extract_state(x.detach())
            actions, log_probs, values = self.agent.select_action(
                state, deterministic=not self.training
            )

            self._rl_data.append({
                'actions': actions,
                'log_probs': log_probs,
                'values': values,
                'state': state,
            })

            # 检查是否所有 batch 都选了 stop
            if (actions == self.stop_idx).all():
                break

            # 对每个 batch item 应用各自选择的操作
            x_new = torch.zeros_like(x)
            for op_idx, op in enumerate(self.ops):
                mask = (actions == op_idx).float().view(-1, 1, 1, 1)
                if mask.sum() > 0:
                    x_new = x_new + mask * op(x)

            x = x_new.clamp(0, 1)
            self._intermediate_features.append(x)

        return x

    def get_rl_loss(self, reward: torch.Tensor) -> torch.Tensor:
        """计算 RL policy gradient loss.

        Args:
            reward: (B,) per-image reward (e.g., -detection_loss or mAP)
        Returns:
            rl_loss: scalar
        """
        if len(self._rl_data) == 0:
            return torch.tensor(0.0, device=reward.device)

        policy_loss = torch.tensor(0.0, device=reward.device)
        value_loss = torch.tensor(0.0, device=reward.device)
        entropy_bonus = torch.tensor(0.0, device=reward.device)

        # Cost penalty: 鼓励更短的 pipeline
        cost = len(self._rl_data) * self.cost_penalty
        adjusted_reward = reward - cost

        for step_data in self._rl_data:
            advantage = adjusted_reward - step_data['values'].detach()

            # Policy gradient
            policy_loss -= (step_data['log_probs'] * advantage).mean()

            # Value loss
            value_loss += F.mse_loss(step_data['values'],
                                     adjusted_reward.detach())

            # Entropy (鼓励探索)
            logits = self.agent.policy(step_data['state'])
            dist = torch.distributions.Categorical(logits=logits)
            entropy_bonus -= dist.entropy().mean()

        total = policy_loss + 0.5 * value_loss + 0.01 * entropy_bonus
        return total / max(len(self._rl_data), 1)

    def get_isp_config(self) -> Dict:
        if not self._rl_data:
            return {}
        pipeline = []
        for step in self._rl_data:
            actions = step['actions']
            # 取 batch 中第一个样本的 action
            a = actions[0].item()
            pipeline.append(self.op_names[a])
        return {
            'pipeline': pipeline,
            'num_stages': len(pipeline),
        }


# 需要 F 用于 get_rl_loss
import torch.nn.functional as F
