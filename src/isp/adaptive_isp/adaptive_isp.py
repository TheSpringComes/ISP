"""
AdaptiveISP — 完整实现.

Pipeline:
  1. DemosaicNet: raw (B,4,H,W) -> linear RGB (B,3,H,W)   [固定首步]
  2. for stage in 1..max_stages:
       agent 观察当前图像 -> 选择模块 M_i + 预测参数 Θ_i
       if M_i == Identity: break (pipeline 终止)
       x = M_i(x, Θ_i)
  3. 输出最终 RGB

训练:
  - ISP module 内部参数 + demosaic: 通过 detection loss 梯度反传更新
  - RL agent (selector + param heads): PPO 更新
    reward = -det_loss (或 mAP improvement)
    cost   = num_stages * cost_penalty
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from ..base_isp import BaseISP
from .isp_ops import (
    DemosaicNet, ISP_MODULES, ISP_NAMES, NUM_OPS,
    STOP_IDX, MAX_PARAMS,
)
from .rl_agent import RLAgent


class AdaptiveISPModule(BaseISP):
    """AdaptiveISP: RL-driven scene-adaptive ISP pipeline.

    Args:
        in_channels:      输入 RAW 通道数 (4 for RGGB)
        max_stages:       最大 pipeline 长度
        cost_penalty:     每增加一个 stage 的 reward 惩罚
        agent_state_dim:  RL agent state 向量维度
    """

    def __init__(
        self,
        in_channels: int = 4,
        max_stages: int = 5,
        cost_penalty: float = 0.01,
        agent_state_dim: int = 256,
    ):
        super().__init__()
        self.max_stages = max_stages
        self.cost_penalty = cost_penalty

        # 固定首步: demosaic
        self.demosaic = DemosaicNet()

        # ISP 操作库 (nn.ModuleList 确保参数被注册)
        self.ops = nn.ModuleList([cls() for cls in ISP_MODULES])

        # RL Agent
        self.agent = RLAgent(state_dim=agent_state_dim)

        # 训练期间的 rollout buffer
        self._rollout: List[Dict] = []
        self._num_stages_used: int = 0

    # ----------------------------------------------------------
    # Forward
    # ----------------------------------------------------------
    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            raw: (B, 4, H, W) RGGB packed Bayer
        Returns:
            rgb: (B, 3, H, W) 最终处理后的 RGB
        """
        self._intermediate_features = []
        self._rollout = []

        # Step 0: Demosaic (固定)
        x = self.demosaic(raw)
        self._intermediate_features.append(x)

        # Step 1..N: RL-driven ISP stages
        for stage_idx in range(self.max_stages):
            agent_out = self.agent(
                x.detach() if self.training else x,
                deterministic=not self.training,
            )

            actions = agent_out['action']      # (B,)
            params  = agent_out['params']      # (B, MAX_PARAMS)

            # 保存 rollout (训练时用于 RL loss)
            self._rollout.append(agent_out)

            # 全部选了 stop → 终止
            if (actions == STOP_IDX).all():
                break

            # 对 batch 中不同样本应用各自选择的操作
            x_new = torch.zeros_like(x)
            for op_idx, op in enumerate(self.ops):
                mask = (actions == op_idx)
                if not mask.any():
                    continue
                np_ = ISP_MODULES[op_idx].num_params
                op_params = params[mask, :np_] if np_ > 0 else None
                x_new[mask] = op(x[mask], op_params)

            # stop 的样本保持不变
            stop_mask = (actions == STOP_IDX)
            if stop_mask.any():
                x_new[stop_mask] = x[stop_mask]

            x = x_new.clamp(0, 1)
            self._intermediate_features.append(x)

        self._num_stages_used = len(self._rollout)
        return x

    # ----------------------------------------------------------
    # RL Loss (PPO-style)
    # ----------------------------------------------------------
    def compute_rl_loss(
        self,
        reward: torch.Tensor,
        ppo_clip: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
    ) -> torch.Tensor:
        """计算 PPO 风格的 RL 损失.

        Args:
            reward: (B,) per-image reward (通常 = -det_loss)
            ppo_clip: PPO clip ratio ε
            entropy_coef: entropy bonus 系数
            value_coef: value loss 系数

        Returns:
            rl_loss: scalar
        """
        if len(self._rollout) == 0:
            return reward.new_tensor(0.0)

        # Cost penalty: 鼓励短 pipeline
        cost = len(self._rollout) * self.cost_penalty
        adj_reward = reward - cost

        total_loss = reward.new_tensor(0.0)
        for step_data in self._rollout:
            log_prob = step_data['log_prob']
            value    = step_data['value']
            state    = step_data['state']
            action   = step_data['action']

            advantage = (adj_reward - value.detach())

            # Re-evaluate for PPO (如果需要更新多 epoch)
            new_log_prob, entropy, new_value = self.agent.evaluate_action(
                state.detach(), action.detach()
            )

            # PPO clipped objective
            ratio = torch.exp(new_log_prob - log_prob.detach())
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - ppo_clip, 1 + ppo_clip) * advantage
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(new_value, adj_reward.detach())

            # Entropy bonus (鼓励探索)
            entropy_loss = -entropy.mean()

            total_loss = total_loss + policy_loss \
                         + value_coef * value_loss \
                         + entropy_coef * entropy_loss

        return total_loss / max(len(self._rollout), 1)

    # ----------------------------------------------------------
    # Info
    # ----------------------------------------------------------
    def get_isp_config(self) -> Dict:
        if not self._rollout:
            return {'pipeline': [], 'num_stages': 0}
        pipeline = []
        for step in self._rollout:
            a = step['action'][0].item()
            pipeline.append(ISP_NAMES[a])
        return {
            'pipeline': pipeline,
            'num_stages': len(pipeline),
        }
