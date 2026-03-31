"""统一训练器.

支持两种训练策略:
  1. E2E (RAW-Adapter): 标准 SGD/Adam 训练 ISP + Detector
  2. RL (AdaptiveISP): RL 训练 agent + gradient 训练 ISP 参数, detector 冻结
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Optional

from .evaluator import DetectionEvaluator


class Trainer:

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        device: str = 'cuda',
        num_classes: int = 8,
        work_dir: str = 'work_dirs/experiment',
        max_epochs: int = 12,
        log_interval: int = 50,
        eval_interval: int = 1,
        training_mode: str = 'e2e',  # 'e2e' or 'rl'
        rl_optimizer=None,           # AdaptiveISP RL agent 的 optimizer
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.work_dir = work_dir
        self.max_epochs = max_epochs
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.training_mode = training_mode
        self.rl_optimizer = rl_optimizer

        self.evaluator = DetectionEvaluator(
            num_classes=num_classes,
            iou_thresholds=[0.5],
        )

        os.makedirs(work_dir, exist_ok=True)
        self.best_map = 0.0

        # Tensorboard (可选)
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(os.path.join(work_dir, 'tb_logs'))
        except ImportError:
            self.writer = None

    def train(self):
        """主训练循环."""
        print(f"{'='*60}")
        print(f"Training: mode={self.training_mode}, epochs={self.max_epochs}")
        print(f"Device: {self.device}, work_dir: {self.work_dir}")
        print(f"{'='*60}")

        for epoch in range(self.max_epochs):
            if self.training_mode == 'rl':
                train_loss = self._train_epoch_rl(epoch)
            else:
                train_loss = self._train_epoch_e2e(epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            print(f"Epoch {epoch+1}/{self.max_epochs} — "
                  f"train_loss: {train_loss:.4f}, "
                  f"lr: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Log
            if self.writer:
                self.writer.add_scalar('train/loss', train_loss, epoch)
                self.writer.add_scalar(
                    'train/lr', self.optimizer.param_groups[0]['lr'], epoch
                )

            # Evaluation
            if (epoch + 1) % self.eval_interval == 0:
                metrics = self.evaluate(epoch)
                map50 = metrics.get('mAP50', 0.0)
                print(f"  -> mAP@0.5: {map50:.4f}")

                if map50 > self.best_map:
                    self.best_map = map50
                    self._save_checkpoint(epoch, is_best=True)
                    print(f"  -> New best! Saved to {self.work_dir}/best.pth")

            # 定期保存
            if (epoch + 1) % 4 == 0:
                self._save_checkpoint(epoch)

        print(f"\nTraining finished. Best mAP@0.5: {self.best_map:.4f}")
        return self.best_map

    def _train_epoch_e2e(self, epoch: int) -> float:
        """端到端训练 (RAW-Adapter 模式)."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} [E2E]')
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in t.items()} for t in targets]

            # Forward
            loss_dict = self.model(images, targets)

            if isinstance(loss_dict, dict):
                loss = sum(loss_dict.values())
            else:
                loss = loss_dict

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % self.log_interval == 0:
                pbar.set_postfix(loss=f'{loss.item():.4f}')

        return total_loss / max(num_batches, 1)

    def _train_epoch_rl(self, epoch: int) -> float:
        """RL + gradient 训练 (AdaptiveISP 模式).

        1. ISP 参数: 用 detection loss 的梯度更新
        2. RL agent: 用 REINFORCE 更新 (reward = -det_loss - cost_penalty)
        """
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} [RL]')
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in t.items()} for t in targets]

            # Forward: ISP -> Detector
            loss_dict = self.model(images, targets)

            if isinstance(loss_dict, dict):
                det_loss = sum(loss_dict.values())
            else:
                det_loss = loss_dict

            # --- 更新 ISP 参数 (通过 detection loss gradient) ---
            self.optimizer.zero_grad()
            det_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(
                self.model.isp.parameters(), 10.0
            )
            self.optimizer.step()

            # --- 更新 RL agent ---
            if (self.rl_optimizer is not None
                    and hasattr(self.model.isp, 'get_rl_loss')):
                # Reward: 负 detection loss (越小越好)
                B = images.shape[0]
                reward = -det_loss.detach().expand(B)
                rl_loss = self.model.isp.get_rl_loss(reward)

                self.rl_optimizer.zero_grad()
                rl_loss.backward()
                self.rl_optimizer.step()

            total_loss += det_loss.item()
            num_batches += 1

            if (batch_idx + 1) % self.log_interval == 0:
                # 打印 ISP pipeline 信息
                isp_config = self.model.isp.get_isp_config()
                pipeline_str = ' -> '.join(isp_config.get('pipeline', []))
                pbar.set_postfix(
                    loss=f'{det_loss.item():.4f}',
                    pipe=pipeline_str[:40],
                )

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def evaluate(self, epoch: int = 0) -> Dict[str, float]:
        """统一评估."""
        self.model.eval()
        self.evaluator.reset()

        for images, targets in tqdm(self.val_loader, desc='Evaluating'):
            images = images.to(self.device)

            # Forward (推理模式)
            predictions = self.model(images)

            if isinstance(predictions, dict):
                # 训练模式下返回了 loss, 跳过
                continue

            self.evaluator.update(predictions, targets)

        metrics = self.evaluator.evaluate()

        if self.writer:
            for k, v in metrics.items():
                self.writer.add_scalar(f'val/{k}', v, epoch)

        return metrics

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        state = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'best_map': self.best_map,
        }
        if is_best:
            path = os.path.join(self.work_dir, 'best.pth')
        else:
            path = os.path.join(self.work_dir, f'epoch_{epoch+1}.pth')
        torch.save(state, path)
