#!/usr/bin/env python3
"""
统一训练入口.

Usage:
    # 训练 AdaptiveISP on LOD
    python tools/train.py --config configs/adaptive_isp/adaptive_isp_lod.yaml

    # 训练 RAW-Adapter on LOD
    python tools/train.py --config configs/raw_adapter/raw_adapter_lod.yaml

    # 覆盖配置
    python tools/train.py --config configs/adaptive_isp/adaptive_isp_lod.yaml \
        --override training.batch_size=8 training.max_epochs=12
"""

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader

# 将项目根目录加入 path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.utils.config import Config
from src.datasets.lod import LODDataset
from src.isp.adaptive_isp.adaptive_isp import AdaptiveISPModule
from src.isp.raw_adapter.raw_adapter import RAWAdapterModule
from src.detectors.yolov3_detector import YOLOv3Detector
from src.detectors.sparse_rcnn_detector import SparseRCNNDetector
from src.models.isp_detector import ISPDetector
from src.engine.trainer import Trainer


# ====================================================================
# Builder functions
# ====================================================================

def build_dataset(cfg, split='train'):
    """根据配置构建数据集."""
    data_cfg = cfg.data
    is_train = (split == 'train')

    ann_file = data_cfg.get('train_ann') if is_train else data_cfg.get('val_ann')
    xml_dir = data_cfg.get('train_xml_dir') if is_train else data_cfg.get('val_xml_dir')
    classes = tuple(data_cfg.get('classes', []))

    dataset = LODDataset(
        img_dir=data_cfg.get('img_dir'),
        ann_file=ann_file,
        xml_dir=xml_dir,
        classes=classes if classes else None,
        img_size=data_cfg.get('img_size', 512),
        input_channels=data_cfg.get('input_channels', 4),
        raw_suffix=data_cfg.get('raw_suffix', '.png'),
        is_training=is_train,
    )
    return dataset


def build_isp(cfg):
    """根据配置构建 ISP 模块."""
    isp_cfg = cfg.isp
    isp_type = isp_cfg.get('type', 'AdaptiveISP')

    if isp_type == 'AdaptiveISP':
        return AdaptiveISPModule(
            in_channels=isp_cfg.get('in_channels', 4),
            max_stages=isp_cfg.get('max_stages', 5),
            cost_penalty=isp_cfg.get('cost_penalty', 0.01),
            agent_hidden_dim=isp_cfg.get('agent_hidden_dim', 256),
        )
    elif isp_type == 'RAWAdapter':
        bb_ch = isp_cfg.get('backbone_channels', [256, 512, 1024, 2048])
        return RAWAdapterModule(
            in_channels=isp_cfg.get('in_channels', 4),
            backbone_channels=bb_ch,
            adapter_dim=isp_cfg.get('adapter_dim', 64),
            use_qal=isp_cfg.get('use_qal', True),
        )
    else:
        raise ValueError(f"Unknown ISP type: {isp_type}")


def build_detector(cfg):
    """根据配置构建检测器."""
    det_cfg = cfg.detector
    det_type = det_cfg.get('type', 'YOLOv3')
    num_classes = det_cfg.get('num_classes', 8)
    pretrained = det_cfg.get('pretrained', True)
    freeze = det_cfg.get('freeze', False)

    if det_type in ('YOLOv3', 'yolov3'):
        return YOLOv3Detector(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze=freeze,
        )
    elif det_type in ('SparseRCNN', 'sparse_rcnn', 'FasterRCNN'):
        return SparseRCNNDetector(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze=freeze,
        )
    else:
        raise ValueError(f"Unknown detector type: {det_type}")


def build_optimizer(params, opt_cfg):
    """构建优化器."""
    opt_type = opt_cfg.get('type', 'Adam')
    lr = opt_cfg.get('lr', 0.001)
    wd = opt_cfg.get('weight_decay', 0.0001)

    if opt_type == 'Adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    elif opt_type == 'AdamW':
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    elif opt_type == 'SGD':
        return torch.optim.SGD(params, lr=lr, weight_decay=wd, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {opt_type}")


def build_scheduler(optimizer, sch_cfg, max_epochs):
    """构建学习率调度器."""
    if sch_cfg is None:
        return None
    sch_type = sch_cfg.get('type', 'CosineAnnealingLR')
    if sch_type == 'CosineAnnealingLR':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=sch_cfg.get('T_max', max_epochs),
            eta_min=sch_cfg.get('eta_min', 1e-6),
        )
    elif sch_type == 'StepLR':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sch_cfg.get('step_size', 8),
            gamma=sch_cfg.get('gamma', 0.1),
        )
    return None


# ====================================================================
# Main
# ====================================================================

def main():
    parser = argparse.ArgumentParser(description='Unified ISP-Det Training')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file')
    parser.add_argument('--override', nargs='*', default=[],
                        help='Override config: key=value pairs')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda / cpu)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    args = parser.parse_args()

    # 1. 加载配置
    cfg = Config.from_file(args.config)

    # 应用 override
    for kv in args.override:
        key, val = kv.split('=', 1)
        # 简单类型转换
        try:
            val = int(val)
        except ValueError:
            try:
                val = float(val)
            except ValueError:
                if val.lower() in ('true', 'false'):
                    val = val.lower() == 'true'
        # 设置嵌套键
        keys = key.split('.')
        d = cfg.to_dict()
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = val

    # 2. Device
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 3. 构建数据集
    print("\n--- Building datasets ---")
    train_dataset = build_dataset(cfg, 'train')
    val_dataset = build_dataset(cfg, 'val')

    train_cfg = cfg.training
    batch_size = train_cfg.get('batch_size', 4)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.data.get('num_workers', 4),
        collate_fn=LODDataset.collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.data.get('num_workers', 4),
        collate_fn=LODDataset.collate_fn,
        pin_memory=True,
    )
    print(f"Train: {len(train_dataset)} images, Val: {len(val_dataset)} images")

    # 4. 构建模型
    print("\n--- Building model ---")
    isp = build_isp(cfg)
    detector = build_detector(cfg)
    use_adapter = cfg.model.get('use_model_adapter', False)

    model = ISPDetector(
        isp=isp,
        detector=detector,
        use_model_adapter=use_adapter,
    )

    # 打印参数量
    isp_params = sum(p.numel() for p in isp.parameters())
    det_params = sum(p.numel() for p in detector.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ISP params:      {isp_params:>12,}")
    print(f"Detector params: {det_params:>12,}")
    print(f"Total params:    {total_params:>12,}")
    print(f"Trainable:       {trainable:>12,}")
    print(f"ISP type:        {cfg.isp.get('type')}")
    print(f"Detector type:   {cfg.detector.get('type')}")
    print(f"Model adapter:   {use_adapter}")

    # 5. 构建优化器
    print("\n--- Building optimizer ---")
    training_mode = train_cfg.get('mode', 'e2e')
    opt_cfg = train_cfg.get('optimizer', {})

    if training_mode == 'rl':
        # RL 模式: 只优化 ISP 参数 (不含 RL agent)
        isp_params_list = [p for n, p in isp.named_parameters()
                           if 'agent' not in n and p.requires_grad]
        optimizer = build_optimizer(isp_params_list, opt_cfg)

        # RL agent 单独的 optimizer
        rl_cfg = train_cfg.get('rl_optimizer', {'type': 'Adam', 'lr': 3e-4})
        agent_params = [p for n, p in isp.named_parameters()
                        if 'agent' in n and p.requires_grad]
        rl_optimizer = build_optimizer(agent_params, rl_cfg) if agent_params else None
    else:
        # E2E 模式: 优化所有可训练参数
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = build_optimizer(trainable_params, opt_cfg)
        rl_optimizer = None

    max_epochs = train_cfg.get('max_epochs', 12)
    sch_cfg = train_cfg.get('scheduler')
    scheduler = build_scheduler(optimizer, sch_cfg, max_epochs)

    # 6. Resume
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        print(f"\nResuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['model_state'], strict=False)
        if 'optimizer_state' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state'])
        start_epoch = ckpt.get('epoch', 0)

    # 7. 训练
    print(f"\n--- Starting training ({training_mode}) ---")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_classes=cfg.data.get('num_classes', 8),
        work_dir=cfg.experiment.get('work_dir', 'work_dirs/exp'),
        max_epochs=max_epochs,
        log_interval=train_cfg.get('log_interval', 50),
        eval_interval=train_cfg.get('eval_interval', 1),
        training_mode=training_mode,
        rl_optimizer=rl_optimizer,
    )

    trainer.train()


if __name__ == '__main__':
    main()
