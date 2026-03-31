#!/usr/bin/env python3
"""
统一测试入口.

Usage:
    python tools/test.py \
        --config configs/adaptive_isp/adaptive_isp_lod.yaml \
        --checkpoint work_dirs/adaptive_isp_lod/best.pth \
        --visualize
"""

import os
import sys
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.utils.config import Config
from src.datasets.lod import LODDataset
from src.engine.evaluator import DetectionEvaluator


def main():
    parser = argparse.ArgumentParser(description='Unified ISP-Det Testing')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--visualize', action='store_true',
                        help='保存 ISP 输出的可视化结果')
    parser.add_argument('--vis-dir', type=str, default='vis_results')
    args = parser.parse_args()

    cfg = Config.from_file(args.config)
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # 复用 train.py 的 builder
    from train import build_dataset, build_isp, build_detector
    from src.models.isp_detector import ISPDetector

    # 数据集
    val_dataset = build_dataset(cfg, 'val')
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=cfg.data.get('num_workers', 4),
        collate_fn=LODDataset.collate_fn,
    )

    # 模型
    isp = build_isp(cfg)
    detector = build_detector(cfg)
    model = ISPDetector(
        isp=isp,
        detector=detector,
        use_model_adapter=cfg.model.get('use_model_adapter', False),
    ).to(device)

    # 加载权重
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['model_state'], strict=False)
    print(f"Loaded checkpoint from {args.checkpoint}")
    if 'best_map' in ckpt:
        print(f"  (recorded best mAP: {ckpt['best_map']:.4f})")

    # 评估
    model.eval()
    evaluator = DetectionEvaluator(
        num_classes=cfg.data.get('num_classes', 8),
        iou_thresholds=[0.5],
    )

    vis_count = 0
    if args.visualize:
        os.makedirs(args.vis_dir, exist_ok=True)

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc='Testing'):
            images = images.to(device)
            predictions = model(images)

            if isinstance(predictions, dict):
                continue

            evaluator.update(predictions, targets)

            # 可视化 ISP 输出
            if args.visualize and vis_count < 50:
                rgb_output = model.get_isp_output(images)
                for i in range(min(rgb_output.shape[0], 5)):
                    img_np = rgb_output[i].cpu().permute(1, 2, 0).numpy()
                    img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
                    import cv2
                    cv2.imwrite(
                        os.path.join(args.vis_dir, f'isp_output_{vis_count}.png'),
                        cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR),
                    )
                    vis_count += 1

    # 打印结果
    metrics = evaluator.evaluate()
    print("\n" + "=" * 50)
    print("Evaluation Results:")
    print("=" * 50)
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}")
    print("=" * 50)

    # ISP 配置
    isp_config = model.isp.get_isp_config()
    if isp_config:
        print(f"\nISP Config: {isp_config}")


if __name__ == '__main__':
    main()
