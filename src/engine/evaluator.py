"""统一的检测评估器 — 计算 VOC/COCO style mAP."""

import torch
import numpy as np
from typing import Dict, List
from collections import defaultdict


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
    """计算 IoU. box format: (N, 4) xyxy."""
    x1 = np.maximum(box1[:, None, 0], box2[None, :, 0])
    y1 = np.maximum(box1[:, None, 1], box2[None, :, 1])
    x2 = np.minimum(box1[:, None, 2], box2[None, :, 2])
    y2 = np.minimum(box1[:, None, 3], box2[None, :, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    union = area1[:, None] + area2[None, :] - inter
    return inter / np.maximum(union, 1e-8)


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """计算 AP (11-point interpolation or all-point)."""
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    # 单调递减
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # 找到 recall 变化的点
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)


class DetectionEvaluator:
    """检测评估器.

    收集所有预测和真实标注, 最后统一计算 mAP.
    """

    def __init__(self, num_classes: int, iou_thresholds: List[float] = None):
        self.num_classes = num_classes
        if iou_thresholds is None:
            iou_thresholds = [0.5]  # 默认 mAP@0.5
        self.iou_thresholds = iou_thresholds
        self.reset()

    def reset(self):
        self.predictions = []  # list of (boxes, labels, scores)
        self.ground_truths = []  # list of (boxes, labels)

    def update(
        self,
        preds: List[Dict],
        targets: List[Dict],
    ):
        """添加一个 batch 的结果.

        preds: list of dict with 'boxes', 'labels', 'scores'
        targets: list of dict with 'boxes', 'labels'
        """
        for pred, gt in zip(preds, targets):
            if isinstance(pred, dict):
                p_boxes = pred['boxes'].cpu().numpy()
                p_labels = pred['labels'].cpu().numpy()
                p_scores = pred['scores'].cpu().numpy()
            else:
                p_boxes = np.zeros((0, 4))
                p_labels = np.zeros(0, dtype=np.int64)
                p_scores = np.zeros(0)

            g_boxes = gt['boxes'].cpu().numpy()
            g_labels = gt['labels'].cpu().numpy()

            self.predictions.append((p_boxes, p_labels, p_scores))
            self.ground_truths.append((g_boxes, g_labels))

    def evaluate(self) -> Dict[str, float]:
        """计算所有指标."""
        results = {}

        for iou_thr in self.iou_thresholds:
            aps = []
            for cls_id in range(self.num_classes):
                ap = self._compute_class_ap(cls_id, iou_thr)
                aps.append(ap)

            mean_ap = np.mean(aps)
            results[f'mAP@{iou_thr:.2f}'] = mean_ap

            if iou_thr == 0.5:
                results['mAP50'] = mean_ap

        # mAP@0.5:0.95
        if len(self.iou_thresholds) > 1:
            all_aps = []
            for iou_thr in np.arange(0.5, 1.0, 0.05):
                aps = []
                for cls_id in range(self.num_classes):
                    aps.append(self._compute_class_ap(cls_id, iou_thr))
                all_aps.append(np.mean(aps))
            results['mAP50:95'] = np.mean(all_aps)

        return results

    def _compute_class_ap(self, cls_id: int, iou_thr: float) -> float:
        """计算单个类别在指定 IoU 阈值下的 AP."""
        # 收集该类别的所有预测和 GT
        all_scores = []
        all_tp = []
        n_gt = 0

        for (p_boxes, p_labels, p_scores), (g_boxes, g_labels) in zip(
            self.predictions, self.ground_truths
        ):
            # 该类别的 GT
            gt_mask = g_labels == cls_id
            gt_boxes_cls = g_boxes[gt_mask]
            n_gt += len(gt_boxes_cls)
            gt_matched = np.zeros(len(gt_boxes_cls), dtype=bool)

            # 该类别的 predictions
            pred_mask = p_labels == cls_id
            pred_boxes_cls = p_boxes[pred_mask]
            pred_scores_cls = p_scores[pred_mask]

            # 按 score 降序排序
            order = np.argsort(-pred_scores_cls)
            pred_boxes_cls = pred_boxes_cls[order]
            pred_scores_cls = pred_scores_cls[order]

            for i in range(len(pred_boxes_cls)):
                all_scores.append(pred_scores_cls[i])

                if len(gt_boxes_cls) == 0:
                    all_tp.append(False)
                    continue

                ious = compute_iou(
                    pred_boxes_cls[i:i+1], gt_boxes_cls
                )[0]
                best_iou_idx = np.argmax(ious)

                if ious[best_iou_idx] >= iou_thr and not gt_matched[best_iou_idx]:
                    all_tp.append(True)
                    gt_matched[best_iou_idx] = True
                else:
                    all_tp.append(False)

        if n_gt == 0:
            return 0.0

        # 按 score 排序
        all_scores = np.array(all_scores)
        all_tp = np.array(all_tp)
        order = np.argsort(-all_scores)
        all_tp = all_tp[order]

        cum_tp = np.cumsum(all_tp)
        cum_fp = np.cumsum(~all_tp)

        recalls = cum_tp / n_gt
        precisions = cum_tp / (cum_tp + cum_fp)

        return compute_ap(recalls, precisions)
