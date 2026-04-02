"""
YOLOv3 + Darknet-53 — AdaptiveISP 使用的 detector.

论文: "YOLOv3 is utilized as the detection model in all methods.
The pre-trained YOLOv3 model remains unaltered throughout the
training process."

结构:
  Darknet-53 Backbone → 3-scale FPN Neck → 3 Detection Heads
  输入: (B, 3, 512, 512)
  输出: 3 个尺度的 feature maps → decode → (boxes, scores, classes)

此实现可以:
  1. 加载 Darknet 格式预训练权重 (.weights)
  2. 整个 detector freeze, 只传梯度给 ISP
  3. forward 时输出 detection loss (用于反传给 ISP)
  4. inference 时输出 decoded predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


# ====================================================================
# Darknet-53 Building Blocks
# ====================================================================

class ConvBnLeaky(nn.Module):
    """Conv + BatchNorm + LeakyReLU (Darknet 标准模块)."""

    def __init__(self, in_ch, out_ch, ksize, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = (ksize - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, ksize, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DarknetResBlock(nn.Module):
    """Darknet Residual Block: 1x1 降维 + 3x3 升维."""

    def __init__(self, ch):
        super().__init__()
        mid = ch // 2
        self.conv1 = ConvBnLeaky(ch, mid, 1)
        self.conv2 = ConvBnLeaky(mid, ch, 3)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class DarknetStage(nn.Module):
    """Downsample (stride-2 conv) + N 个 ResBlocks."""

    def __init__(self, in_ch, out_ch, num_blocks):
        super().__init__()
        self.downsample = ConvBnLeaky(in_ch, out_ch, 3, stride=2)
        self.blocks = nn.Sequential(
            *[DarknetResBlock(out_ch) for _ in range(num_blocks)]
        )

    def forward(self, x):
        return self.blocks(self.downsample(x))


# ====================================================================
# Darknet-53 Backbone
# ====================================================================

class Darknet53(nn.Module):
    """Darknet-53 backbone.

    输出 3 个尺度的 feature maps (stride 8, 16, 32):
      C3: (B, 256,  H/8,  W/8)
      C4: (B, 512,  H/16, W/16)
      C5: (B, 1024, H/32, W/32)
    """

    def __init__(self):
        super().__init__()
        self.stem = ConvBnLeaky(3, 32, 3)           # /1
        self.stage1 = DarknetStage(32,   64,  1)    # /2
        self.stage2 = DarknetStage(64,  128,  2)    # /4
        self.stage3 = DarknetStage(128, 256,  8)    # /8   → C3
        self.stage4 = DarknetStage(256, 512,  8)    # /16  → C4
        self.stage5 = DarknetStage(512, 1024, 4)    # /32  → C5

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        c3 = self.stage3(x)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        return c3, c4, c5


# ====================================================================
# YOLOv3 Neck (FPN-like)
# ====================================================================

class YOLOConvSet(nn.Module):
    """5-conv set: 1x1 → 3x3 → 1x1 → 3x3 → 1x1."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        mid = out_ch
        self.convs = nn.Sequential(
            ConvBnLeaky(in_ch, mid, 1),
            ConvBnLeaky(mid, mid * 2, 3),
            ConvBnLeaky(mid * 2, mid, 1),
            ConvBnLeaky(mid, mid * 2, 3),
            ConvBnLeaky(mid * 2, mid, 1),
        )

    def forward(self, x):
        return self.convs(x)


class YOLONeck(nn.Module):
    """YOLOv3 FPN neck: top-down 融合 3 个尺度."""

    def __init__(self):
        super().__init__()
        # Scale 1 (最大尺度, stride=32)
        self.conv_set1 = YOLOConvSet(1024, 512)
        # Upsample + concat for scale 2
        self.up_conv1 = ConvBnLeaky(512, 256, 1)
        self.conv_set2 = YOLOConvSet(256 + 512, 256)
        # Upsample + concat for scale 3
        self.up_conv2 = ConvBnLeaky(256, 128, 1)
        self.conv_set3 = YOLOConvSet(128 + 256, 128)

    def forward(self, c3, c4, c5):
        # Scale 1 (stride=32)
        p5 = self.conv_set1(c5)

        # Scale 2 (stride=16)
        up1 = F.interpolate(self.up_conv1(p5), scale_factor=2, mode='nearest')
        p4 = self.conv_set2(torch.cat([up1, c4], dim=1))

        # Scale 3 (stride=8)
        up2 = F.interpolate(self.up_conv2(p4), scale_factor=2, mode='nearest')
        p3 = self.conv_set3(torch.cat([up2, c3], dim=1))

        return p3, p4, p5   # stride 8, 16, 32


# ====================================================================
# YOLOv3 Detection Head
# ====================================================================

class YOLOHead(nn.Module):
    """单个尺度的 YOLO 检测头.

    输出 (B, num_anchors*(5+num_classes), H, W)
    其中 5 = (tx, ty, tw, th, objectness)
    """

    def __init__(self, in_ch, num_anchors=3, num_classes=8):
        super().__init__()
        out_ch = num_anchors * (5 + num_classes)
        self.conv = nn.Sequential(
            ConvBnLeaky(in_ch, in_ch * 2, 3),
            nn.Conv2d(in_ch * 2, out_ch, 1, bias=True),
        )

    def forward(self, x):
        return self.conv(x)


# ====================================================================
# YOLOv3 Complete Model
# ====================================================================

# COCO 预训练的 default anchors (scaled for 512x512)
DEFAULT_ANCHORS = [
    [(10, 13), (16, 30), (33, 23)],       # stride 8  (small)
    [(30, 61), (62, 45), (59, 119)],       # stride 16 (medium)
    [(116, 90), (156, 198), (373, 326)],   # stride 32 (large)
]


class YOLOv3(nn.Module):
    """完整 YOLOv3 检测器.

    Args:
        num_classes: 类别数 (不含 background)
        anchors: 3 组 anchors, 对应 3 个检测尺度
        img_size: 输入图像尺寸
        conf_thresh: 推理时的 confidence 阈值
        nms_thresh: NMS 阈值
    """

    def __init__(
        self,
        num_classes: int = 8,
        anchors: list = None,
        img_size: int = 512,
        conf_thresh: float = 0.01,
        nms_thresh: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.anchors = anchors or DEFAULT_ANCHORS
        self.img_size = img_size
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.strides = [8, 16, 32]

        self.backbone = Darknet53()
        self.neck = YOLONeck()

        # 3 个检测头
        self.head_s = YOLOHead(128, 3, num_classes)  # stride 8
        self.head_m = YOLOHead(256, 3, num_classes)  # stride 16
        self.head_l = YOLOHead(512, 3, num_classes)  # stride 32

        # 预计算 anchor tensor
        self._anchor_tensors = {}

    def _get_anchor_tensor(self, scale_idx, grid_h, grid_w, device):
        """预计算 anchor 尺寸 tensor."""
        key = (scale_idx, grid_h, grid_w, device)
        if key not in self._anchor_tensors:
            anc = torch.tensor(
                self.anchors[scale_idx], dtype=torch.float32, device=device
            )  # (3, 2)
            self._anchor_tensors[key] = anc
        return self._anchor_tensors[key]

    def _decode_predictions(
        self,
        raw_pred: torch.Tensor,
        scale_idx: int,
    ) -> torch.Tensor:
        """将 raw detection output 解码为 (x1, y1, x2, y2, obj, cls...).

        raw_pred: (B, A*(5+C), H, W)
        返回:    (B, A*H*W, 5+C) — 绝对坐标 xyxy
        """
        B, _, H, W = raw_pred.shape
        A = 3
        C = self.num_classes
        stride = self.strides[scale_idx]

        pred = raw_pred.view(B, A, 5 + C, H, W).permute(0, 1, 3, 4, 2)
        pred = pred.contiguous().view(B, A * H * W, 5 + C)

        # Grid offsets
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=raw_pred.device, dtype=torch.float32),
            torch.arange(W, device=raw_pred.device, dtype=torch.float32),
            indexing='ij',
        )
        grid = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)
        grid = grid.view(1, 1, H, W, 2).expand(1, A, -1, -1, -1)
        grid = grid.reshape(1, A * H * W, 2)

        # Anchor sizes
        anc = self._get_anchor_tensor(scale_idx, H, W, raw_pred.device)
        anc = anc.view(1, A, 1, 1, 2).expand(1, -1, H, W, -1)
        anc = anc.reshape(1, A * H * W, 2)

        # Decode
        xy = (torch.sigmoid(pred[..., :2]) + grid) * stride
        wh = torch.exp(pred[..., 2:4].clamp(-10, 10)) * anc
        obj = torch.sigmoid(pred[..., 4:5])
        cls = torch.sigmoid(pred[..., 5:])

        # xyxy
        x1y1 = xy - wh / 2
        x2y2 = xy + wh / 2

        return torch.cat([x1y1, x2y2, obj, cls], dim=-1)

    def forward(
        self,
        images: torch.Tensor,
        targets: List[Dict] = None,
    ):
        """
        训练时 (targets 非 None): 返回 loss dict
        推理时 (targets is None): 返回 predictions list
        """
        c3, c4, c5 = self.backbone(images)
        p3, p4, p5 = self.neck(c3, c4, c5)

        raw_s = self.head_s(p3)  # stride 8
        raw_m = self.head_m(p4)  # stride 16
        raw_l = self.head_l(p5)  # stride 32

        # 用 targets 是否存在判断, 而不是 self.training
        # (AdaptiveISP 中 detector 冻结在 eval mode, 但仍需返回 loss)
        if targets is not None:
            return self._compute_loss(
                [raw_s, raw_m, raw_l], targets
            )
        else:
            return self._inference([raw_s, raw_m, raw_l])

    # ----------------------------------------------------------
    # Loss (simplified YOLO loss)
    # ----------------------------------------------------------
    def _compute_loss(
        self,
        raw_preds: List[torch.Tensor],
        targets: List[Dict],
    ) -> Dict[str, torch.Tensor]:
        """简化的 YOLO loss.

        完整 YOLO loss 比较复杂, 这里实现核心部分:
        - objectness loss (BCE)
        - box loss (CIoU)
        - class loss (BCE)
        """
        device = raw_preds[0].device
        B = raw_preds[0].shape[0]

        loss_obj = torch.tensor(0.0, device=device)
        loss_box = torch.tensor(0.0, device=device)
        loss_cls = torch.tensor(0.0, device=device)

        for scale_idx, raw_pred in enumerate(raw_preds):
            decoded = self._decode_predictions(raw_pred, scale_idx)
            # decoded: (B, N, 5+C)

            stride = self.strides[scale_idx]

            for b in range(B):
                preds_b = decoded[b]  # (N, 5+C)
                gt_boxes = targets[b]['boxes'].to(device)   # (M, 4) xyxy
                gt_labels = targets[b]['labels'].to(device)  # (M,)

                if gt_boxes.numel() == 0:
                    # 无 GT: 所有 obj 应该为 0
                    loss_obj = loss_obj + F.binary_cross_entropy(
                        preds_b[:, 4], torch.zeros(preds_b.shape[0], device=device),
                        reduction='mean'
                    )
                    continue

                # 匹配: 每个 GT 找 IoU 最大的 prediction
                pred_boxes = preds_b[:, :4]   # (N, 4)
                ious = self._batch_iou(pred_boxes, gt_boxes)  # (N, M)
                max_iou_per_pred, _ = ious.max(dim=1)         # (N,)

                # Objectness target: IoU > 0.5 的 pred 标为正
                obj_target = (max_iou_per_pred > 0.5).float()

                # 负样本: IoU < 0.4 的忽略
                ignore_mask = (max_iou_per_pred > 0.4) & (max_iou_per_pred <= 0.5)

                # Objectness loss
                obj_pred = preds_b[:, 4]
                bce_weight = torch.ones_like(obj_target)
                bce_weight[ignore_mask] = 0  # 忽略区域
                loss_obj = loss_obj + F.binary_cross_entropy(
                    obj_pred, obj_target, weight=bce_weight, reduction='mean'
                )

                # 正样本的 box + class loss
                pos_mask = obj_target > 0.5
                if pos_mask.sum() > 0:
                    pos_preds = preds_b[pos_mask]
                    # 每个正 pred 对应的 GT index
                    _, gt_idx = ious[pos_mask].max(dim=1)

                    # Box regression loss (L1)
                    loss_box = loss_box + F.smooth_l1_loss(
                        pos_preds[:, :4], gt_boxes[gt_idx], reduction='mean'
                    ) * 0.05  # scale down

                    # Classification loss
                    cls_target = torch.zeros(
                        pos_preds.shape[0], self.num_classes, device=device
                    )
                    for i, gi in enumerate(gt_idx):
                        label = gt_labels[gi].item()
                        if 0 <= label < self.num_classes:
                            cls_target[i, label] = 1.0

                    loss_cls = loss_cls + F.binary_cross_entropy(
                        pos_preds[:, 5:], cls_target, reduction='mean'
                    )

        n_scales = len(raw_preds)
        return {
            'loss_obj': loss_obj / (B * n_scales),
            'loss_box': loss_box / (B * n_scales),
            'loss_cls': loss_cls / (B * n_scales),
        }

    # ----------------------------------------------------------
    # Inference
    # ----------------------------------------------------------
    def _inference(
        self, raw_preds: List[torch.Tensor]
    ) -> List[Dict[str, torch.Tensor]]:
        """Decode + NMS → 最终 detections."""
        all_decoded = []
        for scale_idx, raw_pred in enumerate(raw_preds):
            decoded = self._decode_predictions(raw_pred, scale_idx)
            all_decoded.append(decoded)

        # 合并 3 个尺度
        merged = torch.cat(all_decoded, dim=1)  # (B, N_total, 5+C)
        B = merged.shape[0]

        results = []
        for b in range(B):
            preds = merged[b]  # (N, 5+C)
            obj_scores = preds[:, 4]
            cls_scores = preds[:, 5:]

            # Combined score
            scores, class_ids = cls_scores.max(dim=1)
            scores = scores * obj_scores

            # Filter by confidence
            keep = scores > self.conf_thresh
            if keep.sum() == 0:
                results.append({
                    'boxes': torch.zeros(0, 4, device=preds.device),
                    'labels': torch.zeros(0, dtype=torch.long, device=preds.device),
                    'scores': torch.zeros(0, device=preds.device),
                })
                continue

            boxes = preds[keep, :4]
            scores = scores[keep]
            class_ids = class_ids[keep]

            # NMS per class
            keep_nms = self._nms(boxes, scores, class_ids, self.nms_thresh)
            results.append({
                'boxes': boxes[keep_nms],
                'labels': class_ids[keep_nms],
                'scores': scores[keep_nms],
            })

        return results

    # ----------------------------------------------------------
    # Utilities
    # ----------------------------------------------------------
    @staticmethod
    def _batch_iou(boxes1, boxes2):
        """IoU between two sets of xyxy boxes. (N,4) x (M,4) -> (N,M)."""
        x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
        y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
        x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
        y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])
        inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
        a1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        a2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = a1[:, None] + a2[None, :] - inter
        return inter / union.clamp(1e-8)

    @staticmethod
    def _nms(boxes, scores, class_ids, iou_thresh):
        """Per-class NMS."""
        # 用 class offset 技巧实现 per-class NMS
        max_coord = boxes.max()
        offsets = class_ids.float() * (max_coord + 1)
        shifted = boxes + offsets[:, None]

        order = scores.argsort(descending=True)
        keep = []
        while order.numel() > 0:
            i = order[0].item()
            keep.append(i)
            if order.numel() == 1:
                break
            ious = YOLOv3._batch_iou(
                shifted[i:i+1], shifted[order[1:]]
            )[0]
            mask = ious < iou_thresh
            order = order[1:][mask]

        return torch.tensor(keep, dtype=torch.long, device=boxes.device)


# ====================================================================
# Wrapper for ISPDetector interface
# ====================================================================

class YOLOv3Detector(nn.Module):
    """YOLOv3 detector wrapper, 匹配 ISPDetector 接口.

    Args:
        num_classes: 类别数
        pretrained: 是否加载预训练权重 (COCO darknet weights)
        freeze: 是否冻结参数 (AdaptiveISP 需要冻结)
        pretrained_weights: .weights 或 .pth 文件路径
    """

    def __init__(
        self,
        num_classes: int = 8,
        pretrained: bool = True,
        freeze: bool = True,
        pretrained_weights: str = None,
        img_size: int = 512,
        input_channels: int = 3,
    ):
        super().__init__()
        self.model = YOLOv3(
            num_classes=num_classes,
            img_size=img_size,
        )

        if pretrained_weights and pretrained:
            self._load_weights(pretrained_weights)

        if freeze:
            self._freeze()

    def _freeze(self):
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    def _load_weights(self, path: str):
        """加载预训练权重 (.pth state_dict)."""
        state = torch.load(path, map_location='cpu')
        if isinstance(state, dict) and 'model_state' in state:
            state = state['model_state']
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        print(f"[YOLOv3] Loaded weights: "
              f"missing={len(missing)}, unexpected={len(unexpected)}")

    def forward(
        self,
        images: torch.Tensor,
        targets: List[Dict] = None,
    ):
        """
        训练: 返回 loss dict
        推理: 返回 list of dicts {boxes, labels, scores}
        """
        return self.model(images, targets)

    def train(self, mode=True):
        """AdaptiveISP 中 detector 始终保持 eval (BN 固定)."""
        super().train(mode)
        # 如果 frozen, 始终保持 eval
        if not any(p.requires_grad for p in self.model.parameters()):
            self.model.eval()
        return self
