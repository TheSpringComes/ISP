"""
LOD Dataset — Low-light Object Detection dataset.

数据集结构 (推荐按 SimROD 的方式组织):
    data/LOD/
    ├── RAW_dark/          # 低光 RAW 图像 (.png, 4通道 RGGB 或已 demosaic 的 3通道)
    │   ├── 2.png
    │   ├── 4.png
    │   └── ...
    ├── RAW_normal/        # 正常光 RAW 图像 (可选, 用于对比)
    ├── annotations/
    │   ├── lod_dark_train.json   # COCO 格式 (推荐)
    │   └── lod_dark_val.json
    ├── xml_annotations/          # VOC XML 格式 (原始格式)
    │   ├── 2.xml
    │   └── ...
    └── raw_cr2/                  # 原始 .CR2 文件 (可选)

支持多种 RAW 输入:
  1. 预处理的 PNG (已 pack 为 4 通道 RGGB, HxWx4)
  2. 预处理的 NPY (numpy array, 4 通道)
  3. 原始 .CR2 文件 (需要 rawpy)
  4. 已 demosaic 的 3 通道 PNG (传统方式)

LOD 类别 (8 classes):
  bicycle, bus, car, motorbike, person, dog, cat, bottle
  (注意: 不同论文可能使用略有不同的类别名, 请以你的 annotation 为准)
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple
import cv2


# LOD 默认类别 — 请根据你的 annotation 文件调整
LOD_CLASSES = (
    'bicycle', 'bus', 'car', 'motorbike',
    'person', 'dog', 'cat', 'bottle',
)


class LODDataset(Dataset):
    """统一的 LOD 数据集类.

    同时支持 COCO JSON 和 VOC XML 两种标注格式.
    输出统一的 dict 格式, 可被任意 detector 消费.
    """

    def __init__(
        self,
        img_dir: str,
        ann_file: str = None,           # COCO JSON 路径
        xml_dir: str = None,            # VOC XML 目录 (与 ann_file 二选一)
        classes: tuple = LOD_CLASSES,
        img_size: int = 512,
        input_channels: int = 4,        # 4=RGGB pack, 3=demosaiced RGB
        raw_suffix: str = '.png',
        transforms=None,
        is_training: bool = True,
    ):
        super().__init__()
        self.img_dir = img_dir
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.num_classes = len(classes)
        self.img_size = img_size
        self.input_channels = input_channels
        self.raw_suffix = raw_suffix
        self.transforms = transforms
        self.is_training = is_training

        # 加载标注
        if ann_file is not None and os.path.exists(ann_file):
            self.ann_format = 'coco'
            self._load_coco_annotations(ann_file)
        elif xml_dir is not None and os.path.exists(xml_dir):
            self.ann_format = 'voc'
            self._load_voc_annotations(xml_dir)
        else:
            raise ValueError(
                f"Must provide either ann_file (COCO JSON) or xml_dir (VOC XML).\n"
                f"  ann_file={ann_file}\n  xml_dir={xml_dir}"
            )

        print(f"[LODDataset] Loaded {len(self.data_list)} images, "
              f"format={self.ann_format}, classes={self.num_classes}, "
              f"input_channels={self.input_channels}")

    # ------------------------------------------------------------------
    # Annotation loaders
    # ------------------------------------------------------------------
    def _load_coco_annotations(self, ann_file: str):
        """加载 COCO 格式标注."""
        with open(ann_file, 'r') as f:
            coco = json.load(f)

        # 建立 category 映射
        cat_id_to_name = {}
        for cat in coco.get('categories', []):
            cat_id_to_name[cat['id']] = cat['name']

        # 建立 image_id -> annotations 映射
        img_id_to_anns = {}
        for ann in coco.get('annotations', []):
            img_id = ann['image_id']
            if img_id not in img_id_to_anns:
                img_id_to_anns[img_id] = []
            img_id_to_anns[img_id].append(ann)

        self.data_list = []
        for img_info in coco['images']:
            img_id = img_info['id']
            filename = img_info['file_name']
            w, h = img_info['width'], img_info['height']

            anns = img_id_to_anns.get(img_id, [])
            boxes, labels = [], []
            for ann in anns:
                x, y, bw, bh = ann['bbox']  # COCO: [x, y, w, h]
                if bw > 0 and bh > 0:
                    boxes.append([x, y, x + bw, y + bh])  # -> xyxy
                    # 映射 category
                    cat_name = cat_id_to_name.get(ann['category_id'], '')
                    if cat_name.lower() in self.class_to_idx:
                        labels.append(self.class_to_idx[cat_name.lower()])
                    else:
                        labels.append(ann['category_id'] - 1)  # fallback

            if len(boxes) == 0 and self.is_training:
                continue  # 跳过无标注图像

            self.data_list.append({
                'filename': filename,
                'width': w,
                'height': h,
                'boxes': boxes,
                'labels': labels,
            })

    def _load_voc_annotations(self, xml_dir: str):
        """加载 VOC XML 格式标注."""
        xml_files = sorted([f for f in os.listdir(xml_dir) if f.endswith('.xml')])
        self.data_list = []

        for xml_file in xml_files:
            tree = ET.parse(os.path.join(xml_dir, xml_file))
            root = tree.getroot()

            # 图像文件名
            fn_elem = root.find('filename')
            if fn_elem is not None:
                filename = fn_elem.text
            else:
                filename = xml_file.replace('.xml', self.raw_suffix)

            size_elem = root.find('size')
            w = int(size_elem.find('width').text)
            h = int(size_elem.find('height').text)

            boxes, labels = [], []
            for obj in root.findall('object'):
                name = obj.find('name').text.lower().strip()
                if name not in self.class_to_idx:
                    continue

                bbox = obj.find('bndbox')
                x1 = float(bbox.find('xmin').text)
                y1 = float(bbox.find('ymin').text)
                x2 = float(bbox.find('xmax').text)
                y2 = float(bbox.find('ymax').text)

                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(self.class_to_idx[name])

            if len(boxes) == 0 and self.is_training:
                continue

            self.data_list.append({
                'filename': filename,
                'width': w,
                'height': h,
                'boxes': boxes,
                'labels': labels,
            })

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------
    def _load_raw_image(self, filepath: str) -> np.ndarray:
        """加载 RAW 图像, 返回 (H, W, C) numpy array, float32, [0, 1].

        支持:
          - .npy: 直接加载
          - .png/.tiff: 用 cv2 加载 (可能是 4 通道 RGGB 或 3 通道 RGB)
          - .CR2/.cr2: 用 rawpy 加载
        """
        ext = os.path.splitext(filepath)[1].lower()

        if ext == '.npy':
            img = np.load(filepath).astype(np.float32)
            if img.max() > 1.0:
                img = img / img.max()
            return img

        elif ext in ('.cr2', '.nef', '.arw', '.dng'):
            import rawpy
            with rawpy.imread(filepath) as raw:
                # 获取 Bayer pattern, pack 为 4 通道
                bayer = raw.raw_image_visible.astype(np.float32)
                black = raw.black_level_per_channel[0]
                white = raw.white_level
                bayer = (bayer - black) / (white - black)
                bayer = np.clip(bayer, 0, 1)

                H, W = bayer.shape
                img = np.stack([
                    bayer[0::2, 0::2],  # R
                    bayer[0::2, 1::2],  # G1
                    bayer[1::2, 0::2],  # G2
                    bayer[1::2, 1::2],  # B
                ], axis=-1)  # (H//2, W//2, 4)
                return img

        else:
            # PNG / TIFF — 用 cv2 加载
            img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(f"Cannot read image: {filepath}")

            if img.ndim == 2:
                # 单通道 Bayer, 需要 pack
                H, W = img.shape
                img = img.astype(np.float32)
                if img.max() > 1.0:
                    img = img / max(img.max(), 1.0)
                packed = np.stack([
                    img[0::2, 0::2],
                    img[0::2, 1::2],
                    img[1::2, 0::2],
                    img[1::2, 1::2],
                ], axis=-1)
                return packed

            # 多通道
            img = img.astype(np.float32)
            if img.max() > 255:
                img = img / max(img.max(), 1.0)  # 16-bit
            elif img.max() > 1.0:
                img = img / 255.0

            if img.shape[-1] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            return img

    # ------------------------------------------------------------------
    # Transforms
    # ------------------------------------------------------------------
    def _resize_and_pad(
        self, img: np.ndarray, boxes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Resize 到 img_size, 保持比例, letterbox padding."""
        h, w = img.shape[:2]
        scale = min(self.img_size / h, self.img_size / w)
        new_h, new_w = int(h * scale), int(w * scale)

        img_resized = cv2.resize(img, (new_w, new_h),
                                 interpolation=cv2.INTER_LINEAR)

        # Pad 到 img_size x img_size
        canvas = np.zeros((self.img_size, self.img_size, img.shape[-1]),
                          dtype=img.dtype)
        pad_h = (self.img_size - new_h) // 2
        pad_w = (self.img_size - new_w) // 2
        canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = img_resized

        # 调整 boxes
        if len(boxes) > 0:
            boxes = boxes * scale
            boxes[:, [0, 2]] += pad_w
            boxes[:, [1, 3]] += pad_h

        return canvas, boxes, scale

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        info = self.data_list[idx]

        # 加载图像
        img_path = os.path.join(self.img_dir, info['filename'])
        if not os.path.exists(img_path):
            # 尝试替换后缀
            base = os.path.splitext(info['filename'])[0]
            img_path = os.path.join(self.img_dir, base + self.raw_suffix)

        img = self._load_raw_image(img_path)  # (H, W, C), float32, [0,1]

        boxes = np.array(info['boxes'], dtype=np.float32).reshape(-1, 4)
        labels = np.array(info['labels'], dtype=np.int64)

        # Resize
        img, boxes, scale = self._resize_and_pad(img, boxes)

        # 确保通道数一致
        if self.input_channels == 4 and img.shape[-1] == 3:
            # 3ch -> 4ch: 复制 G 通道
            img = np.concatenate([
                img[:, :, :1], img[:, :, 1:2],
                img[:, :, 1:2], img[:, :, 2:3]
            ], axis=-1)
        elif self.input_channels == 3 and img.shape[-1] == 4:
            # 4ch RGGB -> 3ch: 平均两个 G
            r = img[:, :, 0:1]
            g = (img[:, :, 1:2] + img[:, :, 2:3]) / 2.0
            b = img[:, :, 3:4]
            img = np.concatenate([r, g, b], axis=-1)

        # To tensor: (C, H, W)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()

        # Clamp
        img_tensor = img_tensor.clamp(0, 1)

        target = {
            'boxes': torch.from_numpy(boxes).float(),
            'labels': torch.from_numpy(labels).long(),
            'image_id': torch.tensor([idx]),
            'orig_size': torch.tensor([info['height'], info['width']]),
            'scale': torch.tensor([scale]),
        }

        if self.transforms is not None:
            img_tensor, target = self.transforms(img_tensor, target)

        return img_tensor, target

    @staticmethod
    def collate_fn(batch):
        """自定义 collate, 因为不同图像的 box 数量不同."""
        images = torch.stack([item[0] for item in batch])
        targets = [item[1] for item in batch]
        return images, targets
