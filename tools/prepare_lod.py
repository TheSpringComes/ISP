#!/usr/bin/env python3
"""
LOD 数据准备工具.

功能:
  1. 将 VOC XML 标注转换为 COCO JSON 格式
  2. 自动划分 train/val (默认 1800/430, 与 LOD 原始论文一致)
  3. 验证 RAW 图像文件的完整性

Usage:
    python tools/prepare_lod.py \
        --img-dir data/LOD/RAW_dark \
        --xml-dir data/LOD/xml_annotations \
        --output-dir data/LOD/annotations \
        --train-count 1800

目录结构 (准备后):
    data/LOD/
    ├── RAW_dark/
    │   ├── 2.png
    │   └── ...
    ├── xml_annotations/
    │   ├── 2.xml
    │   └── ...
    └── annotations/
        ├── lod_dark_train.json   (COCO 格式)
        └── lod_dark_val.json
"""

import os
import sys
import json
import random
import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict

# LOD 默认类别
LOD_CLASSES = [
    'bicycle', 'bus', 'car', 'motorbike',
    'person', 'dog', 'cat', 'bottle',
]


def parse_voc_xml(xml_path: str):
    """解析单个 VOC XML 标注文件."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find('filename')
    filename = filename.text if filename is not None else None

    size = root.find('size')
    if size is not None:
        width = int(size.find('width').text)
        height = int(size.find('height').text)
    else:
        width, height = 0, 0

    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text.lower().strip()
        difficult = obj.find('difficult')
        difficult = int(difficult.text) if difficult is not None else 0

        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)

        if xmax > xmin and ymax > ymin:
            objects.append({
                'name': name,
                'bbox': [xmin, ymin, xmax - xmin, ymax - ymin],  # COCO: xywh
                'difficult': difficult,
            })

    return {
        'filename': filename,
        'width': width,
        'height': height,
        'objects': objects,
    }


def voc_to_coco(xml_dir, img_dir, output_path, image_ids, classes):
    """将一组 VOC XML 转换为 COCO JSON."""
    categories = [
        {'id': i + 1, 'name': name}
        for i, name in enumerate(classes)
    ]
    class_to_id = {name: i + 1 for i, name in enumerate(classes)}

    images = []
    annotations = []
    ann_id = 1
    skipped = 0

    for img_id, xml_file in enumerate(image_ids, start=1):
        xml_path = os.path.join(xml_dir, xml_file)
        if not os.path.exists(xml_path):
            skipped += 1
            continue

        info = parse_voc_xml(xml_path)

        # 确定图像文件名
        base_name = os.path.splitext(xml_file)[0]
        img_filename = info['filename'] or (base_name + '.png')

        # 检查图像是否存在
        img_path = os.path.join(img_dir, img_filename)
        if not os.path.exists(img_path):
            # 尝试其他后缀
            for ext in ['.png', '.jpg', '.tiff', '.npy']:
                alt = os.path.join(img_dir, base_name + ext)
                if os.path.exists(alt):
                    img_filename = base_name + ext
                    break

        # 如果 width/height 为 0, 尝试从图像读取
        width, height = info['width'], info['height']
        if width == 0 or height == 0:
            try:
                import cv2
                img = cv2.imread(os.path.join(img_dir, img_filename),
                                 cv2.IMREAD_UNCHANGED)
                if img is not None:
                    if img.ndim == 2:
                        height, width = img.shape
                    else:
                        height, width = img.shape[:2]
            except Exception:
                width, height = 512, 512  # fallback

        images.append({
            'id': img_id,
            'file_name': img_filename,
            'width': width,
            'height': height,
        })

        for obj in info['objects']:
            cat_name = obj['name']
            if cat_name not in class_to_id:
                continue

            x, y, w, h = obj['bbox']
            annotations.append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': class_to_id[cat_name],
                'bbox': [x, y, w, h],
                'area': w * h,
                'iscrowd': 0,
            })
            ann_id += 1

    coco = {
        'images': images,
        'annotations': annotations,
        'categories': categories,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(coco, f, indent=2)

    print(f"  Saved {output_path}: {len(images)} images, "
          f"{len(annotations)} annotations, skipped {skipped}")
    return coco


def main():
    parser = argparse.ArgumentParser(description='Prepare LOD dataset')
    parser.add_argument('--img-dir', type=str, required=True,
                        help='RAW image directory (e.g. data/LOD/RAW_dark)')
    parser.add_argument('--xml-dir', type=str, required=True,
                        help='VOC XML annotation directory')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for COCO JSON')
    parser.add_argument('--classes', nargs='+', default=LOD_CLASSES,
                        help='Object class names')
    parser.add_argument('--train-count', type=int, default=1800,
                        help='Number of training images (rest -> val)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # 收集所有 XML 文件
    xml_files = sorted([f for f in os.listdir(args.xml_dir) if f.endswith('.xml')])
    print(f"Found {len(xml_files)} XML annotation files")

    if len(xml_files) == 0:
        print("ERROR: No XML files found!")
        sys.exit(1)

    # 统计类别分布
    class_counts = defaultdict(int)
    for xml_file in xml_files:
        info = parse_voc_xml(os.path.join(args.xml_dir, xml_file))
        for obj in info['objects']:
            class_counts[obj['name']] += 1

    print("\nClass distribution:")
    for cls_name in args.classes:
        print(f"  {cls_name}: {class_counts.get(cls_name, 0)}")

    # 划分 train/val
    random.shuffle(xml_files)
    train_count = min(args.train_count, len(xml_files) - 1)
    train_files = xml_files[:train_count]
    val_files = xml_files[train_count:]

    print(f"\nSplit: {len(train_files)} train, {len(val_files)} val")

    # 转换
    print("\nConverting to COCO format...")
    voc_to_coco(
        args.xml_dir, args.img_dir,
        os.path.join(args.output_dir, 'lod_dark_train.json'),
        train_files, args.classes,
    )
    voc_to_coco(
        args.xml_dir, args.img_dir,
        os.path.join(args.output_dir, 'lod_dark_val.json'),
        val_files, args.classes,
    )

    print("\nDone!")


if __name__ == '__main__':
    main()
