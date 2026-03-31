#!/bin/bash
# ============================================================
# 准备 LOD 数据 (VOC XML -> COCO JSON)
# ============================================================
echo "=== Preparing LOD dataset ==="
python tools/prepare_lod.py \
    --img-dir data/LOD/RAW_dark \
    --xml-dir data/LOD/xml_annotations \
    --output-dir data/LOD/annotations \
    --train-count 1800

echo "Done."
