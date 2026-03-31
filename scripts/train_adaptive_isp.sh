#!/bin/bash
# ============================================================
# Train AdaptiveISP on LOD
# ============================================================
echo "=== Training AdaptiveISP on LOD ==="
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    --config configs/adaptive_isp/adaptive_isp_lod.yaml
