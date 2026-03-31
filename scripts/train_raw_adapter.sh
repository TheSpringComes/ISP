#!/bin/bash
# ============================================================
# Train RAW-Adapter on LOD
# ============================================================
echo "=== Training RAW-Adapter on LOD ==="
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    --config configs/raw_adapter/raw_adapter_lod.yaml
