#!/bin/bash
# ==========================================================
# 运行实验 E10 (ViT S2.3 冻结 Encoder 验证)
# ==========================================================
set -e

PYTHON=/home/duomeitinrfx/.conda/envs/plate/bin/python
WORK_DIR=/home/duomeitinrfx/users/yunhe/reproduce/idea2
cd $WORK_DIR

export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

mkdir -p results/logs results/metrics results/checkpoints

echo "=== Starting E10 (ViT S2.3 Frozen Encoder) on GPU 0 ==="
$PYTHON -u train_stage2.py --config configs/stage2_E10.yaml > results/logs/stage2_E10.log 2>&1

echo "=== Running WeChat notification... ==="
$PYTHON $WORK_DIR/notification.py

echo "E10 Finished successfully!"
