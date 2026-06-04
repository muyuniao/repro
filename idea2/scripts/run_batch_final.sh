#!/bin/bash
# ==========================================================
# Batch run for final experiments: E7 and E12
# GPU 0: E7 (ResNet-50 S2.2 CORAL)
# GPU 1: E12 (ViT Baseline CORAL)
# ==========================================================
set -e

PYTHON=/home/duomeitinrfx/.conda/envs/plate/bin/python
WORK_DIR=/home/duomeitinrfx/users/yunhe/reproduce/idea2
cd $WORK_DIR

export HF_HUB_OFFLINE=1

mkdir -p results/logs results/metrics results/checkpoints

# 1. 启动 E7 (GPU 0)
echo "Starting E7 (ResNet-50 S2.2 CORAL) on GPU 0..."
CUDA_VISIBLE_DEVICES=0 $PYTHON -u train_stage2.py --config configs/stage2_E7.yaml > results/logs/stage2_E7.log 2>&1 &
PID_E7=$!

# 2. 启动 E12 (GPU 1)
echo "Starting E12 (ViT Baseline CORAL) on GPU 1..."
CUDA_VISIBLE_DEVICES=1 $PYTHON -u train_stage2.py --config configs/stage2_E12.yaml > results/logs/stage2_E12.log 2>&1 &
PID_E12=$!

echo "Waiting for final experiments to finish..."
wait $PID_E7
wait $PID_E12

echo "Running WeChat notification..."
$PYTHON $WORK_DIR/notification.py

echo "Final experiments completed!"
