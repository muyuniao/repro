#!/bin/bash
# ==========================================================
# Batch run for ResNet-50 Experiments: E1, E3, E5, E6
# GPU 0: E1, E3
# GPU 1: E5, E6
# ==========================================================
set -e

PYTHON=/home/duomeitinrfx/.conda/envs/plate/bin/python
WORK_DIR=/home/duomeitinrfx/users/yunhe/reproduce/idea2
cd $WORK_DIR

export HF_HUB_OFFLINE=1

mkdir -p results/logs results/metrics results/checkpoints

# 1. 启动 E1 (GPU 0)
echo "Starting E1 (ResNet-50 S2.1 CE) on GPU 0..."
CUDA_VISIBLE_DEVICES=0 $PYTHON -u train_stage2.py --config configs/stage2_E1.yaml > results/logs/stage2_E1.log 2>&1 &
PID_E1=$!

# 2. 启动 E3 (GPU 0)
echo "Starting E3 (ResNet-50 S2.3 Frozen CE) on GPU 0..."
CUDA_VISIBLE_DEVICES=0 $PYTHON -u train_stage2.py --config configs/stage2_E3.yaml > results/logs/stage2_E3.log 2>&1 &
PID_E3=$!

# 3. 启动 E5 (GPU 1)
echo "Starting E5 (ResNet-50 Baseline CORAL) on GPU 1..."
CUDA_VISIBLE_DEVICES=1 $PYTHON -u train_stage2.py --config configs/stage2_E5.yaml > results/logs/stage2_E5.log 2>&1 &
PID_E5=$!

# 4. 启动 E6 (GPU 1)
echo "Starting E6 (ResNet-50 S2.1 CORAL) on GPU 1..."
CUDA_VISIBLE_DEVICES=1 $PYTHON -u train_stage2.py --config configs/stage2_E6.yaml > results/logs/stage2_E6.log 2>&1 &
PID_E6=$!

echo "Waiting for all experiments to finish..."
wait $PID_E1
wait $PID_E3
wait $PID_E5
wait $PID_E6

echo "Running WeChat notification..."
$PYTHON $WORK_DIR/notification.py

echo "All ResNet experiments completed!"
