#!/bin/bash

# 确保激活 plate 环境并清理代理
source /opt/anaconda3/bin/activate plate
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

# 创建 checkpoints 保存目录
mkdir -p ./ckpts/

echo "===================================================================="
echo "🛡️ GECO 4-GPU Parallel Hyperparameter Sweeps Daemon"
echo "===================================================================="

# 1. 自动轮询等待离线特征提取完成
echo ">>> Checking feature extraction process status..."
while ps -ef | grep -i extract_features.py | grep -v grep > /dev/null; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] extract_features.py is still running. Waiting 60 seconds..."
    sleep 60
done

echo "🎉 Feature extraction completed successfully! Launching parallel grid sweeps..."

# 2. 多卡网格并发扫描，跑不同尺度的博弈协同奖惩超参 (pair_bonus, coop_bonus, kl_target)
# 充分占满 4 张 RTX 3090 卡，极大缩短寻优周期！

# 🏎️ GPU 0 (Sweep 1)
python -u run_geco.py \
  --device cuda:0 \
  --pair_bonus 0.5 \
  --coop_bonus 0.5 \
  --kl_target 0.001 \
  --ckpt ./ckpts/sweep_gpu0_pb0.5_cb0.5.pth > train_gpu0.log 2>&1 &
echo "[GPU 0] Sweep 1 launched. (pair_bonus=0.5, coop_bonus=0.5, kl_target=0.001)"

# 🏎️ GPU 1 (Sweep 2)
python -u run_geco.py \
  --device cuda:1 \
  --pair_bonus 1.0 \
  --coop_bonus 1.0 \
  --kl_target 0.001 \
  --ckpt ./ckpts/sweep_gpu1_pb1.0_cb1.0.pth > train_gpu1.log 2>&1 &
echo "[GPU 1] Sweep 2 launched. (pair_bonus=1.0, coop_bonus=1.0, kl_target=0.001)"

# 🏎️ GPU 2 (Sweep 3)
python -u run_geco.py \
  --device cuda:2 \
  --pair_bonus 1.5 \
  --coop_bonus 1.5 \
  --kl_target 0.002 \
  --ckpt ./ckpts/sweep_gpu2_pb1.5_cb1.5.pth > train_gpu2.log 2>&1 &
echo "[GPU 2] Sweep 3 launched. (pair_bonus=1.5, coop_bonus=1.5, kl_target=0.002)"

# 🏎️ GPU 3 (Sweep 4 - GPU 3 特征提取后立刻完美复用)
python -u run_geco.py \
  --device cuda:3 \
  --pair_bonus 2.0 \
  --coop_bonus 2.0 \
  --kl_target 0.005 \
  --ckpt ./ckpts/sweep_gpu3_pb2.0_cb2.0.pth > train_gpu3.log 2>&1 &
echo "[GPU 3] Sweep 4 launched. (pair_bonus=2.0, coop_bonus=2.0, kl_target=0.005)"

echo "--------------------------------------------------------------------"
echo "🌟 All sweeps are running in background! Use 'tail -f train_gpuX.log' to monitor."
echo "===================================================================="
