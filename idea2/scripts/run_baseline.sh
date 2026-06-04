#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1


echo "Starting Baseline (Direct Classification Fine-tuning) for 50 Epochs..."
/home/duomeitinrfx/.conda/envs/plate/bin/python train_stage2.py --encoder resnet50 --epochs 50 --batch_size 32 --lr 1e-3

echo "Baseline Experiment Finished."
