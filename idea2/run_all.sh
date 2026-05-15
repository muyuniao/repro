#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com

echo "Starting Stage 1 (Ranking Pretraining)..."
/home/duomeitinrfx/.conda/envs/plate/bin/python train_stage1.py --encoder resnet50 --epochs 5 --batch_size 16 --lr 1e-4

echo "Starting Stage 2 (Classification Fine-tuning)..."
/home/duomeitinrfx/.conda/envs/plate/bin/python train_stage2.py --encoder resnet50 --stage1_ckpt results/checkpoints/stage1_resnet50.pt --epochs 5 --batch_size 32 --lr 1e-3

echo "Experiment Finished."
