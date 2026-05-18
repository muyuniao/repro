#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
export HF_HUB_OFFLINE=1
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
/home/duomeitinrfx/.conda/envs/plate/bin/python train_stage2.py \
    --encoder vit_base_patch16_224.augreg_in21k_ft_in1k \
    --mode s2.1 --loss ce --exp_name E11_vit_baseline --epochs 50 --batch_size 32 --lr 1e-3
