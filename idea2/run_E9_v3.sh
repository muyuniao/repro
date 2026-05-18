#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
/home/duomeitinrfx/.conda/envs/plate/bin/python train_stage2.py \
    --encoder vit_base_patch16_224.augreg_in21k_ft_in1k \
    --stage1_ckpt results/checkpoints/stage1_vit_base_patch16_224.augreg_in21k_ft_in1k.pt \
    --mode s2.2 --loss ce --exp_name E9_vit_s2.2_rerun --epochs 50 --batch_size 32 --lr 1e-3
