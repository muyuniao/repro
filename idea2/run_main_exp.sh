#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
export HF_HUB_OFFLINE=1

echo "Starting Main Experiment E2 (ResNet-50)"

echo "--> Running Stage 1 Ranking Pretraining (80 epochs)..."
/home/duomeitinrfx/.conda/envs/plate/bin/python train_stage1.py --encoder resnet50 --epochs 80 --batch_size 16 --lr 1e-4 > stage1_E2.log 2>&1

echo "--> Running Stage 2 Classification Fine-tuning (S2.2 Joint Training, 50 epochs)..."
/home/duomeitinrfx/.conda/envs/plate/bin/python train_stage2.py --encoder resnet50 --stage1_ckpt results/checkpoints/stage1_resnet50.pt --mode s2.2 --loss ce --exp_name E2_resnet50_s2.2 --epochs 50 --batch_size 32 --lr 1e-3 > stage2_E2.log 2>&1

echo "Main Experiment E2 Finished."
