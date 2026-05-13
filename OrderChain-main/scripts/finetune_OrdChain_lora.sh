#!/bin/bash
#/home/duomeitinrfx/users/yunhe/models
#/home/duomeitinrfx/data/Adience
export PYTHONPATH="$PWD:$PYTHONPATH"
export TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT="OrderChain_Reproduction"  # 自定义你的 WandB 项目大分类名称
export WANDB_NAME="SingleGPU_ZeRO2_Run2"        # 自定义本次实验具体的 Run 名称
# 使用空闲的单张 GPU 3
include=localhost:3

python -m deepspeed.launcher.runner --include $include llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /home/duomeitinrfx/users/yunhe/models/llava-v1.5-7b \
    --data_path /home/duomeitinrfx/data/Adience/Adience_llava_train.json \
    --image_folder /home/duomeitinrfx/data/Adience/faces/ \
    --vision_tower /home/duomeitinrfx/users/yunhe/models/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /home/duomeitinrfx/users/yunhe/reproduce/OrderChain-main/checkpoints_v2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --num_train_epochs 2 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 5 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
