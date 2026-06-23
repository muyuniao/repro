#!/bin/bash

# 默认参数
DRY_RUN_FLAG=""
EPOCHS=200

# 检查是否传入了 --dry-run
for arg in "$@"
do
    if [ "$arg" == "--dry_run" ] || [ "$arg" == "--dry-run" ]; then
        DRY_RUN_FLAG="--dry_run"
        EPOCHS=1
        echo "=== Running 5-Fold in DRY-RUN mode (1 batch check) ==="
    fi
done

# 激活 conda 环境
source /opt/anaconda3/bin/activate plate

# 获取通知脚本路径
NOTIFY_PATH=""
for path in "../notification.py" "./notification.py"
do
    if [ -f "$path" ]; then
        NOTIFY_PATH="$path"
        break
    fi
done

# 循环运行 5 个 Fold
for fold in 0 1 2 3 4
do
    # 检查当前 Fold 是否已经跑完指定 Epoch 数的评估
    EVAL_LOG="vit-b_adience_eval_fold${fold}.txt"
    if [ "$DRY_RUN_FLAG" == "" ] && [ -f "$EVAL_LOG" ] && grep -q "Epoch $((EPOCHS - 1))," "$EVAL_LOG"; then
        echo "========================================="
        echo "Fold $fold has already completed all $EPOCHS epochs. Skipping."
        echo "========================================="
        continue
    fi

    echo "========================================="
    echo "Starting Fold $fold (Epochs: $EPOCHS)..."
    echo "========================================="
    
    CUDA_VISIBLE_DEVICES=1 http_proxy= https_proxy= python main.py \
      --num_classes 8 \
      --images_root "/home/duomeitinrfx/data/Adience/" \
      --data_file "/home/duomeitinrfx/data/Adience/Adience_train_fold${fold}.csv" \
      --val_data_file "/home/duomeitinrfx/data/Adience/Adience_test_fold${fold}.csv" \
      --vit_pretrained_path "/home/duomeitinrfx/users/yunhe/models/vit_base_patch16_224.augreg_in21k_ft_in1k.safetensors" \
      --epochs $EPOCHS \
      --batch_size 64 \
      --fold ${fold} \
      --no_notify \
      ${DRY_RUN_FLAG}
      
    # 如果某折训练失败，发送通知并直接退出
    if [ $? -ne 0 ]; then
        echo "Error: Fold $fold failed! Aborting."
        if [ -n "$NOTIFY_PATH" ]; then
            echo "Sending error notification..."
            python "$NOTIFY_PATH"
        fi
        exit 1
    fi
done

echo "=== All 5 Folds completed successfully! ==="
if [ -n "$NOTIFY_PATH" ]; then
    echo "Sending success notification..."
    python "$NOTIFY_PATH"
fi
