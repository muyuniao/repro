#!/bin/bash
# ==========================================================
# 多种子验证实验：4 组超参 × 3 种子
# GPU 使用策略：尽量跑满一张 GPU，再使用下一张
# ==========================================================
set -e

PYTHON=/home/duomeitinrfx/.conda/envs/plate/bin/python
WORK_DIR=/home/duomeitinrfx/users/yunhe/reproduce/idea2
cd $WORK_DIR

export HF_HUB_OFFLINE=1

mkdir -p results/logs results/metrics results/checkpoints

# ---- 4 组候选超参配置 ----
# 配置格式: "名称 lr lambda_rank lr_encoder_ratio lr_ranker_ratio"
CONFIGS=(
    "cfgA 0.004356 0.9009 0.001086 0.054632"
    "cfgB 0.000830 0.9785 0.021708 0.070373"
    "cfgC 0.004954 0.8787 0.003730 0.062041"
    "cfgD 0.001160 0.7132 0.006769 0.070650"
)

SEEDS=(42 123 456)

# ---- 确定可用 GPU ----
# GPU 0 和 3 空闲，每个约 6GB 显存 → 每张 GPU 可同时跑 3 个任务
# 策略：先跑满 GPU 0，再用 GPU 3
GPU_LIST=(0 3)
JOBS_PER_GPU=3

echo "======================================================"
echo "  Multi-Seed Validation: 4 configs × 3 seeds = 12 jobs"
echo "  GPUs: ${GPU_LIST[*]} | Jobs per GPU: $JOBS_PER_GPU"
echo "  Start: $(date)"
echo "======================================================"

# ---- 构建所有任务 ----
declare -a ALL_TASKS
for cfg_str in "${CONFIGS[@]}"; do
    read -r cfg_name lr lam lr_enc lr_rnk <<< "$cfg_str"
    for seed in "${SEEDS[@]}"; do
        ALL_TASKS+=("${cfg_name} ${lr} ${lam} ${lr_enc} ${lr_rnk} ${seed}")
    done
done

TOTAL=${#ALL_TASKS[@]}
echo "Total tasks: $TOTAL"

# ---- 分批运行 ----
MAX_PARALLEL=$((${#GPU_LIST[@]} * JOBS_PER_GPU))  # 2 GPU × 3 = 6 并行
BATCH=0
IDX=0

while [ $IDX -lt $TOTAL ]; do
    BATCH=$((BATCH + 1))
    BATCH_SIZE=0
    PIDS=()
    
    echo ""
    echo "======== Batch $BATCH (starting task $((IDX+1))) ========"
    
    for gpu_idx in "${!GPU_LIST[@]}"; do
        GPU=${GPU_LIST[$gpu_idx]}
        for ((j=0; j<JOBS_PER_GPU; j++)); do
            if [ $IDX -ge $TOTAL ]; then
                break 2
            fi
            
            read -r cfg_name lr lam lr_enc lr_rnk seed <<< "${ALL_TASKS[$IDX]}"
            EXP_NAME="seed_val_${cfg_name}_s${seed}"
            LOG_FILE="results/logs/${EXP_NAME}.log"
            
            echo "[Task $((IDX+1))/$TOTAL] GPU=$GPU  cfg=$cfg_name  seed=$seed  → $EXP_NAME"
            
            CUDA_VISIBLE_DEVICES=$GPU $PYTHON train_stage2.py \
                --encoder vit_base_patch16_224.augreg_in21k_ft_in1k \
                --stage1_ckpt results/checkpoints/stage1_vit_base_patch16_224.augreg_in21k_ft_in1k.pt \
                --mode s2.2 \
                --loss ce \
                --exp_name "$EXP_NAME" \
                --epochs 50 \
                --batch_size 32 \
                --lr "$lr" \
                --lambda_rank "$lam" \
                --lr_encoder_ratio "$lr_enc" \
                --lr_ranker_ratio "$lr_rnk" \
                --seed "$seed" \
                > "$LOG_FILE" 2>&1 &
            
            PIDS+=($!)
            IDX=$((IDX + 1))
            BATCH_SIZE=$((BATCH_SIZE + 1))
        done
    done
    
    echo "Waiting for $BATCH_SIZE tasks (PIDs: ${PIDS[*]})..."
    
    FAILED=0
    for pid in "${PIDS[@]}"; do
        if ! wait $pid; then
            FAILED=$((FAILED + 1))
            echo "WARNING: PID $pid exited with error"
        fi
    done
    
    echo "Batch $BATCH done. Failed: $FAILED/$BATCH_SIZE"
done

echo ""
echo "======================================================"
echo "  All $TOTAL tasks completed at $(date)"
echo "======================================================"

# ---- 生成汇总报告 ----
echo ""
echo "Generating summary report..."
$PYTHON << 'PYEOF'
import json, glob, os
import numpy as np
from collections import defaultdict

# 收集所有 seed validation 结果
files = sorted(glob.glob('results/metrics/seed_val_*.json'))
if not files:
    print("ERROR: No seed validation results found!")
    exit(1)

# 按配置分组
groups = defaultdict(list)
for f in files:
    with open(f) as fh:
        d = json.load(fh)
    basename = os.path.basename(f).replace('.json', '')
    # seed_val_cfgA_s42 → cfgA
    parts = basename.split('_')
    cfg = parts[2]  # cfgA, cfgB, etc.
    groups[cfg].append(d)

print("=" * 80)
print("多种子验证结果汇总")
print("=" * 80)

results = []
for cfg in sorted(groups.keys()):
    trials = groups[cfg]
    test_maes = [t['test_mae'] for t in trials]
    test_accs = [t['test_acc'] for t in trials]
    test_qwks = [t['test_qwk'] for t in trials]
    test_spearmans = [t['test_spearman'] for t in trials]
    val_maes = [t['val_best_mae'] for t in trials]
    
    result = {
        'cfg': cfg,
        'n_seeds': len(trials),
        'test_mae_mean': np.mean(test_maes),
        'test_mae_std': np.std(test_maes),
        'test_acc_mean': np.mean(test_accs),
        'test_qwk_mean': np.mean(test_qwks),
        'test_spearman_mean': np.mean(test_spearmans),
        'val_mae_mean': np.mean(val_maes),
        'val_mae_std': np.std(val_maes),
        'lr': trials[0]['args']['lr'],
        'lambda_rank': trials[0]['args']['lambda_rank'],
    }
    results.append(result)
    
    print(f"\n--- {cfg} (lr={result['lr']}, λ_rank={result['lambda_rank']}) ---")
    print(f"  Test MAE:  {result['test_mae_mean']:.4f} ± {result['test_mae_std']:.4f}  (seeds: {[f'{m:.4f}' for m in test_maes]})")
    print(f"  Test ACC:  {result['test_acc_mean']:.4f}")
    print(f"  Test QWK:  {result['test_qwk_mean']:.4f}")
    print(f"  Spearman:  {result['test_spearman_mean']:.4f}")
    print(f"  Val MAE:   {result['val_mae_mean']:.4f} ± {result['val_mae_std']:.4f}")

# 按 test_mae_mean 排名
results.sort(key=lambda x: x['test_mae_mean'])
print("\n" + "=" * 80)
print("最终排名 (按 Test MAE 均值)")
print("=" * 80)
for i, r in enumerate(results):
    marker = " ← BEST" if i == 0 else ""
    print(f"  #{i+1}: {r['cfg']}  Test MAE = {r['test_mae_mean']:.4f} ± {r['test_mae_std']:.4f}  (QWK={r['test_qwk_mean']:.4f}){marker}")

# 保存完整报告为 JSON
report = {'results': results, 'ranking': [r['cfg'] for r in results]}
with open('results/seed_validation_report.json', 'w') as f:
    json.dump(report, f, indent=2)
print(f"\nReport saved to results/seed_validation_report.json")
PYEOF

echo ""
echo "======================================================"
echo "  Running WeChat notification..."
echo "======================================================"
$PYTHON $WORK_DIR/notification.py

echo "All done! $(date)"
