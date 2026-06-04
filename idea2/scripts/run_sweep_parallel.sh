#!/bin/bash
# 激活环境 (如果以独立脚本运行，确保使用正确的 python 二进制文件，这在 Python 层面通过 sys.executable 完成了)
export HF_HUB_OFFLINE=1

echo "Initializing hyperparameter sweep..."
mkdir -p results/metrics results/logs results/checkpoints

# 如果有旧数据库，在此处清理（为了断点续传，我们屏蔽此行以保留已完成的试验）
# rm -f results/optuna_study.db

# 试验配置
N_TRIALS=${N_TRIALS:-8}    # 每个 GPU 默认跑 8 次
EPOCHS=${EPOCHS:-50}       # 默认 50 个 epoch

# 先用单进程初始化 Study，创建好 SQLite 数据库结构，防止多进程并发初始化时发生 sqlite3.OperationalError 冲突
echo "Initializing Optuna SQLite database structure..."
/home/duomeitinrfx/.conda/envs/plate/bin/python -c "
import optuna
optuna.create_study(study_name='vit_s22_tuning', storage='sqlite:///results/optuna_study.db', load_if_exists=True)
print('Optuna study initialized successfully.')
"

echo "Starting parallel workers on GPU 0, 2, 3..."

# 启动 GPU 0 Worker
CUDA_VISIBLE_DEVICES=0 /home/duomeitinrfx/.conda/envs/plate/bin/python sweep_optuna.py --n_trials $N_TRIALS --epochs $EPOCHS > results/logs/worker_gpu0.log 2>&1 &
GPU0_PID=$!
echo "Worker 1 started on GPU 0 (PID: $GPU0_PID)"

# 启动 GPU 2 Worker
CUDA_VISIBLE_DEVICES=2 /home/duomeitinrfx/.conda/envs/plate/bin/python sweep_optuna.py --n_trials $N_TRIALS --epochs $EPOCHS > results/logs/worker_gpu2.log 2>&1 &
GPU2_PID=$!
echo "Worker 2 started on GPU 2 (PID: $GPU2_PID)"

# 启动 GPU 3 Worker
CUDA_VISIBLE_DEVICES=3 /home/duomeitinrfx/.conda/envs/plate/bin/python sweep_optuna.py --n_trials $N_TRIALS --epochs $EPOCHS > results/logs/worker_gpu3.log 2>&1 &
GPU3_PID=$!
echo "Worker 3 started on GPU 3 (PID: $GPU3_PID)"

echo "Waiting for all workers to finish. You can monitor logs at results/logs/"
wait $GPU0_PID $GPU2_PID $GPU3_PID

echo "=================================================="
echo "All parallel workers finished!"
echo "Summary of best trials:"
/home/duomeitinrfx/.conda/envs/plate/bin/python -c "
import optuna
try:
    study = optuna.load_study(study_name='vit_s22_tuning', storage='sqlite:///results/optuna_study.db')
    print('Best Trial:', study.best_trial.number)
    print('Best Val MAE:', study.best_trial.value)
    print('Best Params:', study.best_trial.params)
except Exception as e:
    print('Failed to read study:', e)
"
echo "Check results/sweep_results.md for a detailed report."
