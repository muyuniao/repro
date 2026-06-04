import os
import sys
import argparse
import subprocess
import json
from datetime import datetime
import optuna

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trials', type=int, default=8, help="Number of trials to run in this process")
    parser.add_argument('--epochs', type=int, default=50, help="Number of epochs to train for each trial")
    return parser.parse_args()

def generate_report():
    try:
        study = optuna.load_study(
            study_name="vit_s22_tuning",
            storage="sqlite:///results/optuna_study.db"
        )
    except Exception as e:
        print("Could not load study to generate report:", e)
        return
        
    trials = study.trials
    completed_trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        print("No completed trials to generate report.")
        return
        
    # Sort by value (MAE) ascending
    completed_trials.sort(key=lambda t: t.value if t.value is not None else float('inf'))
    
    lines = []
    lines.append("# 超参数搜索结果汇总报告\n")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"总试验次数: {len(trials)} (已完成: {len(completed_trials)})\n")
    
    lines.append("## 最佳超参配置\n")
    best_trial = study.best_trial
    lines.append(f"- **最佳试验编号**: Trial {best_trial.number}")
    lines.append(f"- **验证集 MAE**: {best_trial.value:.4f}")
    lines.append(f"- **参数配置**:")
    for k, v in best_trial.params.items():
        lines.append(f"  - `{k}`: `{v}`")
        
    lines.append("\n## 所有已完成试验排行\n")
    lines.append("| 排名 | 试验编号 | 验证集 MAE | lr | lambda_rank | lr_encoder_ratio | lr_ranker_ratio | 测试集 ACC | 测试集 MAE | 测试集 QWK |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    
    for idx, t in enumerate(completed_trials):
        metrics_path = f"results/metrics/sweep_trial_{t.number}.json"
        test_acc = "N/A"
        test_mae = "N/A"
        test_qwk = "N/A"
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, 'r') as f:
                    m = json.load(f)
                    test_acc = f"{m.get('test_acc', 0.0):.4f}"
                    test_mae = f"{m.get('test_mae', 0.0):.4f}"
                    test_qwk = f"{m.get('test_qwk', 0.0):.4f}"
            except Exception:
                pass
                
        params = t.params
        lines.append(f"| {idx+1} | Trial {t.number} | {t.value:.4f} | {params.get('lr', 0.0):.2e} | {params.get('lambda_rank', 0.0):.4f} | {params.get('lr_encoder_ratio', 0.0):.4f} | {params.get('lr_ranker_ratio', 0.0):.4f} | {test_acc} | {test_mae} | {test_qwk} |")
        
    report_path = "results/sweep_results.md"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Report successfully updated at {report_path}")

def main():
    args = parse_args()
    
    # Initialize the Optuna SQLite DB study
    os.makedirs("results", exist_ok=True)
    study = optuna.create_study(
        study_name="vit_s22_tuning",
        storage="sqlite:///results/optuna_study.db",
        direction="minimize",
        load_if_exists=True
    )
    
    def objective(trial):
        # Suggest hyperparameters
        lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
        lambda_rank = trial.suggest_float('lambda_rank', 0.1, 1.0)
        lr_encoder_ratio = trial.suggest_float('lr_encoder_ratio', 0.001, 0.05, log=True)
        lr_ranker_ratio = trial.suggest_float('lr_ranker_ratio', 0.01, 0.2, log=True)
        
        log_path = f"results/logs/trial_{trial.number}.log"
        os.makedirs("results/logs", exist_ok=True)
        
        cmd = [
            sys.executable, "train_stage2.py",
            "--encoder", "vit_base_patch16_224.augreg_in21k_ft_in1k",
            "--stage1_ckpt", "results/checkpoints/stage1_vit_base_patch16_224.augreg_in21k_ft_in1k.pt",
            "--mode", "s2.2",
            "--loss", "ce",
            "--exp_name", f"sweep_trial_{trial.number}",
            "--epochs", str(args.epochs),
            "--batch_size", "32",
            "--lr", f"{lr:.6f}",
            "--lambda_rank", f"{lambda_rank:.4f}",
            "--lr_encoder_ratio", f"{lr_encoder_ratio:.6f}",
            "--lr_ranker_ratio", f"{lr_ranker_ratio:.6f}"
        ]
        
        print(f"Starting Trial {trial.number} on GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Unknown')}")
        print(f"Parameters: lr={lr:.2e}, lambda_rank={lambda_rank:.3f}, lr_encoder_ratio={lr_encoder_ratio:.4f}, lr_ranker_ratio={lr_ranker_ratio:.4f}")
        
        with open(log_path, 'w') as log_f:
            subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT)
            
        metrics_path = f"results/metrics/sweep_trial_{trial.number}.json"
        if not os.path.exists(metrics_path):
            print(f"Error: Metrics file {metrics_path} not found for Trial {trial.number}")
            return float('inf')
            
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            
        val_best_mae = metrics.get('val_best_mae', float('inf'))
        print(f"Trial {trial.number} completed. Val MAE: {val_best_mae:.4f}")
        return val_best_mae

    # Run the optimization
    study.optimize(objective, n_trials=args.n_trials)
    
    # Generate the report
    generate_report()

if __name__ == '__main__':
    main()
