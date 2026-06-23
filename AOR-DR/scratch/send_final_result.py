import requests

def send_final_results():
    send_key = "SCT359045TapV8Dwo5hF2FAzSRDDKdtkIh"
    url = f"https://sctapi.ftqq.com/{send_key}.send"
    
    title = "🎉 AOR-DR 五折交叉验证实验圆满结束！"
    desp = (
        "### 📊 Adience 数据集学术级（无泄露）五折交叉验证最终指标汇报：\n\n"
        "| 统计指标 | 评估数值 (5-Fold Mean) |\n"
        "| :--- | :--- |\n"
        "| **平均准确率 (Mean Accuracy)** | **55.93% ± 5.96%** |\n"
        "| **平均绝对误差 (Mean MAE)** | **0.5525 ± 0.0811** |\n"
        "| **平均 F1-Score (Macro)** | **49.41%** |\n"
        "| **平均敏感度 (Mean Sens)** | **51.85%** |\n"
        "| **平均特异度 (Mean Spec)** | **76.18%** |\n\n"
        "#### 📁 各折详细表现：\n"
        "- **Fold 0**: Acc=66.96%, MAE=0.4055\n"
        "- **Fold 1**: Acc=51.75%, MAE=0.6009\n"
        "- **Fold 2**: Acc=57.26%, MAE=0.5303\n"
        "- **Fold 3**: Acc=50.59%, MAE=0.6365\n"
        "- **Fold 4**: Acc=53.10%, MAE=0.5892\n\n"
        "所有的最优模型权重 `vit-b_adience_position_fold{0..4}.pth` 已全部安全归档于工作目录中，实验已完美收官！"
    )
    
    data = {
        "title": title,
        "desp": desp
    }
    
    response = requests.post(url, data=data, proxies={"http": None, "https": None})
    print("Notification sent. Status code:", response.status_code)

if __name__ == '__main__':
    send_final_results()
