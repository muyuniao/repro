# Idea2: 两阶段序数分类 — 排序预训练 + 分类微调

## 环境与资源概况

| 项目 | 详情 |
|:---|:---|
| **数据集** | HCI (Historical Color Images), 5类(1930s-1970s), 每类~265张, 共1326张, JPEG/RGB |
| **数据路径** | `/home/duomeitinrfx/data/HistoricalColor-ECCV2012/data/imgs/decade_database/` |
| **Encoder** | ResNet-50 (`resnet50.a1_in1k`) + ViT-B/16 (`vit_base_patch16_224.augreg_in21k_ft_in1k`) |
| **Conda 环境** | `plate` 
| **GPU** | 自行使用nvidia-smi命令查看,不使用1号GPU |
| **模型权重路径** | `/home/duomeitinrfx/users/yunhe/models/` |
| **项目代码路径** | `/home/duomeitinrfx/users/yunhe/reproduce/idea2/` |

---

## 一、数据准备

### 1.1 数据集结构

```
decade_database/
├── 1930s/  (265 imgs) → label 0
├── 1940s/  (266 imgs) → label 1
├── 1950s/  (265 imgs) → label 2
├── 1960s/  (265 imgs) → label 3
└── 1970s/  (265 imgs) → label 4
```

- 类别数 K = 5，标签为 {0, 1, 2, 3, 4}，具有天然序数关系
- 数据基本均衡，无需特殊的类别平衡策略

### 1.2 数据划分

- **训练集 : 验证集 : 测试集 = 7 : 1.5 : 1.5**（按类别分层采样 stratified split）
- 使用固定随机种子(seed=42)，确保可复现
- 需要保存划分文件 `splits.json`

### 1.3 数据增强

| 阶段 | 增强策略 |
|:---|:---|
| **训练** | RandomResizedCrop(224), RandomHorizontalFlip, ColorJitter(0.2,0.2,0.2,0.1), Normalize(ImageNet) |
| **验证/测试** | Resize(256) → CenterCrop(224) → Normalize(ImageNet) |

---

## 二、整体架构

```
                        Stage 1: 排序预训练
                        ┌─────────────────────┐
  Bag(5张图,各类1张)──→ │ Encoder → scores → Sinkhorn Sorting │──→ L_rank
                        └─────────────────────┘

                        Stage 2: 分类微调 (三种方案)
                        ┌─────────────────────────────────────┐
  单张图 ──→            │ Encoder(加载Stage1) → 分类头          │──→ L_cls (± λ·L_rank)
                        └─────────────────────────────────────┘
```

---

## 三、Stage 1：排序预训练 详细设计

### 3.1 Bag 构造

每个 iteration：从 5 个类别中各随机采样 1 张图 → 组成 1 个 bag，大小固定为 5。
- 一个 batch = 多个 bag（batch_size 个 bag）
- 真实排列：按标签升序 π* = (0,1,2,3,4)

### 3.2 排序网络

**方案 A: Score-and-Sort（主方案）**

```python
# 每个样本独立打分
z_i = encoder(x_i)           # z_i ∈ R^d (d=2048 for ResNet50, 768 for ViT-B)
s_i = MLP(z_i)               # s_i ∈ R^1, MLP: d→256→1
# 构造 score matrix, 通过 Sinkhorn 生成软排列矩阵
P_hat = sinkhorn(score_matrix, tau, n_iters=20)  # P_hat ∈ R^{K×K}
```

**方案 B: Cross-Attention Ranker（对比实验）**

```python
Z = stack([z_1,...,z_K])       # Z ∈ R^{K×d}
Z' = TransformerEncoder(Z, nhead=8, nlayers=2)  # 实例间交互
s_i = Linear(Z'[i])            # 打分
P_hat = sinkhorn(...)
```

### 3.3 排序损失函数

**主损失: Soft Spearman Footrule Distance (SFD)**

L_SFD = (1/K) * Σ_i | Σ_j(j · P_hat_ij) - π*(i) |

- P_hat_ij 为 Sinkhorn 软排列矩阵的元素
- Σ_j(j · P_hat_ij) 为样本 i 的"软位置"
- 天然满足"大幅错排惩罚更大"的需求
- 例：完全反转 {5,4,3,2,1} → SFD=2.4, 相邻互换 {1,2,3,5,4} → SFD=0.4

**辅助损失（消融用）:**

| 损失 | 特点 |
|:---|:---|
| Soft Kendall Tau | 统计逆序对数量，sigmoid 松弛 |
| Weighted Permutation Matrix (WPM) | 位置差加权的排列矩阵 MSE |

### 3.4 Stage 1 超参数

| 参数 | 值 |
|:---|:---|
| Optimizer | AdamW, weight_decay=1e-4 |
| LR (Encoder) | 1e-4 |
| LR (排序网络) | 5e-4 |
| Scheduler | CosineAnnealingLR |
| Batch size | 32 bags (= 160 张图) |
| Epochs | 80 |
| Sinkhorn τ | 0.5→0.05 (线性退火) |
| Sinkhorn iters | 20 |
| Encoder 初始化 | ImageNet pretrained (timm) |

---

## 四、Stage 2：分类微调 详细设计

### 4.1 三种方案

| 方案 | Encoder | 分类头 | 排序网络 | 总 Loss |
|:---|:---|:---|:---|:---|
| **S2.1** 仅分类 | 解冻, LR=1e-5 | 新建, LR=1e-3 | 丢弃 | L_cls |
| **S2.2** 联合训练 | 解冻, LR=1e-5 | 新建, LR=1e-3 | 加载S1, LR=5e-5 | L_cls + λ·L_rank |
| **S2.3** 冻结Encoder | 冻结, LR=0 | 新建, LR=1e-3 | 丢弃 | L_cls |

### 4.2 分类头

```python
ClassifierHead: Linear(d, 256) → ReLU → Dropout(0.3) → Linear(256, K=5)
```

### 4.3 分类损失

主实验对比两种分类损失：
- **Cross-Entropy (CE)**：标准分类
- **CORAL Loss**：累积 logit 二元分类，序数感知

### 4.4 方案 S2.2 联合训练细节

- λ 初始值 0.5，线性衰减至 0（在总 epoch 的 70% 处归零）
- 排序 bag 从当前 mini-batch 中按标签构造
- 消融 λ ∈ {0.1, 0.3, 0.5, 0.7, 1.0}

### 4.5 Stage 2 超参数

| 参数 | 值 |
|:---|:---|
| Optimizer | AdamW, weight_decay=1e-4 |
| Scheduler | CosineAnnealingLR |
| Batch size | 64 |
| Epochs | 50 (S2.1/S2.2), 80 (S2.3) |
| Early stopping | patience=15, monitor=val_MAE |

---

## 五、完整实验矩阵

### 5.1 主实验（必跑）

| 实验ID | Encoder | Stage 1 | Stage 2 方案 | 分类 Loss | 说明 |
|:---|:---|:---|:---|:---|:---|
| E1 | ResNet-50 | ✓ | S2.1 仅分类 | CE | 排序预训练+分类 |
| E2 | ResNet-50 | ✓ | S2.2 联合 | CE | 排序预训练+联合训练 |
| E3 | ResNet-50 | ✓ | S2.3 冻结 | CE | 冻结encoder |
| E4 | ResNet-50 | ✗ | 直接分类 | CE | **Baseline: 无预训练** |
| E5 | ResNet-50 | ✗ | 直接分类 | CORAL | **Baseline: CORAL** |
| E6 | ResNet-50 | ✓ | S2.1 | CORAL | 排序预训练+CORAL |
| E7 | ResNet-50 | ✓ | S2.2 | CORAL | 排序预训练+CORAL+联合 |
| E8 | ViT-B/16 | ✓ | S2.1 | CE | ViT验证 |
| E9 | ViT-B/16 | ✓ | S2.2 | CE | ViT联合验证 |
| E10 | ViT-B/16 | ✓ | S2.3 | CE | ViT冻结验证 |
| E11 | ViT-B/16 | ✗ | 直接分类 | CE | **ViT Baseline** |
| E12 | ViT-B/16 | ✗ | 直接分类 | CORAL | **ViT CORAL Baseline** |

### 5.2 消融实验（在 ResNet-50 + S2.2 + CE 设置下）

| 编号 | 变量 | 取值范围 |
|:---|:---|:---|
| A1 | 排序 Loss 类型 | SFD vs Kendall Tau vs WPM |
| A2 | 联合训练 λ | 0.1, 0.3, 0.5, 0.7, 1.0 |
| A3 | λ 调度策略 | 固定 vs 线性衰减 vs 余弦衰减 |
| A4 | Bag 构造 | Full-Bag vs Sub-Bag(随机3类) vs Hard-Negative(相邻3类) |
| A5 | Stage 1 epoch 数 | 20, 40, 60, 80, 100 |
| A6 | 排序网络 | MLP vs Transformer(方案B) |

---

## 六、评估指标

所有实验统一计算以下指标：

| 指标 | 说明 | 序数敏感 |
|:---|:---|:---|
| **Accuracy (ACC)** | 分类准确率 | ❌ |
| **MAE** | 平均绝对误差 | ✅ |
| **MSE** | 均方误差 | ✅ |
| **RMSE** | 均方根误差 | ✅ |
| **Macro F1** | 宏平均 F1 | ❌ |
| **QWK** | Quadratic Weighted Kappa | ✅ |
| **Spearman ρ** | 排序相关系数 | ✅ |
| **CS@1** | Cumulative Score (θ=1) | ✅ |
| **Per-class ACC** | 每类准确率 | ❌ |

---

## 七、代码文件结构

```
idea2/
├── idea2.md                # 本文件
├── configs/
│   └── default.yaml        # 超参数配置
├── data/
│   ├── dataset.py          # HCI数据集加载 + Bag采样器
│   └── splits.json         # 数据划分
├── models/
│   ├── encoder.py          # ResNet/ViT encoder (timm封装)
│   ├── ranker.py           # 排序网络 (Score-and-Sort + Sinkhorn)
│   ├── classifier.py       # 分类头
│   └── losses.py           # SFD / Kendall / WPM / CORAL loss
├── train_stage1.py         # Stage 1 排序预训练
├── train_stage2.py         # Stage 2 分类微调 (支持S2.1/S2.2/S2.3)
├── evaluate.py             # 统一评估 (输出所有指标)
├── utils.py                # 工具函数
├── run_all.sh              # 一键运行所有实验
└── results/                # 实验结果输出目录
    ├── checkpoints/
    ├── logs/
    └── metrics/
```

---

## 八、执行顺序

### Phase 1: 基础搭建与概念验证 (Day 1-2)
1. 实现数据加载、Bag 采样器、数据划分
2. 实现 Encoder 封装（ResNet-50）
3. 实现 Sinkhorn 排序 + SFD Loss
4. 跑通 Stage 1 → Stage 2 (S2.1) 的最小 pipeline
5. 跑 E4 (Baseline) 确认基线性能

### Phase 2: 主实验 (Day 3-5)
1. 跑 E1-E7 (ResNet-50 全部实验)
2. 实现 ViT Encoder，跑 E8-E12
3. 对比分析结果

### Phase 3: 消融实验 (Day 6-8)
1. 按优先级跑 A1-A6
2. 绘制消融曲线

### Phase 4: 分析与可视化 (Day 9-10)
1. t-SNE/UMAP 可视化特征空间（Stage1前/后、Stage2后）
2. 混淆矩阵
3. 整理所有实验结果表格

---

## 九、运行命令示例

```bash
# 激活环境
conda activate wugang

# Stage 1: 排序预训练 (ResNet-50)
CUDA_VISIBLE_DEVICES=0 python train_stage1.py \
    --encoder resnet50 --epochs 80 --batch_size 32 \
    --lr_encoder 1e-4 --lr_ranker 5e-4 \
    --rank_loss sfd --sinkhorn_tau 0.5

# Stage 2: 分类微调 - 方案 S2.1
CUDA_VISIBLE_DEVICES=0 python train_stage2.py \
    --encoder resnet50 --scheme s2.1 \
    --stage1_ckpt results/checkpoints/stage1_resnet50_best.pt \
    --cls_loss ce --epochs 50 --batch_size 64

# Stage 2: 分类微调 - 方案 S2.2 (联合训练)
CUDA_VISIBLE_DEVICES=0 python train_stage2.py \
    --encoder resnet50 --scheme s2.2 \
    --stage1_ckpt results/checkpoints/stage1_resnet50_best.pt \
    --cls_loss ce --lambda_rank 0.5 --lambda_decay linear --epochs 50

# Stage 2: 方案 S2.3 (冻结 Encoder)
CUDA_VISIBLE_DEVICES=0 python train_stage2.py \
    --encoder resnet50 --scheme s2.3 \
    --stage1_ckpt results/checkpoints/stage1_resnet50_best.pt \
    --cls_loss ce --epochs 80

# Baseline: 无预训练直接分类
CUDA_VISIBLE_DEVICES=0 python train_stage2.py \
    --encoder resnet50 --scheme baseline \
    --cls_loss ce --epochs 50

# 评估
python evaluate.py --checkpoint results/checkpoints/xxx.pt --split test
```

---

## 十、风险与应对

| 风险 | 应对 |
|:---|:---|
| 数据量小(1326张)导致过拟合 | 使用ImageNet预训练+强数据增强+EarlyStopping |
| Sinkhorn 梯度不稳定 | 监控梯度范数; 温度退火不宜过快 |
| Stage 2 灾难性遗忘 | 对比 S2.1 vs S2.2 量化遗忘程度 |
| 排序预训练无增益 | S2.3(冻结)实验可诊断预训练质量 |
| K=5 排序信号太弱 | 尝试 Multi-Instance Bag (每类多张) |

---

## 附录：之前的 Implementation Plan

详见: [implementation_plan.md](/home/duomeitinrfx/.gemini/antigravity/brain/9d34041b-422a-47b1-9358-e0f881f8414b/implementation_plan.md)
