# 两阶段序数分类实验计划：排序预训练 + 分类微调

## 1. 整体思路分析与建议

### 1.1 你的 Idea 核心价值

你的想法本质上是：**用排序任务（Learning to Rank）作为一种自监督/弱监督的预训练手段，迫使 Encoder 学习到包含序数结构（ordinal structure）的特征表征，然后将这种序数感知的表征迁移到下游的序数分类任务中。**

这个思路的合理性在于：
- 排序是比分类更"细粒度"的监督信号——分类只要求把样本分到正确的桶里，而排序要求模型理解样本之间的**相对顺序关系**
- 传统的分类损失（CE Loss）对所有错误一视同仁，但序数分类中，把 Grade 1 误判为 Grade 5 远比误判为 Grade 2 严重——排序预训练天然编码了这种距离信息

### 1.2 关键设计决策的建议

#### 关于 Stage 2 的两个方案：建议使用 **方案 2.2（联合训练）**，但需要更精细的设计

| 维度 | 方案 2.1（仅分类） | 方案 2.2（分类 + 排序联合） |
|:---|:---|:---|
| 优点 | 简单、干净，Stage 2 专注于分类目标 | 持续的排序信号防止 Encoder 在分类微调中遗忘序数结构 |
| 缺点 | Encoder 在 Stage 2 微调过程中可能发生**灾难性遗忘**，丢失序数表征 | 需要仔细调节排序 loss 的权重，否则两个目标可能冲突 |
| 风险 | 如果分类数据充足，遗忘问题可能不严重；但如果数据有限，Stage 1 学到的结构可能被覆盖 | 如果排序 loss 权重过大，可能阻碍分类收敛 |

**最终建议**：以方案 2.2 为主方案，方案 2.1 作为消融实验（ablation）。同时，我建议增加一个**方案 2.3**：
> **方案 2.3: 冻结 Encoder + 仅训练分类头**：Stage 2 中完全冻结 Encoder，只训练分类网络。用于验证 Stage 1 的排序预训练是否真正学到了可直接迁移的序数表征。

这三个方案的对比可以回答一个核心科学问题：**排序预训练学到的序数信息，在多大程度上能够保留并迁移到分类任务？**

---

## 2. 详细实验设计

### 2.1 Stage 1：排序预训练

#### 2.1.1 Bag 构造策略

**基础策略（你描述的）**：从 $K$ 个类别中各随机采样 1 张图像，构成一个大小为 $K$ 的 bag。

**建议的改进与消融**：

| 策略 | 描述 | 目的 |
|:---|:---|:---|
| **Full-Bag** | 每个类各采样 1 张，bag 大小 = $K$ | 基础设置，覆盖所有类别 |
| **Sub-Bag** | 随机选取 $m < K$ 个类别各采样 1 张 | 增加多样性，让模型学习局部排序 |
| **Multi-Instance Bag** | 每个类采样 $n > 1$ 张，bag 大小 = $K \times n$ | 增加类内变异性，更强的排序信号 |
| **Hard-Negative Bag** | 优先采样相邻类别的样本（如只采 Grade 2,3,4） | 强化对细粒度差异的辨别 |

> [!TIP]
> **建议的主实验策略**：以 Full-Bag 为默认设置。Sub-Bag 和 Hard-Negative Bag 作为消融实验，验证对训练的影响。

#### 2.1.2 Encoder 架构

| 选项 | 适用场景 | 推荐度 |
|:---|:---|:---|
| ResNet-50 / ResNet-101 | 通用基线，参数量适中 | ⭐⭐⭐ |
| ConvNeXt-Base | 更现代的 CNN，性能通常优于 ResNet | ⭐⭐⭐⭐ |
| ViT-B/16 (ImageNet pretrained) | Transformer 架构，适合大规模数据 | ⭐⭐⭐⭐ |
| Swin-Transformer-B | 层次化 ViT，适合多尺度特征 | ⭐⭐⭐⭐ |

**建议**：主实验使用 **ResNet-50**（快速迭代）和 **ConvNeXt-Base**（追求 SOTA）。消融实验中加入 ViT-B/16 验证架构无关性。

#### 2.1.3 排序网络设计

排序网络的输入是 bag 中所有实例的 Encoder 输出 $\{\mathbf{z}_1, \mathbf{z}_2, \ldots, \mathbf{z}_K\}$，输出是一个预测的排列（permutation）。

**方案 A：基于评分的排序（Score-and-Sort）**

```
Encoder输出 z_i ∈ R^d → MLP → 标量 score s_i ∈ R
对 {s_1, ..., s_K} 排序得到预测排列
```

- 每个实例独立地映射到一个标量分数
- 排序操作使用**可微排序网络**（Differentiable Sorting Network）实现端到端训练
- 可选：Sinkhorn 算子生成软排列矩阵（Soft Permutation Matrix）

**方案 B：基于交互的排序（Cross-Attention Ranker）**

```
Encoder输出 {z_1, ..., z_K} → Transformer Encoder (self-attention) → 排序分数 {s_1, ..., s_K}
```

- 让 bag 中的实例通过 self-attention 互相交互，利用**相对比较信息**
- 比方案 A 更有表达力，但计算开销更大

> [!IMPORTANT]
> **建议**：主实验使用**方案 A（Score-and-Sort + Sinkhorn 软排列）**，因为简洁且有理论支撑。方案 B 作为增强实验。

#### 2.1.4 排序损失函数设计（核心环节）

你的需求：**排序结果的错误程度不同，惩罚也应不同**。这正是排序距离度量（Ranking Distance Metrics）的研究领域。

**建议使用以下损失函数的组合/对比**：

---

**Loss 1: 可微 Spearman Footrule Distance（位移距离）**

$$\mathcal{L}_{SFD} = \frac{1}{K} \sum_{i=1}^{K} |\hat{\pi}(i) - \pi^*(i)|$$

其中 $\hat{\pi}$ 是预测排列，$\pi^*$ 是真实排列。
- 直觉：每个元素的"位移量"之和。完全反转 {5,4,3,2,1} 的 SFD = $(4+2+0+2+4)/5 = 2.4$，而相邻互换 {1,2,3,5,4} 的 SFD = $(0+0+0+1+1)/5 = 0.4$。
- 自然满足你的需求：大幅度错排比小幅度错排的惩罚更大。
- **可微化**：使用 Sinkhorn 软排列矩阵 $\hat{P}$ 代替硬排列，$\hat{\pi}(i) \approx \sum_j j \cdot \hat{P}_{ij}$。

---

**Loss 2: 可微 Kendall's Tau 距离（逆序对距离）**

$$\mathcal{L}_{KT} = \sum_{i<j} \mathbb{1}[\hat{\pi}(i) > \hat{\pi}(j) \text{ when } \pi^*(i) < \pi^*(j)]$$

- 直觉：统计"逆序对"的数量。完全反转有 $\binom{5}{2}=10$ 个逆序对，相邻互换只有 1 个。
- **可微化**：使用 sigmoid 松弛 $\mathbb{1}[\cdot]$：
$$\mathcal{L}_{KT}^{soft} = \sum_{i<j} \sigma\left(\frac{s_i - s_j}{\tau}\right) \cdot \mathbb{1}[\pi^*(i) < \pi^*(j)]$$

---

**Loss 3: 加权排列矩阵损失（Weighted Permutation Matrix Loss）**

使用 Sinkhorn 算子生成预测的软排列矩阵 $\hat{P} \in \mathbb{R}^{K \times K}$，与真实排列矩阵 $P^*$ 比较：

$$\mathcal{L}_{WPM} = \sum_{i,j} W_{ij} \cdot (P^*_{ij} - \hat{P}_{ij})^2$$

其中权重矩阵 $W_{ij} = |i - j|^{\beta}$，即**位置差距越大的元素，错误的惩罚越大**。

- $\beta = 1$: 线性惩罚
- $\beta = 2$: 二次惩罚（对大幅度错排更严厉）

---

> [!TIP]
> **建议的主损失**：**Loss 1 (Soft SFD)** 或 **Loss 3 (WPM, $\beta=1$)** 作为主损失，Loss 2 (Soft Kendall Tau) 作为辅助正则项。消融实验中对比三种 Loss 的效果。

#### 2.1.5 Stage 1 训练细节

| 超参数 | 建议值 | 备注 |
|:---|:---|:---|
| Optimizer | AdamW | 标准选择 |
| Learning Rate | 1e-4 (Encoder), 5e-4 (排序网络) | Encoder 学习率较低以保留预训练特征 |
| LR Scheduler | Cosine Annealing | - |
| Batch Size | 32-64 bags | 每个 bag 包含 K 张图 |
| Epochs | 50-100 | 根据数据集大小调整 |
| Sinkhorn 温度 $\tau$ | 0.1 → 0.01 (退火) | 逐渐逼近硬排列 |
| Sinkhorn 迭代次数 | 20 | - |
| Encoder 预训练权重 | ImageNet pretrained | 不要从头训练 |

---

### 2.2 Stage 2：分类微调

#### 2.2.1 三个方案的并行实验

| 方案 | Encoder | 分类网络 | 排序网络 | Loss |
|:---|:---|:---|:---|:---|
| **2.1: 仅分类** | 解冻，微调 | 新初始化，训练 | 丢弃 | $\mathcal{L}_{cls}$ |
| **2.2: 联合训练** | 解冻，微调 | 新初始化，训练 | 加载 Stage 1 权重，训练 | $\mathcal{L}_{cls} + \lambda \mathcal{L}_{rank}$ |
| **2.3: 冻结 Encoder** | 冻结 | 新初始化，训练 | 丢弃 | $\mathcal{L}_{cls}$ |

#### 2.2.2 分类网络设计

```
Encoder 输出 z ∈ R^d
    → FC(d, 256) → ReLU → Dropout(0.3)
    → FC(256, K)  // K 个类别
```

#### 2.2.3 分类损失函数选择（消融实验）

| Loss | 公式/描述 | 是否序数感知 |
|:---|:---|:---|
| **Cross-Entropy (CE)** | 标准多分类损失 | ❌ |
| **CORAL Loss** | 累积 logit + 二元 CE | ✅ |
| **Ordinal CE (Weighted)** | $\mathcal{L} = -\sum_k w_{yk} \log p_k$，其中 $w_{yk} = |y - k|^{\alpha}$ | ✅ |
| **Earth Mover's Distance (EMD)** | Wasserstein-1 距离 on 概率分布 | ✅ |

> [!IMPORTANT]
> **建议**：主实验使用 **CE Loss** 和 **CORAL Loss** 各跑一遍。CE 用于验证排序预训练本身就能提供足够的序数信息，CORAL 用于看两者是否互补。

#### 2.2.4 方案 2.2 的联合训练细节

**Bag 构造**：Stage 2 中排序任务的 bag 可以从当前 mini-batch 中构造（不需要额外采样）。

**Loss 权重调度**：建议排序 loss 权重 $\lambda$ 使用**线性衰减**：

$$\lambda(t) = \lambda_0 \cdot \max\left(0,\ 1 - \frac{t}{T_{decay}}\right)$$

- 初始 $\lambda_0 = 0.5$
- 在总训练轮次的 70% 处衰减到 0
- 直觉：训练初期排序 loss 帮助 Encoder 保持序数结构，后期让分类 loss 主导微调

**消融的 $\lambda$ 值**：$\{0.1, 0.3, 0.5, 0.7, 1.0\}$，以及固定 vs 衰减策略。

#### 2.2.5 Stage 2 训练细节

| 超参数 | 方案 2.1 / 2.2 | 方案 2.3 |
|:---|:---|:---|
| Encoder LR | 1e-5 (低学习率微调) | 0 (冻结) |
| 分类头 LR | 1e-3 | 1e-3 |
| 排序网络 LR (方案2.2) | 5e-5 | - |
| Epochs | 30-50 | 50-80 |
| 其他 | 同 Stage 1 | 同 Stage 1 |

---

## 3. 数据集选择

建议在以下数据集上验证（由易到难）：

| 数据集 | 任务 | 类别数 $K$ | 数据量 | 特点 |
|:---|:---|:---|:---|:---|
| **Adience** | 年龄分组 | 8 | ~26k | 天然序数，类间差异大 |
| **DR (Diabetic Retinopathy)** | 视网膜病变分级 | 5 | ~88k (EyePACS) | 医学场景，类不平衡严重 |
| **Historical Color Images** | 图片年代分类 | 5 | ~14k | 细粒度序数 |
| **FER (Facial Expression)** | 情感强度分级 | 自定义 | 多种 | 可选，需构造序数标签 |

> [!TIP]
> **建议至少使用 2 个数据集**：一个"简单"的（如 Adience）用于快速迭代和调参，一个"困难"的（如 DR）用于验证方法在实际场景的有效性。

---

## 4. Baseline 对比

| Baseline | 描述 | 目的 |
|:---|:---|:---|
| **Vanilla CE** | Encoder + FC，标准 Cross-Entropy | 最基础的分类 baseline |
| **CORAL** | Encoder + CORAL Loss | 传统序数分类 SOTA |
| **SORD** | Soft label + KL divergence | 序数软标签方法 |
| **RankSim** | 排序相似性正则化 | 排序表征学习的代表方法 |
| **Rank-N-Contrast** | 基于排序的对比学习 | 对比学习 + 序数的 SOTA |
| **无 Stage 1 预训练** | 直接用 ImageNet 预训练权重 → Stage 2 | 验证排序预训练的增益 |

---

## 5. 评估指标

| 指标 | 公式/描述 | 为什么需要 |
|:---|:---|:---|
| **Accuracy (ACC)** | 分类准确率 | 基本指标 |
| **MAE** | $\frac{1}{N}\sum|y_i - \hat{y}_i|$ | 序数分类核心指标，衡量平均偏移 |
| **RMSE** | $\sqrt{\frac{1}{N}\sum(y_i - \hat{y}_i)^2}$ | 对大偏移更敏感 |
| **Cumulative Score (CS)** | $\frac{1}{N}\sum\mathbb{1}[|y_i - \hat{y}_i| \leq \theta]$ | 衡量"容忍度内"的准确率 |
| **Quadratic Weighted Kappa (QWK)** | Cohen's Kappa 的加权版本 | 衡量一致性，考虑偏移距离 |
| **Spearman's $\rho$** | 排序相关系数 | 衡量预测排序与真实排序的单调性 |
| **Confusion Matrix** | 可视化误分类的分布 | 直观展示是否存在系统性偏移 |

---

## 6. 消融实验列表

按优先级排列：

| 编号 | 消融内容 | 变量 | 回答的问题 |
|:---|:---|:---|:---|
| **A1** | Stage 2 方案对比 | 2.1 vs 2.2 vs 2.3 | 排序预训练的知识如何迁移最有效？ |
| **A2** | 排序 Loss 对比 | SFD vs Kendall vs WPM | 哪种排序损失最适合学习序数表征？ |
| **A3** | 分类 Loss 对比 | CE vs CORAL vs EMD | 排序预训练与哪种分类 loss 最互补？ |
| **A4** | 联合训练的 $\lambda$ | 0.1, 0.3, 0.5, 0.7, 1.0 | 排序 loss 的最优权重是多少？ |
| **A5** | $\lambda$ 调度策略 | 固定 vs 线性衰减 vs 余弦衰减 | 排序 loss 权重应如何变化？ |
| **A6** | Bag 构造策略 | Full vs Sub vs Hard-Negative | 包的采样策略对表征质量的影响？ |
| **A7** | Stage 1 训练轮次 | 10, 25, 50, 100 epochs | 排序预训练需要多充分？ |
| **A8** | 排序网络架构 | MLP vs Transformer | 实例间交互是否重要？ |
| **A9** | Encoder 架构 | ResNet-50 vs ConvNeXt vs ViT | 方法是否架构无关？ |

---

## 7. 实验执行顺序（推荐）

### Phase 1: 概念验证（1-2 周）
- 选定一个小数据集（如 Adience）
- 使用 ResNet-50 + MLP 排序网络 + Soft SFD Loss
- 跑通 Stage 1 → Stage 2（三个方案）的完整 pipeline
- **目标**：确认排序预训练确实提升了序数分类性能

### Phase 2: 核心消融（2-3 周）
- 在小数据集上完成消融 A1-A5
- 确定最优的 Loss 组合、$\lambda$ 值和训练策略
- **目标**：确定最优配置

### Phase 3: 主实验（1-2 周）
- 使用最优配置在所有数据集上跑完整实验
- 与所有 Baseline 对比
- **目标**：验证方法的普适性

### Phase 4: 深度分析（1 周）
- t-SNE/UMAP 可视化 Encoder 特征空间（Stage 1 前 vs Stage 1 后 vs Stage 2 后）
- 消融 A6-A9
- 错误分析：模型的典型失败模式
- **目标**：理解方法为何有效

---

## 8. 可视化与分析计划

| 分析 | 方法 | 目的 |
|:---|:---|:---|
| **特征空间可视化** | t-SNE/UMAP of Encoder 输出 | 验证排序预训练是否让不同等级在特征空间中有序排列 |
| **排序准确率曲线** | Stage 1 训练过程中的 Kendall's $\tau$ | 监控排序学习的收敛 |
| **灾难性遗忘分析** | Stage 2 训练过程中的排序性能变化 | 对比方案 2.1（可能遗忘）vs 2.2（持续排序约束） |
| **混淆矩阵** | 所有方案的 Confusion Matrix | 观察误分类是否集中在相邻类别 |
| **$\lambda$ 敏感性曲线** | MAE vs $\lambda$ 的折线图 | 展示排序 loss 权重的影响 |

---

## 9. 潜在风险与应对

| 风险 | 可能原因 | 应对策略 |
|:---|:---|:---|
| Stage 1 排序不收敛 | Sinkhorn 温度过高/过低、学习率过大 | 调整温度退火策略；检查梯度范数 |
| 排序预训练无明显增益 | 排序任务太简单/太难 | 调整 Bag 大小；尝试 Hard-Negative Bag |
| 方案 2.2 中两个 loss 冲突 | $\lambda$ 不合适 | 使用梯度冲突检测（PCGrad 等）；动态调整 $\lambda$ |
| 类别不平衡影响 Bag 采样 | DR 等数据集的不平衡比高达 1:73 | Bag 构造时使用类别平衡采样（已经是每类采 1 张，天然平衡） |

---

## User Review Required

> [!IMPORTANT]
> **以下几个问题需要您确认后才能开始编码实现：**
> 1. **目标数据集**：您计划在哪些数据集上验证？是否已有数据？
> 2. **Encoder 选型**：您倾向于使用哪种 Encoder（ResNet / ConvNeXt / ViT）？
> 3. **排序网络的实现方式**：您倾向于 Sinkhorn 软排列矩阵方案，还是直接用可微排序网络（Differentiable Sorting Network）？
> 4. **计算资源**：您有多少 GPU？这会影响 batch size 和 Encoder 选型。
> 5. **方案 2.3（冻结 Encoder）**是否纳入实验？这是一个重要的消融实验，但会增加实验量。
> 6. **是否需要我现在就开始搭建代码框架？**
