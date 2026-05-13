# OrdinalDPO: 序数感知偏好优化增强MLLMs序数分类能力

## 1. 核心 Idea（不变）

**问题**：MLLMs的SFT用交叉熵损失，将"预测差1级"和"预测差5级"等同惩罚，忽略序数结构。

**方案**：**Rank-Sensitive DPO (RS-DPO)** —— 将序数距离转化为DPO偏好信号：
- 对真实标签 $y$ 的样本，$|ŷ_1 - y| < |ŷ_2 - y|$ → $ŷ_1$ 是 preferred response
- 偏好对权重与序数距离成正比（距离越远的错误，模型应越确信地拒绝）
- 两阶段训练：SFT → RS-DPO

---

## 2. 硬件约束与训练策略

### 2.1 硬件条件
- 4× RTX 3090 (24GB)，常规可用 2 张（各 10-20GB 空闲）

### 2.2 显存分析

| 组件 | FP16 | QLoRA 4-bit |
|:---|:---|:---|
| LLaVA-1.5-7B 模型权重 | ~14GB | ~4GB |
| LoRA 参数 (rank=16) | ~0.3GB | ~0.3GB |
| DPO 参考模型 (frozen) | +14GB | +4GB |
| 优化器状态 | ~1GB | ~0.5GB |
| 梯度/激活值 | ~4-8GB | ~2-4GB |
| **总计** | **~33GB ❌** | **~11-13GB ✅** |

### 2.3 推荐训练配置

> [!IMPORTANT]
> **必须使用 QLoRA (4-bit量化)** 才能在单卡3090上完成DPO训练。

```yaml
# SFT 阶段配置
quantization: 4bit (nf4)
lora_rank: 16
lora_alpha: 32
lora_target: q_proj, v_proj, k_proj, o_proj
batch_size: 1
gradient_accumulation: 16
gradient_checkpointing: true
optimizer: paged_adamw_8bit
learning_rate: 2e-4
epochs: 3
GPU: 单卡3090 即可

# RS-DPO 阶段配置  
quantization: 4bit (nf4)
lora_rank: 16
lora_alpha: 32
batch_size: 1
gradient_accumulation: 8
beta: 0.1
learning_rate: 5e-5
epochs: 1-2
GPU: 单卡3090
```

### 2.4 多卡策略
- **训练**：单卡即可（QLoRA），无需分布式
- **推理**：可用多卡并行加速评估（不同数据集分配到不同卡）
- **并行实验**：空闲的卡可同时运行不同 ablation 实验

---

## 3. 投稿目标（CCF-B）

| 会议 | CCF | 截稿日期(预估) | 匹配度 | 说明 |
|:---|:---|:---|:---|:---|
| **ICME 2027** | B | ~Dec 2026 | ⭐⭐⭐⭐⭐ | 多媒体+AI，最佳匹配 |
| **ECAI 2026** | B | ~Apr 2026（已过） | ⭐⭐⭐⭐ | AI方法论 |
| **ICPR 2026** | C | Jan 2026（已过） | ⭐⭐⭐ | 模式识别 |
| **PRCV 2026** | CCF-C(国内认可度高) | ~Jul 2026 | ⭐⭐⭐⭐ | 国内视觉会议 |

> [!TIP]
> **推荐目标：ICME 2027**（截稿约2026年12月），时间充裕。如果需要更快出成果，可先投 **Pattern Recognition**（CCF-B期刊，随时可投）或 **PRCV 2026**。

---

## 4. 数据集

| 数据集 | 领域 | 类别数 | 规模 | 优先级 |
|:---|:---|:---|:---|:---|
| **Adience** | 年龄估计 | 8 | ~26K | 🥇 已有经验 |
| **HCI** | 历史图片年代 | 5 | 1,325 | 🥈 小数据集，快速验证 |
| **AVA** | 美学评估 | 5(分段) 或10 | ~250K | 🥈 美学评分 |
| **APTOS 2019 DR** | 糖尿病视网膜病变 | 5 | ~3.6K | 🥉 医学场景 |

> [!NOTE]
> AVA 原始是10级评分，建议按照均分方式分为5个序数等级（或使用均值的5分段）以保持序数特性。

---

## 5. Baseline 清单

### 传统方法
| 方法 | 代码 |
|:---|:---|
| CORAL + ResNet-50 | [coral-pytorch](https://github.com/Raschka-research-group/coral-pytorch) |
| SORD | [GitHub](https://github.com/JHU-DIGIT/Soft-Labels-for-Ordinal-Regression) |

### MLLM 方法
| 方法 | 说明 |
|:---|:---|
| LLaVA-1.5-7B Zero-Shot | 直接推理 |
| LLaVA-1.5-7B + SFT | 标准QLoRA微调 |
| LLaVA-1.5-7B + Standard DPO | 非序数感知DPO |
| OrderChain | CoT提示+SFT |
| **OrdinalDPO (Ours)** | RS-DPO |

---

## 6. 实施时间线

### Phase 0: 环境与数据（1周）
- 搭建环境（transformers, TRL, PEFT, bitsandbytes）
- 下载/预处理 Adience、HCI、AVA、APTOS 数据集
- 实现偏好对生成脚本

### Phase 1: Baseline 复现（2周）
- CORAL + ResNet-50
- LLaVA zero-shot / SFT baseline
- 标准 DPO baseline

### Phase 2: 方法实现（2周）
- 实现序数偏好数据构造 Pipeline
- 继承 TRL DPOTrainer，实现 RS-DPO 损失
- 两阶段训练 Pipeline
- 在 Adience 上初步验证

### Phase 3: 主实验（2周）
- 4个数据集全面对比实验
- 统计结果制表

### Phase 4: Ablation + 分析（1周）
- 权重函数（线性/指数/对数）
- 采样策略
- Stage1 only vs Stage2 only vs 两阶段
- 混淆矩阵可视化

### Phase 5: 论文撰写（2-3周）
- 撰写 + 制图 + 校对

**总计：约 10-11 周**

---

## 7. RS-DPO 核心实现

```python
class RankSensitiveDPOTrainer(DPOTrainer):
    def __init__(self, *args, weight_fn="linear", **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_fn = weight_fn
    
    def compute_ordinal_weight(self, ordinal_distance):
        if self.weight_fn == "linear":
            return ordinal_distance.float()
        elif self.weight_fn == "exp":
            return torch.exp(0.5 * ordinal_distance.float()) - 1
        elif self.weight_fn == "log":
            return torch.log(1 + ordinal_distance.float())
    
    def dpo_loss(self, policy_chosen_logps, policy_rejected_logps,
                 reference_chosen_logps, reference_rejected_logps,
                 ordinal_distances):
        logits = (policy_chosen_logps - reference_chosen_logps) - \
                 (policy_rejected_logps - reference_rejected_logps)
        weights = self.compute_ordinal_weight(ordinal_distances)
        losses = -weights * F.logsigmoid(self.beta * logits)
        return losses.mean()
```

---

## 8. 风险与应对

| 风险 | 应对 |
|:---|:---|
| QLoRA精度损失 | 对比FP16 SFT验证差距，必要时使用更高的LoRA rank |
| DPO训练不稳定 | 小β(0.1)、低学习率(5e-5)、充分warmup |
| AVA数据量大 | 采样子集(50K)训练，全集评估 |
| 与OPO区分度 | 强调视觉+序数的专属设计，OPO是通用LLM对齐 |
