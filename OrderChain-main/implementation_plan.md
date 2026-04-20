# OrderChain 论文实验复现指南

本文档详细列出了复现 [OrderChain: Towards General Instruct-Tuning for Stimulating the Ordinal Understanding Ability of MLLM](https://arxiv.org/abs/2504.04801) (ICCV-2025) 实验时**需要修改的所有地方**。

---

## 总览

代码基于 **LLaVA-1.5** 框架，主要流程为：
1. 准备数据集 → 2. 数据预处理 (OrderChain 格式) → 3. LoRA 微调训练 → 4. 合并 LoRA 权重 → 5. 评估推理

> [!IMPORTANT]
> 代码中大量路径使用 `...` 作为占位符，**必须全部替换为你本机的实际绝对路径**，这是最核心的修改。

---

## 一、缺失文件与依赖（需补充）

### 1.1 缺失的 DeepSpeed 配置文件

> [!CAUTION]
> 训练脚本引用了 `zero2.json` 和 `zero3.json`，但 `scripts/` 目录下没有这两个文件。

需要在 `scripts/` 下创建：

#### [NEW] scripts/zero2.json
```json
{
    "fp16": { "enabled": "auto", "loss_scale": 0, "loss_scale_window": 1000, "initial_scale_power": 16, "hysteresis": 2, "min_loss_scale": 1 },
    "bf16": { "enabled": "auto" },
    "optimizer": { "type": "AdamW", "params": { "lr": "auto", "betas": "auto", "eps": "auto", "weight_decay": "auto" } },
    "scheduler": { "type": "WarmupLR", "params": { "warmup_min_lr": "auto", "warmup_max_lr": "auto", "warmup_num_steps": "auto" } },
    "zero_optimization": { "stage": 2, "offload_optimizer": { "device": "cpu", "pin_memory": true }, "allgather_partitions": true, "allgather_bucket_size": 2e8, "overlap_comm": true, "reduce_scatter": true, "reduce_bucket_size": 2e8, "contiguous_gradients": true },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 1e5,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
```

#### [NEW] scripts/zero3.json
```json
{
    "fp16": { "enabled": "auto", "loss_scale": 0, "loss_scale_window": 1000, "initial_scale_power": 16, "hysteresis": 2, "min_loss_scale": 1 },
    "bf16": { "enabled": "auto" },
    "optimizer": { "type": "AdamW", "params": { "lr": "auto", "betas": "auto", "eps": "auto", "weight_decay": "auto" } },
    "scheduler": { "type": "WarmupLR", "params": { "warmup_min_lr": "auto", "warmup_max_lr": "auto", "warmup_num_steps": "auto" } },
    "zero_optimization": { "stage": 3, "offload_optimizer": { "device": "cpu", "pin_memory": true }, "offload_param": { "device": "cpu", "pin_memory": true }, "overlap_comm": true, "contiguous_gradients": true, "sub_group_size": 1e9, "reduce_bucket_size": "auto", "stage3_prefetch_bucket_size": "auto", "stage3_param_persistence_threshold": "auto", "stage3_max_live_parameters": 1e9, "stage3_max_reuse_distance": 1e9, "stage3_gather_16bit_weights_on_model_save": true },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 1e5,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
```

### 1.2 Python 依赖

需要安装以下依赖包：
```bash
pip install torch torchvision transformers deepspeed peft accelerate
pip install pillow shortuuid tqdm packaging einops
pip install flash-attn --no-build-isolation  # 可选，加速注意力计算
```

---

## 二、训练脚本路径修改（⭐ 最关键）

### 2.1 [MODIFY] [finetune_OrdChain_lora.sh](file:///d:/repro/OrderChain-main/scripts/finetune_OrdChain_lora.sh) — OrderChain LoRA 微调（主脚本）

这是论文核心实验用的训练脚本，有 **5 处 `...` 占位符**需替换：

| 行号 | 原内容 | 修改为 | 说明 |
|------|--------|--------|------|
| 8 | `--deepspeed .../scripts/zero3.json` | `--deepspeed ./scripts/zero3.json` | DeepSpeed 配置路径 |
| 9 | `--model_name_or_path .../llava-v1.5-7b` | `--model_name_or_path /你的路径/llava-v1.5-7b` | LLaVA-1.5-7B 基础模型路径 |
| 11 | `--data_path .../Adience/Adience_llava_train.json` | `--data_path /你的路径/Adience/Adience_llava_train.json` | 对应数据集的训练数据路径 |
| 12 | `--image_folder .../dataset/images/` | `--image_folder /你的路径/dataset/images/` | 图片文件夹根目录 |
| 13 | `--vision_tower .../clip-vit-large-patch14-336` | `--vision_tower /你的路径/clip-vit-large-patch14-336` | CLIP 视觉编码器路径 |
| 21 | `--output_dir .../checkpoints/checkpoints_Adience/llava-v1.5-7b-task-lora` | `--output_dir /你的路径/checkpoints/...` | 输出 checkpoint 路径 |

> [!TIP]
> 如果你要在不同数据集上训练，需要为 **Adience, Aesthetic, Eyepacs, HistoricalColor** 各复制并修改一份脚本。

---

## 三、评估脚本路径修改

### 3.1 所有 4 个评估脚本均缺少 `import os`

> [!WARNING]
> 所有评估脚本 (`test_OrderChain_*.py`) 都使用了 `os.path.expanduser` 和 `os.makedirs`，但**没有 `import os`**，运行时会直接报 `NameError`。

需要在以下 4 个文件中添加 `import os`：

| 文件 | 修改位置 |
|------|----------|
| [test_OrderChain_Adience.py](file:///d:/repro/OrderChain-main/llava/eval/test_OrderChain_Adience.py) | 第1行后加 `import os` |
| [test_OrderChain_Aesthetic.py](file:///d:/repro/OrderChain-main/llava/eval/test_OrderChain_Aesthetic.py) | 第1行后加 `import os` |
| [test_OrderChain_Eyepacs.py](file:///d:/repro/OrderChain-main/llava/eval/test_OrderChain_Eyepacs.py) | 第1行后加 `import os` |
| [test_OrderChain_HCI.py](file:///d:/repro/OrderChain-main/llava/eval/test_OrderChain_HCI.py) | 第1行后加 `import os` |

### 3.2 评估脚本中的 `...` 路径占位符

每个评估脚本有 **3 处** 需修改的默认路径参数：

#### [MODIFY] test_OrderChain_Adience.py (第 153-156 行)
```diff
- parser.add_argument("--model-path", type=str, default=".../checkpoints/checkpoints_Adience/llava-v1.5-7b-task-lora-merged")
+ parser.add_argument("--model-path", type=str, default="/你的实际路径/checkpoints/checkpoints_Adience/llava-v1.5-7b-task-lora-merged")
- parser.add_argument("--question-file", type=str, default=".../Adience/Adience_llava_test.json")
+ parser.add_argument("--question-file", type=str, default="/你的实际路径/Adience/Adience_llava_test.json")
- parser.add_argument("--answers-file", type=str, default=".../Adience/Adience_llava_test_answer.jsonl")
+ parser.add_argument("--answers-file", type=str, default="/你的实际路径/Adience/Adience_llava_test_answer.jsonl")
```

#### [MODIFY] test_OrderChain_Aesthetic.py / Eyepacs / HCI — 同上，对应替换各自的数据集名称

---

## 四、数据集准备

### 4.1 需要下载的数据集

论文使用了 4 个序数回归数据集：

| 数据集 | 任务 | 来源 |
|--------|------|------|
| **Adience** | 人脸年龄估计 (8 类) | [Adience Benchmark](https://talhassner.github.io/home/projects/Adience/Adience-data.html) |
| **Aesthetic (AVA)** | 图像美学评分 (5 类) | [AVA Dataset](https://github.com/mtobeiyf/ava_downloader) |
| **Eyepacs (DR)** | 糖尿病视网膜病变分级 (5 类) | [Kaggle Diabetic Retinopathy](https://www.kaggle.com/c/diabetic-retinopathy-detection) |
| **HistoricalColor (HCI)** | 历史彩色图像分类 (5 类) | [HCI Dataset](https://graphics.cs.cmu.edu/projects/historicalColor/) |

### 4.2 数据预处理

> [!IMPORTANT]
> 下载原始数据后，需要使用 [data_preprocessed.ipynb](file:///d:/repro/OrderChain-main/datasets/data_preprocessed.ipynb) 将数据转换为 LLaVA 对话格式（OrderChain 的 RO-CoT 提示格式）。

预处理后应生成每个数据集对应的：
- `{Dataset}_llava_train.json` — 训练集（OrderChain CoT 对话格式）
- `{Dataset}_llava_test.json` — 测试集

数据格式参考现有的 `Adience_llava_test_answer.jsonl`，每条样本包含：
- `id`: 样本编号
- `image`: 图片路径
- `conversations`: 多轮对话（OrderChain 的三级层次化推理：粗分类 → 细分类 → 最终预测）

### 4.3 数据放置位置

确保图片和 JSON 文件路径与训练/评估脚本中配置的路径一致。推荐目录结构：
```
/你的数据根目录/
├── Adience/
│   ├── images/
│   ├── Adience_llava_train.json
│   └── Adience_llava_test.json
├── Aesthetic/
│   ├── images/
│   ├── Aesthetic_llava_train.json
│   └── Aesthetic_llava_test.json
├── Eyepacs/
│   ├── images/
│   ├── Eyepacs_llava_train.json
│   └── Eyepacs_llava_test.json
└── HistoricalColor/
    ├── images/
    ├── HCI_llava_train.json
    └── HCI_llava_test.json
```

---

## 五、预训练模型下载

### 5.1 LLaVA-v1.5-7B 基础模型

```bash
# 方式1: 从 HuggingFace 下载
git lfs install
git clone https://huggingface.co/liuhaotian/llava-v1.5-7b

# 方式2: 使用国内镜像
# HF_ENDPOINT=https://hf-mirror.com git clone ...
```

### 5.2 CLIP 视觉编码器

```bash
git clone https://huggingface.co/openai/clip-vit-large-patch14-336
```

> [!NOTE]
> 如果无法访问 HuggingFace，可设置 `TRANSFORMERS_OFFLINE=1` 并提前下载模型到本地路径。训练脚本 `finetune_OrdChain_lora.sh` 已包含 `export TRANSFORMERS_OFFLINE=1`。

---

## 六、完整的复现步骤

### Step 1: 数据准备
1. 下载 4 个数据集的原始图片
2. 运行 `datasets/data_preprocessed.ipynb` 生成 OrderChain 格式的训练/测试 JSON

### Step 2: 修改训练脚本路径
修改 `scripts/finetune_OrdChain_lora.sh` 中的所有 `...` 为实际路径

### Step 3: 创建 DeepSpeed 配置
在 `scripts/` 下创建 `zero3.json`（见第一节）

### Step 4: 训练
```bash
# 以 Adience 数据集为例
bash scripts/finetune_OrdChain_lora.sh
```

### Step 5: 合并 LoRA 权重

> [!WARNING]
> 代码中**没有提供** LoRA 合并脚本，评估脚本默认路径后缀是 `-merged`。你需要自己合并：

```python
from peft import PeftModel
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
import torch

# 加载基础模型
base_model = LlavaLlamaForCausalLM.from_pretrained(
    "/你的路径/llava-v1.5-7b",
    torch_dtype=torch.float16
)

# 加载 LoRA 适配器
model = PeftModel.from_pretrained(base_model, "/你的路径/checkpoints/checkpoints_Adience/llava-v1.5-7b-task-lora")

# 合并并保存
model = model.merge_and_unload()
model.save_pretrained("/你的路径/checkpoints/checkpoints_Adience/llava-v1.5-7b-task-lora-merged")
# 同时拷贝 tokenizer 配置文件
```

### Step 6: 评估
```bash
python llava/eval/test_OrderChain_Adience.py \
    --model-path /你的路径/checkpoints/checkpoints_Adience/llava-v1.5-7b-task-lora-merged \
    --question-file /你的路径/Adience/Adience_llava_test.json \
    --answers-file /你的路径/Adience/Adience_llava_test_answer.jsonl
```

---

## 修改汇总清单

| # | 类别 | 文件 | 修改内容 |
|---|------|------|----------|
| 1 | **新建** | `scripts/zero2.json` | DeepSpeed ZeRO-2 配置 |
| 2 | **新建** | `scripts/zero3.json` | DeepSpeed ZeRO-3 配置 |
| 3 | **路径** | `scripts/finetune_OrdChain_lora.sh` | 6 处 `...` → 实际路径 |
| 4 | **路径** | `scripts/finetune.sh` | 根据需要修改路径（可选） |
| 5 | **路径** | `scripts/finetune_lora.sh` | 根据需要修改路径（可选） |
| 6 | **路径** | `scripts/finetune_task.sh` | 根据需要修改路径（可选） |
| 7 | **路径** | `scripts/finetune_task_lora.sh` | 根据需要修改路径（可选） |
| 8 | **路径** | `scripts/pretrain.sh` | 根据需要修改路径（可选） |
| 9 | **Bug** | `llava/eval/test_OrderChain_Adience.py` | 补充 `import os` |
| 10 | **Bug** | `llava/eval/test_OrderChain_Aesthetic.py` | 补充 `import os` |
| 11 | **Bug** | `llava/eval/test_OrderChain_Eyepacs.py` | 补充 `import os` |
| 12 | **Bug** | `llava/eval/test_OrderChain_HCI.py` | 补充 `import os` |
| 13 | **路径** | `llava/eval/test_OrderChain_Adience.py` | 3 处 default 路径 |
| 14 | **路径** | `llava/eval/test_OrderChain_Aesthetic.py` | 3 处 default 路径 |
| 15 | **路径** | `llava/eval/test_OrderChain_Eyepacs.py` | 3 处 default 路径 |
| 16 | **路径** | `llava/eval/test_OrderChain_HCI.py` | 3 处 default 路径 |
| 17 | **数据** | `datasets/` | 下载并预处理 4 个数据集 |
| 18 | **模型** | (外部下载) | 下载 LLaVA-v1.5-7B |
| 19 | **模型** | (外部下载) | 下载 CLIP-ViT-L-14-336 |
| 20 | **脚本** | (需自行编写) | LoRA 权重合并脚本 |

## Open Questions

> [!IMPORTANT]
> 1. 你目前运行环境是什么？Linux 服务器还是 Windows？有几张 GPU？什么型号？
> 2. 你打算复现所有 4 个数据集的实验，还是只复现其中一个（如 Adience）？
> 3. LLaVA-v1.5-7B 模型和 CLIP 视觉编码器是否已经下载好了？
> 4. 4 个数据集的原始图片是否已经准备好了？
