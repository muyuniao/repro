"""ordinalDPO HCI实验 - 服务器版 (使用原始LLaVA代码加载)
=====================================================
运行:
  cd /home/duomeitinrfx/users/yunhe/reproduce/OrderChain-main
  CUDA_VISIBLE_DEVICES=0 python /path/to/hci_server.py
"""

import os, sys, json, random, gc, glob, time
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 允许命令行指定 GPU

# ====== 路径配置 (按需修改) ======
ORDERCHAIN_PATH = "/home/duomeitinrfx/users/yunhe/reproduce/OrderChain-main"
MODEL_PATH = "/home/duomeitinrfx/users/yunhe/models/llava-v1.5-7b"
HCI_DATA_DIR = "/home/duomeitinrfx/data/HistoricalColor-ECCV2012/data/imgs/decade_database"
OUTPUT_DIR = "/home/duomeitinrfx/users/yunhe/reproduce/hci_ordinal_output"

SFT_EPOCHS = 3         # SFT 3 轮（保证底座分类头完美收敛）
DPO_EPOCHS = 2         # DPO 2 轮（保证偏好对齐充分且不过拟合）
SFT_LR = 2e-4
DPO_LR = 1e-5
BETA = 0.1
SEED = 42

# ====== 性能与加速配置 (加速 10x-100x 🚀) ======
USE_GRAD_CHECKPOINT = False      # 设为 False 以禁用梯度检查点，速度提升 ~30%
TRAIN_SUBSET_SIZE = None          # 恢复为 None，强制使用全部 1075 张图片
MAX_DPO_PAIRS = None              # 恢复为 None，不限制 DPO 样本数

# 加入 LLaVA 代码路径
sys.path.insert(0, ORDERCHAIN_PATH)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
import warnings
warnings.filterwarnings("ignore")

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

DEVICE = "cuda:0"

# ==================== 1. 加载HCI数据集 ====================
def load_hci_dataset(data_dir):
    """从本地目录加载HCI数据集, 支持多种目录结构"""
    print(f"加载 HCI 数据集: {data_dir}")
    data = []

    # 尝试结构1: 按年代分文件夹 (1930s/, 1940s/, ...)
    decade_dirs = sorted(glob.glob(os.path.join(data_dir, "*")))
    decade_map = {}
    for d in decade_dirs:
        if os.path.isdir(d):
            name = os.path.basename(d)
            for decade, label in [("1930", 1), ("1940", 2), ("1950", 3), ("1960", 4), ("1970", 5)]:
                if decade in name:
                    decade_map[d] = label
                    break

    if decade_map:
        print(f"  检测到按年代分文件夹结构: {len(decade_map)} 个文件夹")
        for folder, label in sorted(decade_map.items(), key=lambda x: x[1]):
            imgs = glob.glob(os.path.join(folder, "*.jpg")) + \
                   glob.glob(os.path.join(folder, "*.jpeg")) + \
                   glob.glob(os.path.join(folder, "*.png")) + \
                   glob.glob(os.path.join(folder, "*.JPG"))
            for img_path in imgs:
                data.append({"image_path": img_path, "label": label})
            print(f"    {os.path.basename(folder)}: {len(imgs)} images → label {label}")
    else:
        # 尝试结构2: 所有图片在一个目录, 文件名包含年代信息
        all_imgs = glob.glob(os.path.join(data_dir, "**", "*.jpg"), recursive=True) + \
                   glob.glob(os.path.join(data_dir, "**", "*.png"), recursive=True) + \
                   glob.glob(os.path.join(data_dir, "**", "*.jpeg"), recursive=True)
        print(f"  未检测到年代文件夹, 找到 {len(all_imgs)} 张图片")
        print(f"  目录内容: {os.listdir(data_dir)[:20]}")
        # 尝试从路径中提取年代
        for img_path in all_imgs:
            for decade, label in [("1930", 1), ("1940", 2), ("1950", 3), ("1960", 4), ("1970", 5)]:
                if decade in img_path:
                    data.append({"image_path": img_path, "label": label})
                    break

    if not data:
        print(f"  ❌ 无法加载数据! 请检查 {data_dir} 的目录结构")
        print(f"  期望结构: {data_dir}/1930s/*.jpg, {data_dir}/1940s/*.jpg, ...")
        sys.exit(1)

    # 划分训练/测试
    random.seed(SEED)
    by_class = {c: [] for c in range(1, 6)}
    for d in data:
        by_class[d["label"]].append(d)

    train, test = [], []
    for c in range(1, 6):
        items = by_class[c]
        random.shuffle(items)
        split = max(len(items) - 50, int(len(items) * 0.8))
        train.extend(items[:split])
        test.extend(items[split:])

    random.shuffle(train)
    random.shuffle(test)
    print(f"  训练集: {len(train)} | 测试集: {len(test)}")
    return train, test

# ==================== 2. LLaVA 模型加载 ====================
def load_llava(model_path):
    print(f"加载 LLaVA: {model_path}")
    from transformers import BitsAndBytesConfig
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4'
    )
    tokenizer, model, image_processor, ctx_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name="llava-v1.5-7b",
        torch_dtype=torch.float16,
        quantization_config=bnb_cfg,
        device_map="cuda:0",
    )
    model = prepare_model_for_kbit_training(model)
    # 冻结视觉编码器节省显存
    for p in model.get_vision_tower().parameters():
        p.requires_grad = False
    # 启用 gradient checkpointing
    if USE_GRAD_CHECKPOINT:
        model.gradient_checkpointing_enable()
        print("  [Perf] 已启用梯度检查点 (节省显存，减慢速度)")
    else:
        print("  [Perf] 已禁用梯度检查点 (增加显存，速度提升 ~30%)")
    model.config.use_cache = False  # 消除警告
    model.eval()
    return tokenizer, model, image_processor

def make_prompt(question):
    """构造 LLaVA v1 格式的 prompt"""
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()

QUESTION = (
    "This is a historical color photograph. "
    "Classify which decade it was taken in.\n"
    "1=1930s, 2=1940s, 3=1950s, 4=1960s, 5=1970s\n"
    "Answer with only the number:"
)

# ==================== 3. 前向传播 (获取 logits / logprobs) ====================
def get_logits_for_image(model, tokenizer, image_processor, image_path):
    """获取模型对图片的 next-token logits"""
    image = Image.open(image_path).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = image_tensor.to(DEVICE, dtype=torch.float16)

    prompt = make_prompt(QUESTION)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(input_ids=input_ids, images=image_tensor, image_sizes=[image.size])
    return out.logits[0, -1, :]  # last token logits

def get_response_logp(model, tokenizer, image_processor, image_path, response_text):
    """计算 response 的 log probability"""
    image = Image.open(image_path).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = image_tensor.to(DEVICE, dtype=torch.float16)

    prompt = make_prompt(QUESTION)
    full_text = prompt + " " + response_text

    prompt_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    full_ids = tokenizer_image_token(full_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")

    prompt_len = prompt_ids.shape[0]
    input_ids = full_ids.unsqueeze(0).to(DEVICE)

    out = model(input_ids=input_ids, images=image_tensor, image_sizes=[image.size])
    logits = out.logits
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    lp = F.log_softmax(shift_logits, dim=-1)
    
    # 修复CUDA越界错误：shift_labels中包含IMAGE_TOKEN_INDEX（负数）
    shift_labels_safe = shift_labels.clone()
    shift_labels_safe[shift_labels_safe < 0] = 0
    
    token_lps = torch.gather(lp, 2, shift_labels_safe.unsqueeze(-1)).squeeze(-1)

    sl = shift_labels.shape[1]
    resp_mask = torch.zeros(1, sl, device=DEVICE)
    p = max(prompt_len - 1, 0)
    if p < sl: resp_mask[0, p:] = 1.0
    return (token_lps * resp_mask).sum(dim=-1)

# ==================== 4. SFT 训练 ====================
def train_sft(model, tokenizer, image_processor, train_data, epochs, lr):
    print(f"\n  SFT | epochs={epochs}, lr={lr}, data={len(train_data)}")
    # 添加 LoRA
    lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                          target_modules=["q_proj", "v_proj"],
                          bias="none", task_type="CAUSAL_LM")
    model.enable_input_require_grads()
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    model.train()

    for ep in range(epochs):
        random.shuffle(train_data)
        total_loss, n = 0, 0
        for item in train_data:
            opt.zero_grad(set_to_none=True)
            image = Image.open(item["image_path"]).convert("RGB")
            image_tensor = process_images([image], image_processor, model.config)
            image_tensor = image_tensor.to(DEVICE, dtype=torch.float16)

            prompt = make_prompt(QUESTION)
            full_text = prompt + " " + str(item["label"])
            
            prompt_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            prompt_len = prompt_ids.shape[0]

            input_ids = tokenizer_image_token(full_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            input_ids = input_ids.unsqueeze(0).to(DEVICE)
            labels = input_ids.clone()
            
            # Mask out prompt tokens (-100 is PyTorch's default ignore_index)
            labels[0, :prompt_len] = -100
            # Replace IMAGE_TOKEN_INDEX (-200) with IGNORE_INDEX (-100) to avoid CUDA assertion
            labels[labels == IMAGE_TOKEN_INDEX] = -100

            out = model(input_ids=input_ids, images=image_tensor,
                       image_sizes=[image.size], labels=labels)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item(); n += 1
            if n % 10 == 0: print(f"    step {n}/{len(train_data)} loss={loss.item():.4f}")
        print(f"  Epoch {ep+1}/{epochs} | AvgLoss: {total_loss/n:.4f}")
    return model

# ==================== 5. DPO 训练 ====================
def gen_dpo_pairs(data, n_pairs=2, distance_weight=False):
    random.seed(SEED)
    pairs = []
    for item in data:
        y = item["label"]
        wrongs = [l for l in range(1,6) if l != y]
        for _ in range(n_pairs):
            rej = random.choice(wrongs)
            d = abs(y - rej)
            pairs.append({
                "image_path": item["image_path"], "label": y,
                "chosen": str(y), "rejected": str(rej),
                "distance": d, "weight": float(d) if distance_weight else 1.0,
            })
    random.shuffle(pairs)
    if MAX_DPO_PAIRS is not None and len(pairs) > MAX_DPO_PAIRS:
        pairs = pairs[:MAX_DPO_PAIRS]
        print(f"  [Perf] DPO pairs 已采样限制至: {len(pairs)}")
    return pairs

def precompute_ref(model, tokenizer, image_processor, pairs):
    print(f"  预计算 ref logprobs ({len(pairs)} pairs)...")
    model.eval()
    c_lps, r_lps = [], []
    with torch.no_grad():
        for i, p in enumerate(pairs):
            cl = get_response_logp(model, tokenizer, image_processor, p["image_path"], p["chosen"])
            rl = get_response_logp(model, tokenizer, image_processor, p["image_path"], p["rejected"])
            c_lps.append(cl.item()); r_lps.append(rl.item())
            if (i+1) % 200 == 0: print(f"    {i+1}/{len(pairs)}")
    return c_lps, r_lps

def train_dpo(model, tokenizer, image_processor, pairs, ref_c, ref_r, name, epochs, lr, beta):
    print(f"\n{'='*55}\n  {name} | ep={epochs}, lr={lr}, beta={beta}\n{'='*55}")
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    model.train()
    indices = list(range(len(pairs)))
    for ep in range(epochs):
        random.shuffle(indices)
        tl, ta, n = 0, 0, 0
        for idx in indices:
            p = pairs[idx]
            opt.zero_grad()
            pc = get_response_logp(model, tokenizer, image_processor, p["image_path"], p["chosen"])
            pr = get_response_logp(model, tokenizer, image_processor, p["image_path"], p["rejected"])
            rc = torch.tensor(ref_c[idx], device=DEVICE)
            rr = torch.tensor(ref_r[idx], device=DEVICE)
            logits = beta * ((pc - rc) - (pr - rr))
            w = torch.tensor(p["weight"], device=DEVICE)
            loss = (-w * F.logsigmoid(logits)).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item(); ta += (logits > 0).float().item(); n += 1
            if n % 10 == 0: print(f"    step {n}/{len(pairs)} loss={loss.item():.4f}")
        print(f"  Epoch {ep+1}/{epochs} | Loss: {tl/n:.4f} | PrefAcc: {ta/n:.4f}")
    return model

# ==================== 6. 评估 ====================
def evaluate(model, tokenizer, image_processor, test_data, name):
    model.eval()
    digit_toks = {d: tokenizer.encode(f" {d}", add_special_tokens=False)[-1] for d in range(1,6)}
    preds, labels, violations, checks = [], [], 0, 0

    with torch.no_grad():
        for item in test_data:
            logits = get_logits_for_image(model, tokenizer, image_processor, item["image_path"])
            dl = {d: logits[t].item() for d, t in digit_toks.items()}
            pred = max(dl, key=dl.get)
            preds.append(pred); labels.append(item["label"])
            y = item["label"]
            for i in range(1,6):
                for j in range(1,6):
                    if abs(i-y) < abs(j-y):
                        checks += 1
                        if dl[i] < dl[j]: violations += 1

    preds, labels = np.array(preds), np.array(labels)
    errs = np.abs(preds - labels)

    # 计算额外的指标 (QWK, F1, Precision, Recall)
    try:
        from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support
        qwk = float(cohen_kappa_score(labels, preds, weights="quadratic"))
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
        precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    except Exception as e:
        print(f"指标计算异常: {e}")
        qwk, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0
        precision_m, recall_m, f1_m = 0.0, 0.0, 0.0

    r = {"model": name, "acc": float(np.mean(preds==labels)),
         "one_off": float(np.mean(errs<=1)), "mae": float(np.mean(errs)),
         "violation": violations/max(checks,1),
         "qwk": qwk,
         "f1_weighted": f1, "precision_weighted": precision, "recall_weighted": recall,
         "f1_macro": f1_m, "precision_macro": precision_m, "recall_macro": recall_m,
         "err_dist": {d: float(np.mean(errs==d)) for d in range(5)}}

    print(f"\n--- {name} ---")
    print(f"  Acc: {r['acc']*100:.1f}% | 1-off: {r['one_off']*100:.1f}% | "
          f"MAE: {r['mae']:.4f} | Violation: {r['violation']*100:.1f}%")
    print(f"  QWK: {qwk:.4f} | F1-weighted: {f1*100:.1f}% | Prec-weighted: {precision*100:.1f}% | Rec-weighted: {recall*100:.1f}%")
    print(f"  Errors: " + " | ".join(f"d={d}:{r['err_dist'][d]*100:.1f}%" for d in range(5)))
    return r

def free(model):
    del model; gc.collect(); torch.cuda.empty_cache()

# ==================== 7. Main ====================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    t0 = time.time()

    print("=" * 60)
    print("  OrdinalDPO HCI实验 (LLaVA-v1.5-7B, FP16)")
    print("=" * 60)

    train_data, test_data = load_hci_dataset(HCI_DATA_DIR)
    if TRAIN_SUBSET_SIZE is not None and len(train_data) > TRAIN_SUBSET_SIZE:
        random.seed(SEED)
        train_data = random.sample(train_data, TRAIN_SUBSET_SIZE)
        print(f"  [Perf] 已对训练集采样子集进行快速训练，当前数量: {len(train_data)}")
    
    std_pairs = gen_dpo_pairs(train_data, 2, False)
    rs_pairs  = gen_dpo_pairs(train_data, 2, True)
    print(f"DPO pairs: {len(std_pairs)}")

    results = []

    # [1/5] SFT
    print(f"\n[1/5] SFT Training")
    tokenizer, model, image_processor = load_llava(MODEL_PATH)
    model = train_sft(model, tokenizer, image_processor, train_data, SFT_EPOCHS, SFT_LR)
    r_sft = evaluate(model, tokenizer, image_processor, test_data, "SFT-Only")
    results.append(r_sft)
    sft_path = f"{OUTPUT_DIR}/sft_adapter"
    model.save_pretrained(sft_path)
    free(model)

    # [2/5] 预计算 ref logprobs
    print(f"\n[2/5] 预计算 Ref Log-Probs")
    tokenizer, ref, image_processor = load_llava(MODEL_PATH)
    ref = PeftModel.from_pretrained(ref, sft_path)
    ref_c, ref_r = precompute_ref(ref, tokenizer, image_processor, std_pairs)
    free(ref)

    # [3/5] StdDPO
    print(f"\n[3/5] Standard DPO")
    tokenizer, model, image_processor = load_llava(MODEL_PATH)
    model = PeftModel.from_pretrained(model, sft_path, is_trainable=True)
    model = train_dpo(model, tokenizer, image_processor, std_pairs, ref_c, ref_r,
                      "Standard DPO", DPO_EPOCHS, DPO_LR, BETA)
    r_std = evaluate(model, tokenizer, image_processor, test_data, "SFT+StdDPO")
    results.append(r_std)
    std_dpo_path = f"{OUTPUT_DIR}/std_dpo_adapter"
    model.save_pretrained(std_dpo_path)
    print(f"Standard DPO 权重已保存至: {std_dpo_path}")
    free(model)

    # [4/5] RS-DPO
    print(f"\n[4/5] RS-DPO (Ours)")
    tokenizer, model, image_processor = load_llava(MODEL_PATH)
    model = PeftModel.from_pretrained(model, sft_path, is_trainable=True)
    model = train_dpo(model, tokenizer, image_processor, rs_pairs, ref_c, ref_r,
                      "RS-DPO (Ours)", DPO_EPOCHS, DPO_LR, BETA)
    r_rs = evaluate(model, tokenizer, image_processor, test_data, "SFT+RS-DPO")
    results.append(r_rs)
    rs_dpo_path = f"{OUTPUT_DIR}/rs_dpo_adapter"
    model.save_pretrained(rs_dpo_path)
    print(f"RS-DPO 最佳权重已保存至: {rs_dpo_path}")
    free(model)

    # [5/5] 汇总
    elapsed = (time.time() - t0) / 60
    print(f"\n{'='*85}")
    print(f"  HCI 实验结果 (耗时: {elapsed:.1f} min)")
    print(f"{'='*85}")
    print(f"{'Method':<16} {'Acc':>6} {'1-off':>6} {'MAE':>6} {'QWK':>6} {'F1-w':>6} {'Prec-w':>6} {'Rec-w':>6} {'Viol':>6}")
    print("-" * 85)
    for r in results:
        print(f"{r['model']:<16} {r['acc']*100:>5.1f}% {r['one_off']*100:>5.1f}% "
              f"{r['mae']:>6.3f} {r.get('qwk', 0.0):>6.3f} {r.get('f1_weighted', 0.0)*100:>5.1f}% "
              f"{r.get('precision_weighted', 0.0)*100:>5.1f}% {r.get('recall_weighted', 0.0)*100:>5.1f}% "
              f"{r['violation']*100:>5.1f}%")

    s, d, rs = r_sft["mae"], r_std["mae"], r_rs["mae"]
    print(f"\nMAE: SFT={s:.4f} → StdDPO={d:.4f} → RS-DPO={rs:.4f}")
    if rs < d < s:   print("✅ RS-DPO > StdDPO > SFT — idea 成立！")
    elif rs < s:     print("⚠️ RS-DPO 有效但与 StdDPO 接近")
    elif d < s:      print("⚠️ DPO有效但距离加权未超越StdDPO")
    else:            print("❌ DPO未改善SFT")

    with open(f"{OUTPUT_DIR}/results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {OUTPUT_DIR}/results.json")

if __name__ == "__main__":
    main()


