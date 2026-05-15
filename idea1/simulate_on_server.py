"""
OrdinalDPO 文本模拟实验 - 服务器版 (无需bitsandbytes)
=====================================================
策略: 预计算 reference logprobs → 释放 ref model → 加载 policy model 训练
这样一次只需一个 7B 模型在显存中 (~14GB FP16)

运行:
  CUDA_VISIBLE_DEVICES=0 python simulate_on_server.py \
    --model_path /path/to/llava-v1.5-7b
"""

import os, json, random, argparse, gc
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    LlamaForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, PeftModel
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. 合成数据
# ============================================================
TEMPLATES = {
    1: [
        "This product is terrible. It broke after one day. Extremely disappointed. Rating:",
        "Worst purchase ever. Complete waste of money. Totally useless item. Rating:",
        "Awful quality. Does not work at all. I want a refund immediately. Rating:",
        "Horrible experience. The product is defective and dangerous to use. Rating:",
        "Absolutely dreadful. Nothing about this product works properly at all. Rating:",
        "Disgusting quality. Fell apart immediately upon first use attempt. Rating:",
        "Total garbage product. Does not match description at all. A scam. Rating:",
        "Unacceptable product. Damaged on arrival and smells terrible too. Rating:",
    ],
    2: [
        "Not great. The quality is below average and feels cheap overall. Rating:",
        "Disappointing product. Expected much better for the price I paid. Rating:",
        "Below expectations. It works sometimes but is very unreliable now. Rating:",
        "Poor quality overall. Would not recommend this to anyone I know. Rating:",
        "Mediocre at best. Has several major issues that need fixing soon. Rating:",
        "Underwhelming purchase. The materials feel very low quality here. Rating:",
        "Not worth the money. Performs poorly compared to the alternatives. Rating:",
        "Subpar product. Looks okay but functionality is quite lacking here. Rating:",
    ],
    3: [
        "Average product. Nothing special but gets the basic job done here. Rating:",
        "It is okay. Has both good aspects and bad aspects to consider now. Rating:",
        "Decent for the price. Not amazing but not terrible either overall. Rating:",
        "Middle of the road. Some features work well but others do not yet. Rating:",
        "Fair product overall. Meets minimum expectations but nothing more. Rating:",
        "Acceptable quality. Works as intended but lacks any polish at all. Rating:",
        "Standard product. Does what it says with no extra features added. Rating:",
        "Moderate quality item. Neither impressive nor disappointing overall. Rating:",
    ],
    4: [
        "Good product. Works well and I am quite satisfied with it overall. Rating:",
        "Pretty nice product. Quality is above average and worth the price. Rating:",
        "Solid purchase. Does everything it promises and looks great doing. Rating:",
        "Very pleased with this product. Good build quality and nice design. Rating:",
        "Happy with this buy. Reliable and performs better than I expected. Rating:",
        "Great value overall. Functions perfectly and the quality feels nice. Rating:",
        "Nice product overall. Would recommend it to friends and my family. Rating:",
        "Impressed with the quality. Works smoothly and looks very stylish. Rating:",
    ],
    5: [
        "Absolutely perfect. Best product I have ever purchased from anyone. Rating:",
        "Outstanding quality product. Exceeds all my expectations completely. Rating:",
        "Incredible product. Five stars without any hesitation at all here. Rating:",
        "Amazing purchase. Top notch quality and flawless performance daily. Rating:",
        "Exceptional in every single way. Could not be happier with this. Rating:",
        "Superb product. Premium quality and works like a dream every time. Rating:",
        "Phenomenal buy. Best in class and worth every single penny spent. Rating:",
        "Magnificent product. Perfect design perfect function perfect value. Rating:",
    ],
}

def gen_data(n_per_class=80, seed=42):
    random.seed(seed)
    data = []
    for r in range(1, 6):
        for i in range(n_per_class):
            t = TEMPLATES[r][i % len(TEMPLATES[r])]
            data.append({"text": f"{t} {r}", "prompt": t, "label": r})
    random.shuffle(data)
    return data

def gen_dpo_pairs(data, n_pairs=3, distance_weight=False, seed=42):
    random.seed(seed)
    pairs = []
    for item in data:
        y = item["label"]
        wrongs = [l for l in range(1, 6) if l != y]
        for _ in range(n_pairs):
            rej = random.choice(wrongs)
            d = abs(y - rej)
            pairs.append({
                "prompt": item["prompt"],
                "chosen": f" {y}",
                "rejected": f" {rej}",
                "distance": d,
                "weight": float(d) if distance_weight else 1.0,
            })
    random.shuffle(pairs)
    return pairs

# ============================================================
# 2. Datasets
# ============================================================
class SFTDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=96):
        self.items = []
        for d in data:
            enc = tokenizer(d["text"], truncation=True, max_length=max_len,
                           padding="max_length", return_tensors="pt")
            ids = enc["input_ids"].squeeze()
            self.items.append({
                "input_ids": ids,
                "attention_mask": enc["attention_mask"].squeeze(),
                "labels": ids.clone(),
            })
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

# ============================================================
# 3. Log-prob 计算
# ============================================================
def compute_response_logps(model, input_ids, attention_mask, prompt_length):
    """计算 response 部分的 log probability"""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = attention_mask[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_lps = torch.gather(log_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)
    bs, sl = shift_labels.shape
    resp_mask = torch.zeros_like(shift_mask, dtype=torch.float32)
    for i in range(bs):
        p = max(prompt_length[i] - 1, 0)
        if p < sl:
            resp_mask[i, p:] = shift_mask[i, p:].float()
    return (token_lps * resp_mask).sum(dim=-1)

# ============================================================
# 4. 预计算 reference logprobs (关键: 只需一次加载)
# ============================================================
def precompute_ref_logps(model, tokenizer, dpo_pairs, device, max_len=96):
    """预计算所有 DPO 对的 reference log-probs，然后可以释放模型"""
    print("  预计算 reference log-probs...")
    model.eval()
    chosen_lps, rejected_lps = [], []

    with torch.no_grad():
        for i, pair in enumerate(dpo_pairs):
            c_text = pair["prompt"] + pair["chosen"]
            r_text = pair["prompt"] + pair["rejected"]
            p_len = len(tokenizer.encode(pair["prompt"]))

            c_enc = tokenizer(c_text, truncation=True, max_length=max_len,
                             padding="max_length", return_tensors="pt").to(device)
            r_enc = tokenizer(r_text, truncation=True, max_length=max_len,
                             padding="max_length", return_tensors="pt").to(device)

            c_lp = compute_response_logps(model, c_enc["input_ids"],
                                          c_enc["attention_mask"], [p_len])
            r_lp = compute_response_logps(model, r_enc["input_ids"],
                                          r_enc["attention_mask"], [p_len])

            chosen_lps.append(c_lp.item())
            rejected_lps.append(r_lp.item())

            if (i + 1) % 200 == 0:
                print(f"    {i+1}/{len(dpo_pairs)} done")

    print(f"  预计算完成: {len(chosen_lps)} pairs")
    return chosen_lps, rejected_lps

# ============================================================
# 5. DPO 训练 (使用预计算的 ref logprobs)
# ============================================================
class DPOPrecomputedDataset(Dataset):
    def __init__(self, pairs, ref_chosen_lps, ref_rejected_lps, tokenizer, max_len=96):
        self.items = []
        for idx, p in enumerate(pairs):
            c_enc = tokenizer(p["prompt"] + p["chosen"], truncation=True,
                             max_length=max_len, padding="max_length", return_tensors="pt")
            r_enc = tokenizer(p["prompt"] + p["rejected"], truncation=True,
                             max_length=max_len, padding="max_length", return_tensors="pt")
            pl = len(tokenizer.encode(p["prompt"]))
            self.items.append({
                "c_ids": c_enc["input_ids"].squeeze(),
                "c_mask": c_enc["attention_mask"].squeeze(),
                "r_ids": r_enc["input_ids"].squeeze(),
                "r_mask": r_enc["attention_mask"].squeeze(),
                "pl": pl,
                "weight": torch.tensor(p["weight"], dtype=torch.float32),
                "ref_c_lp": torch.tensor(ref_chosen_lps[idx], dtype=torch.float32),
                "ref_r_lp": torch.tensor(ref_rejected_lps[idx], dtype=torch.float32),
            })
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

def train_dpo_precomputed(model, dataset, name, device, epochs=2, lr=2e-5, bs=2, beta=0.1):
    print(f"\n{'='*55}")
    print(f"  {name} | epochs={epochs}, lr={lr}, beta={beta}")
    print(f"{'='*55}")
    loader = DataLoader(dataset, batch_size=bs, shuffle=True)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    model.train()

    for ep in range(epochs):
        total_loss, total_acc, n = 0, 0, 0
        for batch in loader:
            opt.zero_grad()
            c_ids = batch["c_ids"].to(device)
            c_mask = batch["c_mask"].to(device)
            r_ids = batch["r_ids"].to(device)
            r_mask = batch["r_mask"].to(device)
            pl = batch["pl"].tolist()
            w = batch["weight"].to(device)
            ref_c = batch["ref_c_lp"].to(device)
            ref_r = batch["ref_r_lp"].to(device)

            pc = compute_response_logps(model, c_ids, c_mask, pl)
            pr = compute_response_logps(model, r_ids, r_mask, pl)

            logits = beta * ((pc - ref_c) - (pr - ref_r))
            loss = (-w * F.logsigmoid(logits)).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item()
            total_acc += (logits > 0).float().mean().item()
            n += 1

        print(f"  Epoch {ep+1}/{epochs} | Loss: {total_loss/n:.4f} | Acc: {total_acc/n:.4f}")
    return model

# ============================================================
# 6. 评估
# ============================================================
def evaluate(model, tokenizer, test_data, name, device):
    model.eval()
    digit_toks = {}
    for d in range(1, 6):
        toks = tokenizer.encode(f" {d}", add_special_tokens=False)
        digit_toks[d] = toks[-1]

    preds, labels = [], []
    violations, checks = 0, 0

    with torch.no_grad():
        for item in test_data:
            ids = tokenizer.encode(item["prompt"], return_tensors="pt").to(device)
            logits = model(input_ids=ids).logits[0, -1, :]
            dl = {d: logits[t].item() for d, t in digit_toks.items()}
            pred = max(dl, key=dl.get)
            preds.append(pred)
            labels.append(item["label"])

            y = item["label"]
            for i in range(1, 6):
                for j in range(1, 6):
                    if abs(i - y) < abs(j - y):
                        checks += 1
                        if dl[i] < dl[j]:
                            violations += 1

    preds, labels = np.array(preds), np.array(labels)
    errs = np.abs(preds - labels)
    r = {
        "model": name,
        "acc": float(np.mean(preds == labels)),
        "one_off": float(np.mean(errs <= 1)),
        "mae": float(np.mean(errs)),
        "violation": violations / max(checks, 1),
        "err_dist": {d: float(np.mean(errs == d)) for d in range(5)},
    }
    print(f"\n--- {name} ---")
    print(f"  Acc: {r['acc']*100:.1f}% | 1-off: {r['one_off']*100:.1f}% | "
          f"MAE: {r['mae']:.4f} | Violation: {r['violation']*100:.1f}%")
    print(f"  Errors: " + " | ".join(f"d={d}:{r['err_dist'][d]*100:.1f}%" for d in range(5)))
    return r

# ============================================================
# 7. Main
# ============================================================
def free_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()

def load_llm(model_path, device):
    print(f"  加载 LLM from {model_path} (FP16, 忽略视觉部分)...")
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map={"": device},
        ignore_mismatched_sizes=True,
    )
    return model

def add_lora(model, r=16, alpha=32):
    config = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    model.enable_input_require_grads()
    return get_peft_model(model, config)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--sft_epochs", type=int, default=3)
    parser.add_argument("--dpo_epochs", type=int, default=2)
    parser.add_argument("--sft_lr", type=float, default=2e-4)
    parser.add_argument("--dpo_lr", type=float, default=2e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--n_per_class", type=int, default=80)
    parser.add_argument("--output_dir", type=str, default="./ordinal_sim_output")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    seed = 42; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = args.device

    print("=" * 60)
    print("  OrdinalDPO 文本模拟实验 (FP16, 无需bitsandbytes)")
    print(f"  Model: {args.model_path}")
    print(f"  Device: {device}")
    print("=" * 60)

    # --- 数据 ---
    all_data = gen_data(args.n_per_class, seed)
    train = [d for i, d in enumerate(all_data) if i % 5 != 0]
    test = [d for i, d in enumerate(all_data) if i % 5 == 0]
    std_pairs = gen_dpo_pairs(train, 3, distance_weight=False, seed=seed)
    rs_pairs = gen_dpo_pairs(train, 3, distance_weight=True, seed=seed)
    print(f"  Train: {len(train)} | Test: {len(test)} | DPO pairs: {len(std_pairs)}")

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []

    # ===== 阶段1: SFT =====
    print("\n" + "=" * 55)
    print("  [1/5] SFT Training")
    print("=" * 55)
    model = load_llm(args.model_path, device)
    model = add_lora(model)
    model.print_trainable_parameters()

    sft_ds = SFTDataset(train, tokenizer)
    t_args = TrainingArguments(
        output_dir=f"{args.output_dir}/sft",
        num_train_epochs=args.sft_epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=args.sft_lr,
        logging_steps=20,
        save_strategy="no",
        report_to="none",
        fp16=True,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
    )
    Trainer(model=model, args=t_args, train_dataset=sft_ds,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)).train()

    r_sft = evaluate(model, tokenizer, test, "SFT-Only", device)
    results.append(r_sft)

    # 保存 SFT adapter
    sft_path = f"{args.output_dir}/sft_adapter"
    model.save_pretrained(sft_path)
    free_model(model)
    print("  SFT 模型已释放")

    # ===== 阶段2: 预计算 ref logprobs (两种DPO共用) =====
    print("\n" + "=" * 55)
    print("  [2/5] 预计算 Reference Log-Probs")
    print("=" * 55)
    ref = load_llm(args.model_path, device)
    ref = PeftModel.from_pretrained(ref, sft_path)
    ref.eval()

    # 两组 pairs 内容相同(只是权重不同), logprobs 一样, 只需算一次
    ref_c_lps, ref_r_lps = precompute_ref_logps(ref, tokenizer, std_pairs, device)
    free_model(ref)
    print("  Reference 模型已释放")

    # ===== 阶段3: Standard DPO =====
    print("\n" + "=" * 55)
    print("  [3/5] Standard DPO Training")
    print("=" * 55)
    model = load_llm(args.model_path, device)
    model = PeftModel.from_pretrained(model, sft_path, is_trainable=True)

    std_ds = DPOPrecomputedDataset(std_pairs, ref_c_lps, ref_r_lps, tokenizer)
    model = train_dpo_precomputed(model, std_ds, "Standard DPO", device,
                                  args.dpo_epochs, args.dpo_lr, 2, args.beta)
    r_std = evaluate(model, tokenizer, test, "SFT+StdDPO", device)
    results.append(r_std)
    free_model(model)

    # ===== 阶段4: RS-DPO =====
    print("\n" + "=" * 55)
    print("  [4/5] RS-DPO Training (Ours)")
    print("=" * 55)
    model = load_llm(args.model_path, device)
    model = PeftModel.from_pretrained(model, sft_path, is_trainable=True)

    rs_ds = DPOPrecomputedDataset(rs_pairs, ref_c_lps, ref_r_lps, tokenizer)
    model = train_dpo_precomputed(model, rs_ds, "RS-DPO (Ours)", device,
                                  args.dpo_epochs, args.dpo_lr, 2, args.beta)
    r_rs = evaluate(model, tokenizer, test, "SFT+RS-DPO", device)
    results.append(r_rs)
    free_model(model)

    # ===== 阶段5: 汇总 =====
    print("\n" + "=" * 65)
    print("                     实验结果汇总")
    print("=" * 65)
    print(f"{'Method':<18} {'Acc':>7} {'1-off':>7} {'MAE':>7} {'Violation':>9}")
    print("-" * 50)
    for r in results:
        print(f"{r['model']:<18} {r['acc']*100:>6.1f}% {r['one_off']*100:>6.1f}% "
              f"{r['mae']:>7.4f} {r['violation']*100:>8.1f}%")

    s, d, rs = r_sft["mae"], r_std["mae"], r_rs["mae"]
    print(f"\nMAE: SFT={s:.4f} → StdDPO={d:.4f} → RS-DPO={rs:.4f}")
    if rs < d < s:
        print("✅ RS-DPO > StdDPO > SFT: idea 成立！")
    elif rs < s and rs <= d:
        print("⚠️ RS-DPO 有效但与 StdDPO 接近，需调参。")
    elif d < s and rs >= d:
        print("⚠️ DPO有效但距离加权未超越StdDPO。")
    else:
        print("❌ DPO未改善SFT，需重新审视。")

    with open(f"{args.output_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {args.output_dir}/results.json")

if __name__ == "__main__":
    main()
