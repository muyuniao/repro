"""ordinalDPO Adience实验 - 服务器版
=====================================================
运行:
  cd /home/duomeitinrfx/users/yunhe/reproduce/idea1
  CUDA_VISIBLE_DEVICES=2 python adience_rsdpo.py
"""

import os, sys, json, random, gc, glob, time
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
import warnings
warnings.filterwarnings("ignore")

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# ====== 路径配置 (按需修改) ======
ORDERCHAIN_PATH = "/home/duomeitinrfx/users/yunhe/reproduce/OrderChain-main"
MODEL_PATH = "/home/duomeitinrfx/users/yunhe/models/llava-v1.5-7b"
ADIENCE_DATA_DIR = "/home/duomeitinrfx/data/Adience"
OUTPUT_DIR = "/home/duomeitinrfx/users/yunhe/reproduce/idea1/adience_ordinal_output"

SFT_EPOCHS = 3         # 恢复 3 轮确保收敛
DPO_EPOCHS = 2         # 恢复 2 轮确保对齐
SFT_LR = 2e-4
DPO_LR = 1e-5
BETA = 0.1
SEED = 42

# ====== 性能与加速配置 ======
USE_GRAD_CHECKPOINT = False
TRAIN_SUBSET_SIZE = None          # 恢复全量训练
TEST_SUBSET_SIZE = None           # 恢复全量评估
MAX_DPO_PAIRS = None              # 恢复全量 DPO

# 加入 LLaVA 代码路径
sys.path.insert(0, ORDERCHAIN_PATH)
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

DEVICE = "cuda:0"

# ==================== 1. 加载Adience数据集 ====================
def load_adience_dataset(data_dir):
    print(f"加载 Adience 数据集: {data_dir}")
    train_csv = os.path.join(data_dir, "Adience_train.csv")
    test_csv = os.path.join(data_dir, "Adience_test.csv")
    
    if not os.path.exists(train_csv) or not os.path.exists(test_csv):
        print(f"  ❌ 找不到 CSV 文件: {train_csv} 或 {test_csv}")
        sys.exit(1)

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    
    def process_df(df):
        data = []
        for _, row in df.iterrows():
            # 修复路径中的拼写错误 (Aidence -> Adience)
            path = row['image_path'].replace("/Aidence/", "/Adience/")
            if not os.path.exists(path):
                continue
            label = int(row['label']) + 1 # 0-7 -> 1-8
            data.append({"image_path": path, "label": label})
        return data

    train_data = process_df(train_df)
    test_data = process_df(test_df)
    print(f"  训练集: {len(train_data)} | 测试集: {len(test_data)}")
    return train_data, test_data

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
    for p in model.get_vision_tower().parameters():
        p.requires_grad = False
    if USE_GRAD_CHECKPOINT:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.eval()
    return tokenizer, model, image_processor

def make_prompt(question):
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()

QUESTION = (
    "Estimate the age group of the person in the image.\n"
    "Candidate groups: 1 (0-2), 2 (4-6), 3 (8-13), 4 (15-20), 5 (25-32), 6 (38-43), 7 (48-53), 8 (60+).\n"
    "Answer with only the index number:"
)

# ==================== 3. 前馈与评估工具 ====================
def get_logits_for_image(model, tokenizer, image_processor, image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = image_tensor.to(DEVICE, dtype=torch.float16)
    prompt = make_prompt(QUESTION)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(input_ids=input_ids, images=image_tensor, image_sizes=[image.size])
    return out.logits[0, -1, :]

def get_response_logp(model, tokenizer, image_processor, image_path, response_text):
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
    lp = F.log_softmax(out.logits[:, :-1, :], dim=-1)
    shift_labels = input_ids[:, 1:].clone()
    shift_labels[shift_labels < 0] = 0
    token_lps = torch.gather(lp, 2, shift_labels.unsqueeze(-1)).squeeze(-1)
    sl = shift_labels.shape[1]
    resp_mask = torch.zeros(1, sl, device=DEVICE)
    p = max(prompt_len - 1, 0)
    if p < sl: resp_mask[0, p:] = 1.0
    return (token_lps * resp_mask).sum(dim=-1)

# ==================== 4. 训练逻辑 ====================
def train_sft(model, tokenizer, image_processor, train_data, epochs, lr):
    print(f"\n  SFT | epochs={epochs}, lr={lr}, data={len(train_data)}")
    lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "v_proj"], bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_cfg)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    model.train()
    for ep in range(epochs):
        random.shuffle(train_data)
        total_loss, n = 0, 0
        for item in train_data:
            opt.zero_grad()
            image = Image.open(item["image_path"]).convert("RGB")
            image_tensor = process_images([image], image_processor, model.config).to(DEVICE, dtype=torch.float16)
            full_text = make_prompt(QUESTION) + " " + str(item["label"])
            input_ids = tokenizer_image_token(full_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(DEVICE)
            prompt_len = tokenizer_image_token(make_prompt(QUESTION), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").shape[0]
            labels = input_ids.clone()
            labels[0, :prompt_len] = -100
            labels[labels == IMAGE_TOKEN_INDEX] = -100
            loss = model(input_ids=input_ids, images=image_tensor, image_sizes=[image.size], labels=labels).loss
            loss.backward()
            opt.step()
            total_loss += loss.item(); n += 1
            if n % 50 == 0: print(f"    step {n}/{len(train_data)} loss={loss.item():.4f}")
        print(f"  Epoch {ep+1}/{epochs} | AvgLoss: {total_loss/n:.4f}")
    return model

def gen_dpo_pairs(data, n_pairs=2, distance_weight=False):
    random.seed(SEED)
    pairs = []
    for item in data:
        y = item["label"]
        wrongs = [l for l in range(1, 9) if l != y]
        for _ in range(n_pairs):
            rej = random.choice(wrongs)
            d = abs(y - rej)
            pairs.append({
                "image_path": item["image_path"], "label": y,
                "chosen": str(y), "rejected": str(rej),
                "distance": d, "weight": float(d) if distance_weight else 1.0,
            })
    random.shuffle(pairs)
    if MAX_DPO_PAIRS: pairs = pairs[:MAX_DPO_PAIRS]
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
            if (i+1) % 500 == 0: print(f"    {i+1}/{len(pairs)}")
    return c_lps, r_lps

def train_dpo(model, tokenizer, image_processor, pairs, ref_c, ref_r, name, epochs, lr, beta):
    print(f"\n  {name} | ep={epochs}, lr={lr}, beta={beta}")
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
            logits = beta * ((pc - torch.tensor(ref_c[idx], device=DEVICE)) - (pr - torch.tensor(ref_r[idx], device=DEVICE)))
            loss = (-torch.tensor(p["weight"], device=DEVICE) * F.logsigmoid(logits)).mean()
            loss.backward(); opt.step()
            tl += loss.item(); ta += (logits > 0).float().item(); n += 1
            if n % 50 == 0: print(f"    step {n}/{len(pairs)} loss={loss.item():.4f}")
        print(f"  Epoch {ep+1}/{epochs} | Loss: {tl/n:.4f} | PrefAcc: {ta/n:.4f}")
    return model

# ==================== 5. 评估 ====================
def evaluate(model, tokenizer, image_processor, test_data, name):
    print(f"  评估 {name} | data={len(test_data)}")
    model.eval()
    digit_toks = {d: tokenizer.encode(f" {d}", add_special_tokens=False)[-1] for d in range(1, 9)}
    preds, labels, violations, checks = [], [], 0, 0
    with torch.no_grad():
        for i, item in enumerate(test_data):
            logits = get_logits_for_image(model, tokenizer, image_processor, item["image_path"])
            dl = {d: logits[t].item() for d, t in digit_toks.items()}
            pred = max(dl, key=dl.get)
            preds.append(pred); labels.append(item["label"])
            y = item["label"]
            for r1 in range(1, 9):
                for r2 in range(1, 9):
                    if abs(r1-y) < abs(r2-y):
                        checks += 1
                        if dl[r1] < dl[r2]: violations += 1
            if (i+1) % 50 == 0: print(f"    eval {i+1}/{len(test_data)}")
    preds, labels = np.array(preds), np.array(labels)
    errs = np.abs(preds - labels)
    try:
        from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support
        qwk = float(cohen_kappa_score(labels, preds, weights="quadratic"))
        p_w, r_w, f1_w, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    except: qwk, p_w, r_w, f1_w = 0, 0, 0, 0

    r = {"model": name, "acc": float(np.mean(preds==labels)), "one_off": float(np.mean(errs<=1)), "mae": float(np.mean(errs)), "violation": violations/max(checks,1), "qwk": qwk, "f1_weighted": f1_w, "err_dist": {d: float(np.mean(errs==d)) for d in range(8)}}
    print(f"--- {name} --- Acc: {r['acc']*100:.1f}% | MAE: {r['mae']:.4f} | QWK: {qwk:.4f}")
    return r

def free(model):
    del model; gc.collect(); torch.cuda.empty_cache()

# ==================== 6. Main ====================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    t0 = time.time()
    train_data, test_data = load_adience_dataset(ADIENCE_DATA_DIR)
    if TRAIN_SUBSET_SIZE: train_data = random.sample(train_data, TRAIN_SUBSET_SIZE)
    if TEST_SUBSET_SIZE: test_data = random.sample(test_data, TEST_SUBSET_SIZE)
    std_pairs = gen_dpo_pairs(train_data, 2, False)
    rs_pairs  = gen_dpo_pairs(train_data, 2, True)
    
    results = []
    # [SFT]
    tokenizer, model, image_processor = load_llava(MODEL_PATH)
    model = train_sft(model, tokenizer, image_processor, train_data, SFT_EPOCHS, SFT_LR)
    r_sft = evaluate(model, tokenizer, image_processor, test_data, "SFT-Only")
    results.append(r_sft)
    sft_path = f"{OUTPUT_DIR}/sft_adapter"
    model.save_pretrained(sft_path); free(model)
    
    # [Ref]
    tokenizer, ref, image_processor = load_llava(MODEL_PATH)
    ref = PeftModel.from_pretrained(ref, sft_path)
    ref_c, ref_r = precompute_ref(ref, tokenizer, image_processor, std_pairs)
    free(ref)
    
    # [StdDPO]
    tokenizer, model, image_processor = load_llava(MODEL_PATH)
    model = PeftModel.from_pretrained(model, sft_path, is_trainable=True)
    model = train_dpo(model, tokenizer, image_processor, std_pairs, ref_c, ref_r, "Standard DPO", DPO_EPOCHS, DPO_LR, BETA)
    results.append(evaluate(model, tokenizer, image_processor, test_data, "SFT+StdDPO"))
    model.save_pretrained(f"{OUTPUT_DIR}/std_dpo_adapter"); free(model)
    
    # [RS-DPO]
    tokenizer, model, image_processor = load_llava(MODEL_PATH)
    model = PeftModel.from_pretrained(model, sft_path, is_trainable=True)
    model = train_dpo(model, tokenizer, image_processor, rs_pairs, ref_c, ref_r, "RS-DPO (Ours)", DPO_EPOCHS, DPO_LR, BETA)
    results.append(evaluate(model, tokenizer, image_processor, test_data, "SFT+RS-DPO"))
    model.save_pretrained(f"{OUTPUT_DIR}/rs_dpo_adapter"); free(model)
    
    with open(f"{OUTPUT_DIR}/results.json", "w") as f: json.dump(results, f, indent=2)
    print(f"总耗时: {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
