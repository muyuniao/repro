"""ordinalDPO Adience实验 - 极速批处理版 🚀
=====================================================
主要优化：
1. 引入 DataLoader + 多线程预取 (num_workers=4)
2. 开启批处理 (Batch Size = 4)
3. 动态 Padding 支持变长序列
4. 自动利用多显卡 (device_map="auto")
"""

import os, sys, json, random, gc, glob, time
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
import warnings
warnings.filterwarnings("ignore")

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# ====== 路径配置 ======
ORDERCHAIN_PATH = "/home/duomeitinrfx/users/yunhe/reproduce/OrderChain-main"
MODEL_PATH = "/home/duomeitinrfx/users/yunhe/models/llava-v1.5-7b"
ADIENCE_DATA_DIR = "/home/duomeitinrfx/data/Adience"
OUTPUT_DIR = "/home/duomeitinrfx/users/yunhe/reproduce/idea1/adience_ordinal_output"

# ====== 训练超参 (全量版) ======
SFT_EPOCHS = 3
DPO_EPOCHS = 2
SFT_LR = 2e-4
DPO_LR = 1e-5
BETA = 0.1
SEED = 42

# ====== 极速配置 ======
BATCH_SIZE = 8                  # 显存 24GB，设为 8 以保持高效
NUM_WORKERS = 0                # 设为 0 彻底解决死锁问题
USE_GRAD_CHECKPOINT = False      # 显存足够时禁用检查点以提速
TRAIN_SUBSET_SIZE = None          # 恢复全量训练
TEST_SUBSET_SIZE = None           # 恢复全量评估
MAX_DPO_PAIRS = None              # 恢复全量 DPO

sys.path.insert(0, ORDERCHAIN_PATH)
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates

QUESTION = (
    "Estimate the age group of the person in the image.\n"
    "Candidate groups: 1 (0-2), 2 (4-6), 3 (8-13), 4 (15-20), 5 (25-32), 6 (38-43), 7 (48-53), 8 (60+).\n"
    "Answer with only the index number:"
)

# ==================== 1. 数据集定义 ====================
class AdienceDataset(Dataset):
    def __init__(self, data_list, tokenizer, image_processor, model_config):
        self.data = data_list
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]
        
        question = (
            "Estimate the age group of the person in the image.\n"
            "Candidate groups: 1 (0-2), 2 (4-6), 3 (8-13), 4 (15-20), 5 (25-32), 6 (38-43), 7 (48-53), 8 (60+).\n"
            "Answer with only the index number:"
        )
        
        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
        prompt = conv.get_prompt()
        
        # SFT 用：包含答案
        full_text = prompt + " " + str(item["label"])
        
        input_ids = tokenizer_image_token(full_text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        prompt_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        
        labels = input_ids.clone()
        labels[:len(prompt_ids)] = IGNORE_INDEX
        labels[labels == IMAGE_TOKEN_INDEX] = IGNORE_INDEX
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "image_tensor": image_tensor,
            "image_size": image.size,
            "label_idx": item["label"]
        }

def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    images = [item["image_tensor"] for item in batch]
    image_sizes = [item["image_size"] for item in batch]
    
    # Padding
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
    attention_mask = input_ids.ne(0).to(torch.long)
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "images": torch.stack(images),
        "image_sizes": image_sizes,
        "label_indices": [item["label_idx"] for item in batch]
    }

# ==================== 2. 模型加载 ====================
def load_llava_fast(model_path):
    print(f"🚀 加载模型 (单卡模式确保稳定): {model_path}")
    from transformers import BitsAndBytesConfig
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4'
    )
    # 使用单卡模式，避免跨卡通信死锁
    tokenizer, model, image_processor, ctx_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name="llava-v1.5-7b",
        torch_dtype=torch.float16,
        quantization_config=bnb_cfg,
        device_map="cuda:0", 
    )
    model = prepare_model_for_kbit_training(model)
    for p in model.get_vision_tower().parameters(): p.requires_grad = False
    if USE_GRAD_CHECKPOINT: model.gradient_checkpointing_enable()
    model.config.use_cache = False
    return tokenizer, model, image_processor

# ==================== 3. 训练逻辑 ====================
def train_sft_fast(model, tokenizer, image_processor, train_data, epochs, lr):
    print(f"\n🔥 开始 SFT 批处理训练 | Epochs: {epochs} | Batch Size: {BATCH_SIZE}")
    lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "v_proj"], bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_cfg)
    
    ds = AdienceDataset(train_data, tokenizer, image_processor, model.config)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=NUM_WORKERS)
    
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    model.train()
    
    for ep in range(epochs):
        total_loss, n = 0, 0
        t_ep = time.time()
        for batch in dl:
            opt.zero_grad()
            # 这里的 device 设为 model.device (通常是第一张卡的 ID)
            input_ids = batch["input_ids"].to(model.device)
            labels = batch["labels"].to(model.device)
            images = batch["images"].to(model.device, dtype=torch.float16)
            mask = batch["attention_mask"].to(model.device)
            
            out = model(input_ids=input_ids, labels=labels, attention_mask=mask, images=images, image_sizes=batch["image_sizes"])
            loss = out.loss
            loss.backward()
            opt.step()
            
            total_loss += loss.item(); n += 1
            if n % 10 == 0:
                print(f"    Epoch {ep+1} | Step {n*BATCH_SIZE}/{len(train_data)} | Loss: {loss.item():.4f}", flush=True)
        
        avg_loss = total_loss / n
        print(f"✨ Epoch {ep+1} 完成 | 平均 Loss: {avg_loss:.4f} | 耗时: {(time.time()-t_ep)/60:.1f} min")
    return model

# DPO 部分的极速版需要计算 LogP 的 Batch 化
def get_batch_logp(model, tokenizer, batch_input_ids, batch_images, batch_image_sizes, prompt_lens):
    # batch_input_ids 已经包含 prompt + response
    out = model(input_ids=batch_input_ids, images=batch_images, image_sizes=batch_image_sizes)
    logits = out.logits[:, :-1, :]
    labels = batch_input_ids[:, 1:].clone()
    labels[labels < 0] = 0
    
    log_probs = F.log_softmax(logits, dim=-1)
    per_token_logps = torch.gather(log_probs, dim=2, index=labels.unsqueeze(-1)).squeeze(-1)
    
    # 构造掩码只保留 Response 部分
    mask = torch.zeros_like(per_token_logps)
    for i, p_len in enumerate(prompt_lens):
        mask[i, p_len-1:] = 1.0
    
    return (per_token_logps * mask).sum(-1)

def train_dpo_fast(model, tokenizer, image_processor, pairs, ref_c, ref_r, name, epochs, lr, beta):
    print(f"\n⚖️ 开始 {name} 批处理训练 | Epochs: {epochs} | Batch Size: {BATCH_SIZE}")
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    model.train()
    
    num_batches = (len(pairs) + BATCH_SIZE - 1) // BATCH_SIZE
    for ep in range(epochs):
        tl, ta, n = 0, 0, 0
        indices = list(range(len(pairs)))
        random.shuffle(indices)
        
        for i in range(0, len(pairs), BATCH_SIZE):
            batch_idx = indices[i : i + BATCH_SIZE]
            batch_pairs = [pairs[idx] for idx in batch_idx]
            
            # 准备 Chosen 和 Rejected 的数据
            # 为简化，这里直接复用 SFT 的逻辑构造 Input
            def prep_dpo_batch(texts):
                inputs, p_lens = [], []
                imgs, imgs_sizes = [], []
                for j, p in enumerate(batch_pairs):
                    img = Image.open(p["image_path"]).convert("RGB")
                    imgs.append(process_images([img], image_processor, model.config)[0])
                    imgs_sizes.append(img.size)
                    
                    q_text = make_prompt(QUESTION)
                    p_lens.append(tokenizer_image_token(q_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").shape[0])
                    inputs.append(tokenizer_image_token(q_text + " " + texts[j], tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"))
                
                padded_ids = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0).to(model.device)
                return padded_ids, torch.stack(imgs).to(model.device, dtype=torch.float16), imgs_sizes, p_lens

            opt.zero_grad()
            c_ids, c_imgs, c_sizes, c_p_lens = prep_dpo_batch([p["chosen"] for p in batch_pairs])
            r_ids, r_imgs, r_sizes, r_p_lens = prep_dpo_batch([p["rejected"] for p in batch_pairs])
            
            pc = get_batch_logp(model, tokenizer, c_ids, c_imgs, c_sizes, c_p_lens)
            pr = get_batch_logp(model, tokenizer, r_ids, r_imgs, r_sizes, r_p_lens)
            
            rc = torch.tensor([ref_c[idx] for idx in batch_idx], device=model.device)
            rr = torch.tensor([ref_r[idx] for idx in batch_idx], device=model.device)
            weights = torch.tensor([p["weight"] for p in batch_pairs], device=model.device)
            
            logits = beta * ((pc - rc) - (pr - rr))
            loss = (-weights * F.logsigmoid(logits)).mean()
            loss.backward(); opt.step()
            
            tl += loss.item(); ta += (logits > 0).float().mean().item(); n += 1
            if n % 10 == 0: print(f"    Step {n*BATCH_SIZE}/{len(pairs)} | Loss: {loss.item():.4f}", flush=True)
            
    return model

# ==================== 4. 评估 ====================
def evaluate_fast(model, tokenizer, image_processor, test_data, name):
    print(f"📊 评估 {name} | 总数: {len(test_data)}")
    model.eval()
    digit_toks = {d: tokenizer.encode(f" {d}", add_special_tokens=False)[-1] for d in range(1, 9)}
    preds, labels, violations, checks = [], [], 0, 0
    
    for i in range(0, len(test_data), BATCH_SIZE * 2): # 评估可以用更大 Batch
        batch = test_data[i : i + BATCH_SIZE * 2]
        imgs = []
        for item in batch:
            img = Image.open(item["image_path"]).convert("RGB")
            imgs.append(process_images([img], image_processor, model.config)[0])
            labels.append(item["label"])
        
        batch_imgs = torch.stack(imgs).to(model.device, dtype=torch.float16)
        q_prompt = make_prompt(QUESTION)
        input_ids = tokenizer_image_token(q_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
        input_ids = input_ids.repeat(len(batch), 1)
        
        with torch.no_grad():
            out = model(input_ids=input_ids, images=batch_imgs, image_sizes=[Image.open(item["image_path"]).size for item in batch])
            batch_logits = out.logits[:, -1, :] # last token
            
            for j in range(len(batch)):
                dl = {d: batch_logits[j, digit_toks[d]].item() for d in range(1, 9)}
                pred = max(dl, key=dl.get)
                preds.append(pred)
                y = batch[j]["label"]
                for r1 in range(1, 9):
                    for r2 in range(1, 9):
                        if abs(r1-y) < abs(r2-y):
                            checks += 1
                            if dl[r1] < dl[r2]: violations += 1
        if (len(preds)) % 100 == 0: print(f"    进度: {len(preds)}/{len(test_data)}")

    preds, labels = np.array(preds), np.array(labels)
    errs = np.abs(preds - labels)
    from sklearn.metrics import cohen_kappa_score
    qwk = float(cohen_kappa_score(labels, preds, weights="quadratic"))
    r = {"model": name, "acc": float(np.mean(preds==labels)), "mae": float(np.mean(errs)), "violation": violations/max(checks,1), "qwk": qwk}
    print(f"✅ {name} 结果: Acc={r['acc']*100:.1f}%, MAE={r['mae']:.4f}, QWK={qwk:.4f}")
    return r

# ==================== 5. 加载数据逻辑 (复用) ====================
def load_adience_dataset(data_dir):
    print(f"📂 加载 Adience 数据集: {data_dir}")
    train_df = pd.read_csv(os.path.join(data_dir, "Adience_train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "Adience_test.csv"))
    def process(df):
        data = []
        for _, row in df.iterrows():
            path = row['image_path'].replace("/Aidence/", "/Adience/")
            if os.path.exists(path): data.append({"image_path": path, "label": int(row['label']) + 1})
        return data
    return process(train_df), process(test_df)

def make_prompt(question):
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()

def gen_dpo_pairs(data, n_pairs=2, distance_weight=False):
    pairs = []
    for item in data:
        y = item["label"]
        wrongs = [l for l in range(1, 9) if l != y]
        for _ in range(n_pairs):
            rej = random.choice(wrongs)
            d = abs(y - rej)
            pairs.append({"image_path": item["image_path"], "label": y, "chosen": str(y), "rejected": str(rej), "weight": float(d) if distance_weight else 1.0})
    return pairs

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    t0 = time.time()
    
    train_data, test_data = load_adience_dataset(ADIENCE_DATA_DIR)
    if TRAIN_SUBSET_SIZE: train_data = random.sample(train_data, TRAIN_SUBSET_SIZE)
    if TEST_SUBSET_SIZE: test_data = random.sample(test_data, TEST_SUBSET_SIZE)
    
    # [1] SFT
    tokenizer, model, image_processor = load_llava_fast(MODEL_PATH)
    model = train_sft_fast(model, tokenizer, image_processor, train_data, SFT_EPOCHS, SFT_LR)
    sft_path = os.path.join(OUTPUT_DIR, "sft_adapter")
    model.save_pretrained(sft_path)
    print(f"SFT 权重已保存至: {sft_path}")
    results = [evaluate_fast(model, tokenizer, image_processor, test_data, "SFT-Only")]
    
    # [2] Ref LogProbs (由于 DPO 也要 Batch 化，这里先简单串行预计算)
    print("\n🔍 预计算 Ref LogProbs...")
    model.eval()
    std_pairs = gen_dpo_pairs(train_data, 1, False)
    rs_pairs = gen_dpo_pairs(train_data, 1, True)
    ref_c, ref_r = [], []
    with torch.no_grad():
        for i, p in enumerate(std_pairs):
            # 这里暂时用之前的串行逻辑，因为预计算只跑一次
            def get_lp(txt, path):
                img = Image.open(path).convert("RGB")
                it = process_images([img], image_processor, model.config).to(model.device, dtype=torch.float16)
                pr = make_prompt(QUESTION)
                ids = tokenizer_image_token(pr + " " + txt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
                o = model(input_ids=ids, images=it, image_sizes=[img.size])
                lp = F.log_softmax(o.logits[:, :-1, :], dim=-1)
                lbl = ids[:, 1:].clone(); lbl[lbl<0]=0
                plp = torch.gather(lp, 2, lbl.unsqueeze(-1)).squeeze(-1)
                msk = torch.zeros_like(plp); msk[0, tokenizer_image_token(pr, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").shape[0]-1:] = 1.0
                return (plp * msk).sum().item()
            ref_c.append(get_lp(p["chosen"], p["image_path"]))
            ref_r.append(get_lp(p["rejected"], p["image_path"]))
            if (i+1)%100==0: print(f"    {i+1}/{len(std_pairs)}")
    
    # [3] StdDPO
    model = train_dpo_fast(model, tokenizer, image_processor, std_pairs, ref_c, ref_r, "Standard DPO", DPO_EPOCHS, DPO_LR, BETA)
    std_dpo_path = os.path.join(OUTPUT_DIR, "std_dpo_adapter")
    model.save_pretrained(std_dpo_path)
    print(f"StdDPO 权重已保存至: {std_dpo_path}")
    results.append(evaluate_fast(model, tokenizer, image_processor, test_data, "SFT+StdDPO"))
    
    # [4] RS-DPO
    # 重新加载 SFT 模型进行 RS-DPO
    del model; gc.collect(); torch.cuda.empty_cache()
    tokenizer, model, image_processor = load_llava_fast(MODEL_PATH)
    model = PeftModel.from_pretrained(model, sft_path, is_trainable=True)
    model = train_dpo_fast(model, tokenizer, image_processor, rs_pairs, ref_c, ref_r, "RS-DPO (Ours)", DPO_EPOCHS, DPO_LR, BETA)
    rs_dpo_path = os.path.join(OUTPUT_DIR, "rs_dpo_adapter")
    model.save_pretrained(rs_dpo_path)
    print(f"RS-DPO 权重已保存至: {rs_dpo_path}")
    results.append(evaluate_fast(model, tokenizer, image_processor, test_data, "SFT+RS-DPO"))
    
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f: json.dump(results, f, indent=2)
    print(f"\n🏁 总耗时: {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__": main()
