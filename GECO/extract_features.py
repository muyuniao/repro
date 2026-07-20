import os
import gc
import json
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import (
    LlavaForConditionalGeneration, AutoProcessor,
    Qwen2VLForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer,
    CLIPModel, CLIPProcessor
)

CSV_PATH = "/home/duomeitinrfx/data/PrideMM/PrideMM.csv"
IMG_DIR = "/home/duomeitinrfx/data/PrideMM/Images"
OUT_DIR = "/home/duomeitinrfx/users/yunhe/reproduce/GECO/hidden_states_Pridemm"

DEVICE = "cuda:0"
NUM_CHUNKS = 1
CHUNK_IDX = 0

def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()

def extract_clip():
    print(f">>> [DEVICE: {DEVICE}] Extracting CLIP ViT-B/32 features...")
    df = pd.read_csv(CSV_PATH)
    os.makedirs(os.path.join(OUT_DIR, "clip_image"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "clip_text"), exist_ok=True)
    
    unprocessed_rows = []
    for idx, row in df.iterrows():
        if NUM_CHUNKS > 1 and idx % NUM_CHUNKS != CHUNK_IDX:
            continue
        base_name, _ = os.path.splitext(row["name"])
        img_out = os.path.join(OUT_DIR, "clip_image", f"{base_name}.pt")
        txt_out = os.path.join(OUT_DIR, "clip_text", f"{base_name}.pt")
        if not (os.path.exists(img_out) and os.path.exists(txt_out)):
            unprocessed_rows.append(row)
            
    if not unprocessed_rows:
        print("🎉 CLIP features already fully extracted! Skipping...")
        return
        
    print(f"Pending CLIP samples count: {len(unprocessed_rows)} / {len(df)}")
    clip_path = "/home/duomeitinrfx/data/models/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(clip_path).to(DEVICE).eval()
    processor = CLIPProcessor.from_pretrained(clip_path)
    
    with torch.no_grad():
        for row in tqdm(unprocessed_rows):
            base_name, _ = os.path.splitext(row["name"])
            
            # 1. Image sequence extraction
            img_path = os.path.join(IMG_DIR, row["name"])
            img = Image.open(img_path).convert("RGB")
            inputs_img = processor(images=img, return_tensors="pt").to(DEVICE)
            vision_outputs = model.vision_model(pixel_values=inputs_img.pixel_values)
            img_seq = model.visual_projection(vision_outputs.last_hidden_state).squeeze(0) # [seq_len, 768]
            torch.save(img_seq.cpu(), os.path.join(OUT_DIR, "clip_image", f"{base_name}.pt"))
            
            # 2. Text sequence extraction
            inputs_txt = processor(text=str(row["text"]), return_tensors="pt", padding=True, truncation=True, max_length=77).to(DEVICE)
            text_outputs = model.text_model(input_ids=inputs_txt.input_ids, attention_mask=inputs_txt.attention_mask)
            txt_seq = model.text_projection(text_outputs.last_hidden_state).squeeze(0) # [seq_len, 768]
            torch.save(txt_seq.cpu(), os.path.join(OUT_DIR, "clip_text", f"{base_name}.pt"))
            
    del model, processor
    clear_cache()
    print("🎉 CLIP extraction finished!")

def extract_llava():
    print(f">>> [DEVICE: {DEVICE}] Extracting LLaVA-v1.5-13B features...")
    df = pd.read_csv(CSV_PATH)
    os.makedirs(os.path.join(OUT_DIR, "llava"), exist_ok=True)
    
    unprocessed_rows = []
    for idx, row in df.iterrows():
        if NUM_CHUNKS > 1 and idx % NUM_CHUNKS != CHUNK_IDX:
            continue
        base_name, _ = os.path.splitext(row["name"])
        out_path = os.path.join(OUT_DIR, "llava", f"{base_name}.pt")
        if not os.path.exists(out_path):
            unprocessed_rows.append(row)
            
    if not unprocessed_rows:
        print("🎉 LLaVA features already fully extracted! Skipping...")
        return
        
    print(f"Pending LLaVA samples count: {len(unprocessed_rows)} / {len(df)}")
    llava_path = "/home/duomeitinrfx/data/models/llava-v1.5-13b"
    model = LlavaForConditionalGeneration.from_pretrained(
        llava_path, load_in_4bit=True, device_map={"": DEVICE}
    ).eval()
    
    # 动态将模型期待的图像标志物 image_token_index 强制指定为负数 index (-200)
    model.config.image_token_index = -200
    
    # 动态注入安全 Embedding 前向猴子补丁，防御任何负数索引(-200)或大数越界(>=32000)触发的 GPU IndexSelect 断言崩溃！
    old_embed_fwd = model.get_input_embeddings().forward
    model.get_input_embeddings().forward = lambda x: old_embed_fwd(torch.where((x < 0) | (x >= 32000), 0, x))
    
    processor = AutoProcessor.from_pretrained(llava_path)
    
    prompt = "USER: <image>\nIs this meme hateful? Answer yes or no.\nASSISTANT:"
    
    with torch.no_grad():
        for row in tqdm(unprocessed_rows):
            base_name, _ = os.path.splitext(row["name"])
            img_path = os.path.join(IMG_DIR, row["name"])
            img = Image.open(img_path).convert("RGB")
            
            inputs = processor(text=prompt, images=img, return_tensors="pt").to(DEVICE)
            inputs['input_ids'][inputs['input_ids'] == 32000] = -200
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[-1][0, -1, :].cpu()
            torch.save(hidden, os.path.join(OUT_DIR, "llava", f"{base_name}.pt"))
            
    del model, processor
    clear_cache()
    print("🎉 LLaVA extraction finished!")

def extract_qwen_and_gemma_via_qwen():
    """
    通过 Qwen2-VL-7B 模型，同时并行提取：
    1. 多模态分支（qwen）：图像 + 问题 -> 3584维特征
    2. 纯文本 OCR 分支（gemma）：纯文本问题（作为纯文本OCR智能体） -> 3584维特征
    极大节约显存与大模型载入开销，绕过 python3.8 下 Gemma-3 兼容性限制！
    """
    print(f">>> [DEVICE: {DEVICE}] Extracting Qwen and Gemma (using Qwen2-VL Text encoder) features...")
    df = pd.read_csv(CSV_PATH)
    os.makedirs(os.path.join(OUT_DIR, "qwen"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "gemma"), exist_ok=True)
    
    unprocessed_rows = []
    for idx, row in df.iterrows():
        if NUM_CHUNKS > 1 and idx % NUM_CHUNKS != CHUNK_IDX:
            continue
        base_name, _ = os.path.splitext(row["name"])
        qwen_out = os.path.join(OUT_DIR, "qwen", f"{base_name}.pt")
        gemma_out = os.path.join(OUT_DIR, "gemma", f"{base_name}.pt")
        if not (os.path.exists(qwen_out) and os.path.exists(gemma_out)):
            unprocessed_rows.append(row)
            
    if not unprocessed_rows:
        print("🎉 Both Qwen and Gemma features already fully extracted! Skipping...")
        return
        
    print(f"Pending samples count for Qwen/Gemma: {len(unprocessed_rows)} / {len(df)}")
    qwen_path = "/home/duomeitinrfx/data/models/Qwen2-VL-7B-Instruct"
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        qwen_path, 
        load_in_4bit=True, 
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
        device_map={"": DEVICE}
    ).eval()
    processor = AutoProcessor.from_pretrained(qwen_path)
    
    # 纯文本 prompt，代替原先 the Gemma 纯文本推理智能体角色
    text_prompt_template = "Determine if this meme text is hateful: \"{}\". Answer yes or no."
    multimodal_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Is this meme hateful? Answer yes or no.<|im_end|>\n<|im_start|>assistant\n"
    
    with torch.no_grad():
        for row in tqdm(unprocessed_rows):
            base_name, _ = os.path.splitext(row["name"])
            
            # 1. 提取多模态分支（qwen）
            qwen_out_path = os.path.join(OUT_DIR, "qwen", f"{base_name}.pt")
            if not os.path.exists(qwen_out_path):
                img_path = os.path.join(IMG_DIR, row["name"])
                img = Image.open(img_path).convert("RGB")
                try:
                    inputs_mm = processor(text=[multimodal_prompt], images=[img], padding=True, return_tensors="pt").to(DEVICE)
                    outputs_mm = model(**inputs_mm, output_hidden_states=True)
                    hidden_mm = outputs_mm.hidden_states[-1][0, -1, :].cpu()
                    torch.save(hidden_mm, qwen_out_path)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"\n⚠️ Sample {row['name']} (size: {img.size}) triggered OOM with raw resolution! Retrying with thumbnail (1024, 1024)...")
                    resized_img = img.copy()
                    resized_img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                    inputs_mm = processor(text=[multimodal_prompt], images=[resized_img], padding=True, return_tensors="pt").to(DEVICE)
                    outputs_mm = model(**inputs_mm, output_hidden_states=True)
                    hidden_mm = outputs_mm.hidden_states[-1][0, -1, :].cpu()
                    torch.save(hidden_mm, qwen_out_path)
            
            # 2. 提取纯文本 OCR 分支（gemma，在此由 Qwen-Text 替代提取）
            gemma_out_path = os.path.join(OUT_DIR, "gemma", f"{base_name}.pt")
            if not os.path.exists(gemma_out_path):
                pure_text_prompt = text_prompt_template.format(str(row["text"]))
                inputs_txt = processor(text=[pure_text_prompt], images=None, padding=True, return_tensors="pt").to(DEVICE)
                outputs_txt = model(**inputs_txt, output_hidden_states=True)
                hidden_txt = outputs_txt.hidden_states[-1][0, -1, :].cpu()
                torch.save(hidden_txt, gemma_out_path)
            
    del model, processor
    clear_cache()
    print("🎉 Qwen and Gemma extractions finished!")

def extract_gemma3_native():
    """
    使用 deim 环境下的原生 Gemma-3 (gemma-3-4b-it) 提取 100% 官方原汁原味的纯文本隐藏状态特征。
    输出维度：2560 维 (与 train.py 中的 gemma_dim=2560 完美契合！)
    """
    print(f">>> [DEVICE: {DEVICE}] Extracting NATIVE Gemma-3 (gemma-3-4b-it) features...")
    df = pd.read_csv(CSV_PATH)
    gemma_dir = os.path.join(OUT_DIR, "gemma")
    os.makedirs(gemma_dir, exist_ok=True)
    
    unprocessed_rows = []
    for idx, row in df.iterrows():
        if NUM_CHUNKS > 1 and idx % NUM_CHUNKS != CHUNK_IDX:
            continue
        unprocessed_rows.append(row)
            
    print(f"Pending samples count for Native Gemma-3: {len(unprocessed_rows)} / {len(df)}")
    gemma_path = "/home/duomeitinrfx/data/models/gemma-3-4b-it"
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(gemma_path)
    model = AutoModelForCausalLM.from_pretrained(
        gemma_path,
        torch_dtype=torch.float16,
        device_map={"": DEVICE}
    ).eval()
    
    text_prompt_template = "Determine if this meme text is hateful: \"{}\". Answer yes or no."
    
    with torch.no_grad():
        for row in tqdm(unprocessed_rows):
            base_name, _ = os.path.splitext(row["name"])
            gemma_out_path = os.path.join(gemma_dir, f"{base_name}.pt")
            
            pure_text_prompt = text_prompt_template.format(str(row["text"]))
            inputs = tokenizer(pure_text_prompt, return_tensors="pt").to(DEVICE)
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[-1][0, -1, :].cpu()
            torch.save(hidden, gemma_out_path)
            
    del model, tokenizer
    clear_cache()
    print("🎉 Native Gemma-3 extraction finished!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["clip", "llava", "qwen", "gemma"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    args = parser.parse_args()
    
    DEVICE = args.device
    NUM_CHUNKS = args.num_chunks
    CHUNK_IDX = args.chunk_idx
    
    if args.model == "clip":
        extract_clip()
    elif args.model == "llava":
        extract_llava()
    elif args.model == "qwen":
        extract_qwen_and_gemma_via_qwen()
    elif args.model == "gemma":
        extract_gemma3_native()
