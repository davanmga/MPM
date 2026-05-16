'''
    design by 田宇瀚 2025.12.02
'''
import os
import json
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel
from tqdm import tqdm

# ======================
# 路径配置
# ======================
BASE_MODEL_PATH = "/home/a/work/Qwen3-VL-8B-Instruct"
LORA_PATH = "/home/a/work/Qwen3-VL-main/qwen-vl-finetune/qwenvl/train/output-lora-final"
VAL_JSON = "/home/a/work/outdataset/val.json"

print("Loading base model...")
base = Qwen3VLForConditionalGeneration.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
print("Loading LoRA...")
model = PeftModel.from_pretrained(base, LORA_PATH)
model.eval()

processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH)

print("Loading dataset...")
dataset = json.load(open(VAL_JSON, "r", encoding="utf-8"))

correct = 0
total = 0

for item in tqdm(dataset):
    img_paths = item["image"]
    label = item["conversations"][1]["value"].strip()

    # 1. 加载全部图片
    images = [Image.open(p).convert("RGB") for p in img_paths]

    # 2. 构建 messages（必须与 test.py 一模一样）
    contents = []
    for img in images:
        contents.append({
            "type": "image",
            "image": img
        })
    contents.append({
        "type": "text",
        "text": "请根据这些特征图预测中心潜力值。"
    })

    messages = [{"role": "user", "content": contents}]

    # 3. 使用 apply_chat_template（这一行最关键）
    prompt_text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True
    )

    # 4. 编码
    inputs = processor(
        text=prompt_text,
        images=images,
        return_tensors="pt"
    ).to(model.device)

    # 5. 生成
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=10)

    output = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

    # 简单判断：包含 1 就预测 1
    pred = "1" if "1" in output else "0"

    if pred == label:
        correct += 1
    total += 1

print(f"\nAccuracy = {correct}/{total} = {correct/total:.4f}\n")