'''
    design by 田宇瀚 2025.12.02
'''
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel
import re

BASE_MODEL_PATH = "/home/a/work/Qwen3-VL-8B-Instruct"
LORA_PATH = "/home/a/work/Qwen3-VL-main/qwen-vl-finetune/qwenvl/train/output-lora-final"

IMAGE_PATHS = [
    "/home/a/work/outdataset/dc/负样本/70.png",
    "/home/a/work/outdataset/dl/负样本/70.png",
    "/home/a/work/outdataset/sb/负样本/70.png",
    "/home/a/work/outdataset/yx/负样本/70.png",
    "/home/a/work/outdataset/as/负样本/70.png",
    "/home/a/work/outdataset/au/负样本/70.png",
    "/home/a/work/outdataset/b/负样本/70.png",
    "/home/a/work/outdataset/cr/负样本/70.png",
    "/home/a/work/outdataset/cu/负样本/70.png",
    "/home/a/work/outdataset/hg/负样本/70.png",
    "/home/a/work/outdataset/mgo/负样本/70.png",
    "/home/a/work/outdataset/ni/负样本/70.png",
    "/home/a/work/outdataset/sb1/负样本/70.png",
    "/home/a/work/outdataset/v/负样本/70.png",
    "/home/a/work/outdataset/w/负样本/70.png"
]

PROMPT_TEXT = "请根据这些特征图预测中心潜力值。"

# ------------- Load model + LoRA -------------
processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()

# ------------- Load images -------------------
images = [Image.open(p).convert("RGB") for p in IMAGE_PATHS]

# ------------- Build message ------------------
contents = []
for img in images:
    contents.append({"type": "image", "image": img})
contents.append({"type": "text", "text": PROMPT_TEXT})

messages = [{"role": "user", "content": contents}]

# ------------- Apply chat template --------------
prompt_text = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=False
)

# ------------- Build inputs for model ----------
inputs = processor(
    text=prompt_text,
    images=images,
    return_tensors="pt"
).to(model.device)

# ------------- Generate ------------------------
with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=10)

output = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

print("\n===== RAW OUTPUT =====")
print(output)

# ---- Extract numeric value ----
match = re.search(r"[-+]?\d*\.\d+|\d+", output)
value = float(match.group()) if match else None

print("\n===== 预测潜力值 =====")
print(value)
print("==================\n")
