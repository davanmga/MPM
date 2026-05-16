import os
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel
from tqdm import tqdm
import re

# ========== 配置 ==========
BASE_MODEL_PATH = "/home/a/work/Qwen3-VL-8B-Instruct"
LORA_PATH = "/home/a/work/Qwen3-VL-main/qwen-vl-finetune/qwenvl/train/output-lora-final"

FEATURE_DIR = "/home/a/work/Qwen3-VL-main/qwen-vl-finetune/png"
FEATURE_NAMES = ["dc","dl","sb","yx","as","au","b","cr","cu","hg","mgo","ni","sb1","v","w"]

PATCH_SIZE = 121
STRIDE = 8
PRINT_INTERVAL = 20

PROMPT_TEXT = "请根据这些特征图预测中心潜力值。"


# ========== 加载特征图 ==========
def load_feature_images():
    imgs = []
    for name in FEATURE_NAMES:
        path = os.path.join(FEATURE_DIR, f"{name}.png")
        if not os.path.exists(path):
            raise FileNotFoundError(f"⚠️ 缺少特征图: {path}")
        imgs.append(Image.open(path).convert("RGB"))
    return imgs


def extract_patch(img, x, y, size=101):
    return img.crop((x, y, x + size, y + size))


# ========== 主流程 ==========
def main():

    print("🚀 正在加载模型...")
    processor = AutoProcessor.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True
    )

    base = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )

    model = PeftModel.from_pretrained(base, LORA_PATH).eval()
    print(f"✅ 模型加载完成，设备：{model.device}")

    # 加载特征图
    feature_imgs = load_feature_images()
    W, H = feature_imgs[0].size
    print(f"📷 成功加载 15 个特征图 → 尺寸：{W} × {H}")

    out_h = (H - PATCH_SIZE) // STRIDE + 1
    out_w = (W - PATCH_SIZE) // STRIDE + 1
    result_map = np.zeros((out_h, out_w), dtype=np.float32)

    print(f"🏁 输出潜力图尺寸：{out_w} × {out_h}\n")

    total = out_h * out_w
    pbar = tqdm(total=total, desc="预测中")

    # 滑窗预测
    count = 0
    for y in range(0, H - PATCH_SIZE + 1, STRIDE):
        for x in range(0, W - PATCH_SIZE + 1, STRIDE):

            patch_imgs = [extract_patch(img, x, y, PATCH_SIZE) for img in feature_imgs]

            contents = []
            for pimg in patch_imgs:
                contents.append({"type": "image", "image": pimg})
            contents.append({"type": "text", "text": PROMPT_TEXT})

            messages = [{"role": "user", "content": contents}]

            prompt_text = processor.apply_chat_template(
                messages,
                add_generation_prompt=True
            )

            inputs = processor(
                text=prompt_text,
                images=patch_imgs,
                return_tensors="pt"
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False
                )

            text = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
            pred = float(nums[0]) if nums else 0.0

            iy = y // STRIDE
            ix = x // STRIDE
            result_map[iy, ix] = pred

            count += 1
            if count % PRINT_INTERVAL == 0:
                print(f"Patch({x},{y}) → {pred:.3f}")

            pbar.update(1)

    pbar.close()

    # ========== 输出 3 种矩阵格式 ==========

    output_matrix = result_map.astype(np.float32)

    # 1. npy 文件
    np.save("potential_matrix.npy", output_matrix)
    print("📁 已保存：potential_matrix.npy（float32 原始矩阵）")

    # 2. CSV 文件
    np.savetxt("potential_matrix.csv", output_matrix, delimiter=",", fmt="%.6f")
    print("📁 已保存：potential_matrix.csv（逗号分隔 CSV）")

    # 3. txt 文件
    with open("potential_matrix.txt", "w") as f:
        for row in output_matrix:
            f.write(" ".join(f"{v:.6f}" for v in row) + "\n")
    print("📁 已保存：potential_matrix.txt（空格分隔纯文本）")

    # 控制台预览
    print("\n📌 部分矩阵预览（前 5×8）：")
    h, w = output_matrix.shape
    print(output_matrix[:min(5, h), :min(8, w)])

    print("\n🎉 所有矩阵输出格式生成完毕！")


if __name__ == "__main__":
    main()
