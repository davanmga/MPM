import os
import json
import numpy as np
from PIL import Image

# =============================
# 配置区
# =============================
base_dir = r"/home/a/work/outdataset"

feature_dirs = ["dc", "dl", "sb", "yx", "as", "au", "b",
                "cr", "cu", "hg", "mgo", "ni", "sb1", "v", "w"]

POS_NUM = 40
NEG_NUM = 40

output_json = "train_dataset_multiimage.json"

# =============================
# TIF → PNG
# =============================
def tif_to_png(path):
    if not os.path.exists(path):
        return None

    png_path = path[:-4] + ".png"
    if os.path.exists(png_path):
        return png_path

    img = Image.open(path)

    if img.mode == "F":
        arr = np.array(img)
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min)
        else:
            arr = np.zeros_like(arr)
        arr = (arr * 255).astype("uint8")
        img = Image.fromarray(arr, mode="L")
    else:
        img = img.convert("RGB")

    img.save(png_path)
    return png_path

# =============================
# 旋转增强
# =============================
def augment_image(img_path):
    png = tif_to_png(img_path)
    if png is None:
        return

    img = Image.open(png)
    base = png[:-4]
    for a in [90, 180, 270]:
        out = f"{base}_r{a}.png"
        if not os.path.exists(out):
            img.rotate(a, expand=True).save(out)

# =============================
# 构造单条 JSON
# =============================
def build_item(image_paths, base_label: float):
    """
    base_label: 1.0 (正样本) 或 0.0 (负样本)
    """

    # ======= ★ 最优连续标签策略 =======
    if base_label == 1.0:
        noisy = np.random.uniform(0.85, 1.00)
    else:
        noisy = np.random.uniform(0.00, 0.15)
    # =================================

    label_str = f"{noisy:.4f}"

    human_value = "\n".join(["<image>" for _ in image_paths])
    human_value += "\n请根据这些特征图预测中心潜力值。"

    return {
        "image": image_paths,
        "conversations": [
            {"from": "human", "value": human_value},
            {"from": "gpt", "value": label_str},
        ],
    }

# =============================
# 主流程
# =============================
all_items = []

print("🔄 正在转换 PNG + 旋转增强...")
for feat in feature_dirs:
    for i in range(POS_NUM):
        augment_image(os.path.join(base_dir, feat, "正样本", f"{i}.tif"))
    for i in range(40, 40 + NEG_NUM):
        augment_image(os.path.join(base_dir, feat, "负样本", f"{i}.tif"))

print("📝 正在生成 JSON 数据...")

for i in range(POS_NUM):
    for v in [f"{i}.png", f"{i}_r90.png", f"{i}_r180.png", f"{i}_r270.png"]:
        paths = [os.path.join(base_dir, feat, "正样本", v) for feat in feature_dirs]
        all_items.append(build_item(paths, 1.0))

for i in range(40, 40 + NEG_NUM):
    for v in [f"{i}.png", f"{i}_r90.png", f"{i}_r180.png", f"{i}_r270.png"]:
        paths = [os.path.join(base_dir, feat, "负样本", v) for feat in feature_dirs]
        all_items.append(build_item(paths, 0.0))

with open(output_json, "w", encoding="utf-8") as f:
    json.dump(all_items, f, ensure_ascii=False, indent=2)

print(f"🎉 数据集生成完成: {output_json}")
print(f"📚 总样本数: {len(all_items)}")
