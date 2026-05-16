'''
    design by 田宇瀚 2025.12.02
'''
import json

train_data = json.load(open("/home/a/work/outdataset/train.json"))
val_data = []

for item in train_data:
    images = item["image"]
    # 选择包含 _r180 的作为验证集
    if any("_r180" in p for p in images):
        val_data.append(item)

json.dump(val_data, open("val.json", "w", encoding="utf8"), indent=2, ensure_ascii=False)

print("验证集大小：", len(val_data))