import os
import torch
from os.path import join as ospj

split_name = "compositional-split-natural"

# 数据集根目录
DATASET_ROOT = "F:/Project/CZSL/code/Disentangling-before-Composing/Disentangling-before-Composing/dataset/German"
IMAGE_ROOT = ospj(DATASET_ROOT, "images")
SPLIT_FOLDER = ospj(DATASET_ROOT, split_name)
OUTPUT_FILE = ospj(DATASET_ROOT, f"metadata_{split_name}.t7")


# 解析 train/val/test_pairs.txt
def parse_pairs(file_path):
    if not os.path.exists(file_path):
        print(f"警告: {file_path} 不存在，跳过。")
        return []

    with open(file_path, 'r') as f:
        lines = f.read().strip().split("\n")

    pairs = [tuple(line.split()) for line in lines]
    return pairs


# 遍历 images 目录，收集所有图片
def collect_images():
    image_dict = {}  # { (attr, obj): [image_path1, image_path2, ...] }

    if not os.path.exists(IMAGE_ROOT):
        print(f"错误: {IMAGE_ROOT} 不存在！")
        return {}

    for folder in os.listdir(IMAGE_ROOT):
        folder_path = ospj(IMAGE_ROOT, folder)
        if not os.path.isdir(folder_path):
            continue  # 跳过非文件夹项

        parts = folder.split("_")  # 解析文件夹名，例如 "red_car" -> ["red", "car"]
        if len(parts) < 2:
            print(f"警告: {folder} 目录名格式错误，跳过！")
            continue

        attr, obj = parts[0], parts[1]  # 取属性和对象
        image_dict.setdefault((attr, obj), [])

        for img_file in os.listdir(folder_path):
            img_path = ospj(folder, img_file)  # 仅存相对路径，如 "red_car/img_001.jpg"
            image_dict[(attr, obj)].append(img_path)

    print(f"成功收集 {sum(len(v) for v in image_dict.values())} 张图片！")
    return image_dict


# 生成 .t7 数据
def process_dataset():
    image_dict = collect_images()
    dataset = []

    for split in ["train", "val", "test"]:
        pairs_file = ospj(SPLIT_FOLDER, f"{split}_pairs.txt")
        pairs = parse_pairs(pairs_file)

        for (attr, obj), images in image_dict.items():
            if (attr, obj) in pairs:
                for image in images:
                    dataset.append({
                        "image": image,  # 相对路径
                        "attr": attr,
                        "obj": obj,
                        "set": split
                    })

    return dataset


# 保存 .t7
def save_metadata():
    metadata = process_dataset()

    if not metadata:
        print("错误: 没有找到任何匹配的图片，请检查数据集结构！")
        return

    torch.save(metadata, OUTPUT_FILE)
    print(f"成功生成 {OUTPUT_FILE}，共 {len(metadata)} 条数据！")


if __name__ == "__main__":
    save_metadata()
