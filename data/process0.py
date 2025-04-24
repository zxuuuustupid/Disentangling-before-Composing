import os
import torch
import random
from os.path import join as ospj
import numpy as np
from collections import defaultdict

# 数据集根目录
DATASET_ROOT = "dataset/German"
IMAGE_ROOT = ospj(DATASET_ROOT, "images")
SPLIT_FOLDER = ospj(DATASET_ROOT, "compositional-split-natural")
OUTPUT_FILE = ospj(DATASET_ROOT, "metadata_compositional-split-natural.t7")

def collect_images():
    """收集所有图片路径并按属性-对象对组织"""
    image_dict = defaultdict(list)
    
    # 遍历图片目录
    for root, _, files in os.walk(IMAGE_ROOT):
        for file in files:
            if file.endswith(('.jpg', '.png', '.jpeg')):
                # 从路径中提取属性和对象
                rel_path = os.path.relpath(root, IMAGE_ROOT)
                folder_name = os.path.basename(root)
                
                # 解析文件夹名，例如 "Load1_Health" -> ["Load1", "Health"]
                parts = folder_name.split("_")
                if len(parts) != 2:
                    print(f"警告: {folder_name} 目录名格式错误，跳过！")
                    continue
                
                condition, attr = parts[0], parts[1]  # 工况和属性
                image_dict[(condition, attr)].append(os.path.join(rel_path, file))
    
    return image_dict

def split_dataset(image_dict):
    """实现特定的数据集划分策略"""
    # 按属性和对象分类
    attr_obj_pairs = list(image_dict.keys())
    conditions = sorted(list(set(pair[0] for pair in attr_obj_pairs)))
    attrs = sorted(list(set(pair[1] for pair in attr_obj_pairs)))
    
    # 分离健康样本和故障样本
    healthy_pairs = [(condition, attr) for condition, attr in attr_obj_pairs if attr == 'Health']
    fault_pairs = [(condition, attr) for condition, attr in attr_obj_pairs if attr != 'Health']
    
    # 获取所有工况
    all_conditions = sorted(list(set(condition for condition, _ in attr_obj_pairs)))
    
    # 训练集：健康样本包含所有工况，故障样本随机选择一种工况
    train_pairs = []
    # 添加所有健康样本
    train_pairs.extend(healthy_pairs)
    # 为每个故障类型随机选择一种工况
    fault_types = sorted(list(set(attr for _, attr in fault_pairs)))
    for fault_type in fault_types:
        fault_conditions = [condition for condition, attr in fault_pairs if attr == fault_type]
        selected_condition = random.choice(fault_conditions)
        train_pairs.append((selected_condition, fault_type))
    
    # 验证集和测试集：包含所有属性-对象对
    val_pairs = attr_obj_pairs
    test_pairs = attr_obj_pairs
    
    return train_pairs, val_pairs, test_pairs

def save_split_files(train_pairs, val_pairs, test_pairs):
    """保存划分文件"""
    os.makedirs(SPLIT_FOLDER, exist_ok=True)
    
    # 保存训练集划分
    with open(ospj(SPLIT_FOLDER, 'train_pairs.txt'), 'w') as f:
        for attr, obj in train_pairs:
            f.write(f"{attr} {obj}\n")  # attr是工况，obj是健康状态
    
    # 保存验证集划分
    with open(ospj(SPLIT_FOLDER, 'val_pairs.txt'), 'w') as f:
        for attr, obj in val_pairs:
            f.write(f"{attr} {obj}\n")  # attr是工况，obj是健康状态
    
    # 保存测试集划分
    with open(ospj(SPLIT_FOLDER, 'test_pairs.txt'), 'w') as f:
        for attr, obj in test_pairs:
            f.write(f"{attr} {obj}\n")  # attr是工况，obj是健康状态

def process_dataset():
    """处理数据集并生成元数据"""
    image_dict = collect_images()
    train_pairs, val_pairs, test_pairs = split_dataset(image_dict)
    
    # 保存划分文件
    save_split_files(train_pairs, val_pairs, test_pairs)
    
    # 生成元数据
    dataset = []
    
    # 处理训练集
    for condition, attr in train_pairs:
        for image in image_dict[(condition, attr)]:
            dataset.append({
                "image": image,
                "attr": condition,  # attr对应工况
                "obj": attr,        # obj对应健康状态
                "set": "train"
            })
    
    # 处理验证集
    for condition, attr in val_pairs:
        for image in image_dict[(condition, attr)]:
            dataset.append({
                "image": image,
                "attr": condition,  # attr对应工况
                "obj": attr,        # obj对应健康状态
                "set": "val"
            })
    
    # 处理测试集
    for condition, attr in test_pairs:
        for image in image_dict[(condition, attr)]:
            dataset.append({
                "image": image,
                "attr": condition,  # attr对应工况
                "obj": attr,        # obj对应健康状态
                "set": "test"
            })
    
    return dataset

def save_metadata():
    """保存元数据文件"""
    metadata = process_dataset()
    
    if not metadata:
        print("错误: 没有找到任何匹配的图片，请检查数据集结构！")
        return
    
    torch.save(metadata, OUTPUT_FILE)
    print(f"成功生成 {OUTPUT_FILE}，共 {len(metadata)} 条数据！")
    
    # 打印数据集统计信息
    train_count = sum(1 for item in metadata if item['set'] == 'train')
    val_count = sum(1 for item in metadata if item['set'] == 'val')
    test_count = sum(1 for item in metadata if item['set'] == 'test')
    
    print(f"\n数据集统计信息:")
    print(f"训练集: {train_count} 条数据")
    print(f"验证集: {val_count} 条数据")
    print(f"测试集: {test_count} 条数据")

def get_split_info(self):
    data = torch.load(ospj(self.root, 'metadata_{}.t7'.format(self.split)))
    train_data, val_data, test_data = [], [], []

    for instance in data:
        image, attr, obj, settype = instance['image'], instance['attr'], \
            instance['obj'], instance['set']
        curr_data = [image, attr, obj]
        if attr == 'NA' or (attr, obj) not in self.pairs or settype == 'NA':
            continue
        if settype == 'train':
            train_data.append(curr_data)
        elif settype == 'val':
            val_data.append(curr_data)
        else:
            test_data.append(curr_data)

    return train_data, val_data, test_data

if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    save_metadata() 