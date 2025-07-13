import os
import shutil
import re

# 源路径和目标路径
source_root = r'F:\Project\TripletLoss\CWT-XJT\test\anomaly'
target_root = r'F:\Project\CZSL\code\Disentangling-before-Composing\Disentangling-before-Composing\dataset\SWJTU\images'

# 遍历 anomaly 目录下的所有以数字-开头的文件夹
for folder_name in os.listdir(source_root):
    if not re.match(r'^\d+-', folder_name):
        continue

    folder_path = os.path.join(source_root, folder_name)
    if not os.path.isdir(folder_path):
        continue

    # 提取数字编号 m，例如 "2-InnerWear" -> 2
    m_number = folder_name.split('-')[0]

    # 遍历 WCn 文件夹
    for wcn_name in os.listdir(folder_path):
        if not re.match(r'^WC\d+$', wcn_name):
            continue

        wc_path = os.path.join(folder_path, wcn_name)
        if not os.path.isdir(wc_path):
            continue

        # 找到 WCn 中的唯一子文件夹
        subfolders = [d for d in os.listdir(wc_path) if os.path.isdir(os.path.join(wc_path, d))]
        if len(subfolders) != 1:
            print(f"⚠️ {wc_path} 下不是唯一子文件夹，跳过")
            continue

        subfolder_path = os.path.join(wc_path, subfolders[0])

        # 提取 n 号
        n_number = wcn_name.replace('WC', '')

        # 目标文件夹名称
        dest_folder_name = f'WC{n_number}_M{m_number}'
        dest_folder_path = os.path.join(target_root, dest_folder_name)

        os.makedirs(dest_folder_path, exist_ok=True)

        # 复制所有图片
        for file in os.listdir(subfolder_path):
            src_file = os.path.join(subfolder_path, file)
            dst_file = os.path.join(dest_folder_path, file)
            if os.path.isfile(src_file):
                shutil.copy2(src_file, dst_file)

        print(f"✅ 已复制 {subfolder_path} 中的图片到 {dest_folder_path}")
