import os
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE

# 设置你的数据目录和输出文件路径
input_dir = '../result/features/dataset1/BJTU-leftaxlebox/before/'  # 这里改成你的目标目录
output_csv = '../result/features/dataset1/BJTU-leftaxlebox/before/tsne.csv'

# 收集所有csv中的数据
all_features = []
all_labels = []

for fname in os.listdir(input_dir):
    if fname.endswith('.csv'):
        fpath = os.path.join(input_dir, fname)
        try:
            data = pd.read_csv(fpath, header=None)
        except Exception as e:
            print(f"跳过文件 {fname}，原因：{e}")
            continue
        label = os.path.splitext(fname)[0]  # 去掉 .csv 后缀
        all_features.append(data.values)
        all_labels += [label] * len(data)

# 合并所有样本
X = np.vstack(all_features)
labels = np.array(all_labels)

# 用 t-SNE 进行降维
print("Running t-SNE...")
tsne = TSNE(n_components=2, random_state=42)
X_2d = tsne.fit_transform(X)

# 构建降维后的 DataFrame 并保存
df = pd.DataFrame({
    'label': labels,
    'dim1': X_2d[:, 0],
    'dim2': X_2d[:, 1]
})
df.to_csv(output_csv, index=False)
print(f"t-SNE 降维完成，保存至 {output_csv}")
