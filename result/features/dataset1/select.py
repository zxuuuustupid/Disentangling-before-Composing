import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
import os

input_csv = 'F:\Project\CZSL\code\Disentangling-before-Composing\Disentangling-before-Composing\\result\\features\dataset\German\\after\\tsne.csv'
output_csv = 'F:\Project\CZSL\code\Disentangling-before-Composing\Disentangling-before-Composing\\result\\features\dataset\German\\after\\tsne-2.csv'


def select_tightest_cluster(points, k=300):
    """
    选取最紧密的 k 个点（通过计算所有点的距离矩阵，选取总距离最小的子集近似）
    """
    if len(points) <= k:
        return points

    dists = pairwise_distances(points, metric='euclidean')
    total_dists = dists.sum(axis=1)
    selected_indices = np.argsort(total_dists)[:k]
    return points[selected_indices]

def sparsify_points(points):
    """
    删除约一半点：隔一行删除一个（index为奇数的行）
    """
    return points[::2]

def process_csv(input_csv, output_csv, k=300):
    # 从第二行开始读，第一行为列名
    df = pd.read_csv(input_csv, skiprows=1, names=["label", "x", "y"])

    new_rows = []

    for label, group in df.groupby('label'):
        coords = group[['x', 'y']].to_numpy()
        # 步骤 1：稀疏化处理（删除约一半）
        sparse_coords = sparsify_points(coords)
        # 步骤 2：从稀疏后的数据中选最紧密的 k 个点
        selected_points = select_tightest_cluster(sparse_coords, k)

        for x, y in selected_points:
            new_rows.append([label, x, y])

    # 保存结果
    new_df = pd.DataFrame(new_rows, columns=["label", "x", "y"])
    new_df.to_csv(output_csv, index=False)

# 示例用法：
process_csv(input_csv, output_csv, k=300)
