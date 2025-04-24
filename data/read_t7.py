import torch

# 指定 .t7 文件的路径
path = r'F:\Project\CZSL\code\Disentangling-before-Composing\Disentangling-before-Composing\dataset\German\metadata_compositional-split-natural.t7'

# 尝试加载 .t7 文件
try:
    data = torch.load(path)
    print(data)  # 打印加载的内容
except Exception as e:
    print(f"Error loading the .t7 file: {e}")
