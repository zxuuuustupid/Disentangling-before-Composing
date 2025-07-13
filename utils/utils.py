import os
from os.path import join as ospj
import shutil
import sys
import yaml
import PIL
import numpy as np
from  matplotlib import pyplot as plt
import skimage, skimage.transform

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def save_args(args, log_path, argfile):
    shutil.copy('train.py', log_path)
    modelfiles = ospj(log_path, 'models')
    try:
        shutil.copy(argfile, log_path)
    except:
        print('Config exists')
    try:
        shutil.copytree('models/', modelfiles)
    except:
        print('Already exists')
    with open(ospj(log_path,'args_all.yaml'),'w') as f:
        yaml.dump(args, f, default_flow_style=False, allow_unicode=True)
    with open(ospj(log_path, 'args.txt'), 'w') as f:
        f.write('\n'.join(sys.argv[1:]))

# def load_args(filename, args):
#     with open(filename, 'r') as stream:
#         data_loaded = yaml.safe_load(stream)
#     for key, group in data_loaded.items():
#         for key, val in group.items():
#             setattr(args, key, val)

import yaml

def load_args(config_file, args):
    with open(config_file, 'r') as f:
        cfg = yaml.safe_load(f)

    def flatten(d, parent_key=''):
        items = []
        for k, v in d.items():
            new_key = f'{parent_key}.{k}' if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)

    flat_cfg = flatten(cfg)

    # 最后一行加上，把扁平结果直接更新 args 的属性
    for key, val in flat_cfg.items():
        setattr(args, key.replace('.', '_'), val)

    # 补上兼容旧逻辑：如果存在 dataset_data_dir 就设置 data_dir
    if hasattr(args, 'dataset_data_dir'):
        args.data_dir = getattr(args, 'dataset_data_dir')
    if hasattr(args, 'dataset_dataset'):
        args.dataset = getattr(args, 'dataset_dataset')
    if hasattr(args, 'experiment_name'):
        args.name = getattr(args, 'experiment_name')


def imresize(img, height=None, width=None):
    # load image
    if height is not None and width is not None:
        ny = height
        nx = width
    elif height is not None:
        ny = height
        nx = img.shape[1] * ny / img.shape[0]
    elif width is not None:
        nx = width
        ny = img.shape[0] * nx / img.shape[1]
    else:
        ny = img.shape[0]
        nx = img.shape[1]

    return skimage.transform.resize(img, (int(ny), int(nx)), mode='constant')


# Heat map visualization
def show_heatmaps(imgs, masks, K, enhance=1, title=None, cmap='gist_rainbow'):

    if K > 0:
        _cmap = plt.cm.get_cmap(cmap)
        colors = [np.array(_cmap(i)[:3]) for i in np.arange(0,1,1/K)]
    plt.figure(figsize=(4 * len(imgs), 4))
    if title is not None:
        plt.suptitle(title+'\n', fontsize=24).set_y(1.05)
    for i in range(len(imgs)):
        plt.subplot(1, len(imgs), i + 1)

        img = imgs[i]
        if img.max()<=1:
            img *= 255
        img = np.array(PIL.ImageEnhance.Color(PIL.Image.fromarray(np.uint8(img))).enhance(enhance))
        plt.imshow(img)
        plt.axis('off')
        for k in range(K):
            layer = np.ones((*img.shape[:2],4))
            for c in range(3): layer[:,:,c] *= colors[k][c]
            mask = masks[i][k]
            layer[:,:,3] = mask
            plt.imshow(layer)
            plt.axis('off')

    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.show()

    plt.savefig('ff.png') 