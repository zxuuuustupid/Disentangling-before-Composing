import sys
import os
import csv

# from PyQt5.QtCore.QProcess import state

# sys.argv += ['--config', 'configs/BJTU-leftaxlebox.yml']
# sys.argv += ['--config', 'configs/BJTU-gearbox.yml']
# sys.argv += ['--config', 'configs/BJTU-motor.yml']
# sys.argv += ['--config', 'configs/German.yml']
# sys.argv += ['--config', 'configs/Canada.yml']
# sys.argv += ['--config', 'configs/SWJTU.yml']

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.dbc import DBC

from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

import tqdm
from tqdm import tqdm
import os

from os.path import join as ospj
import csv

from data import dataset as dset
from utils.common import Evaluator
from utils.utils import save_args, load_args
from flags import parser, DATA_FOLDER

def get_task_id(splitname):
    # 从 'compositional-split-natural-4' 中提取 4
    if 'compositional-split-natural-' in splitname:
        return int(splitname.split('-')[-1])
    else:
        raise ValueError(f"无法从 splitname '{splitname}' 中提取 task id")

def get_dataset_name_from_experiment_name(name):
    parts = name.split('/')
    if len(parts) >= 2:
        return parts[1]
    else:
        raise ValueError(f"无法从 name '{name}' 中提取数据集名称")


torch.multiprocessing.set_sharing_strategy('file_system')

best_auc = 0
best_hm = 0
best_unseen = 0
best_obj = 0
compose_switch = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def freeze(m):
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
        p.grad = None


def main():
    args = parser.parse_args()
    load_args(args.config, args)
    logpath = os.path.join(args.cv_dir, args.name)
    os.makedirs(logpath, exist_ok=True)
    save_args(args, logpath, args.config)
    writer = SummaryWriter(log_dir=logpath, flush_secs=30)

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    trainset = dset.CompositionDataset(
        root=os.path.join(DATA_FOLDER, args.data_dir).replace('\\', '/'),
        phase='train',
        split=args.splitname
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers)

    testset = dset.CompositionDataset(
        root=os.path.join(DATA_FOLDER, args.data_dir).replace('\\', '/'),
        phase=args.test_set,
        split=args.splitname
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.workers)

    model = DBC(trainset, args)
    model = model.to(device)
    freeze(model.feat_extractor)

    model_params = [param for name, param in model.named_parameters() if param.requires_grad]
    optim_params = [{'params': model_params}]
    optimizer = optim.Adam(optim_params, lr=args.lr, weight_decay=args.wd)

    train = train_normal
    evaluator_val = Evaluator(testset, model)

    start_epoch = 0
    if args.load is not None:
        checkpoint = torch.load(args.load)
        model.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        print('Loaded model from ', args.load)

    for epoch in tqdm(range(start_epoch, args.max_epochs + 1), desc='Current epoch', dynamic_ncols=True):
        # print("====================",epoch,args.max_epochs + 1)
        train(epoch, model, trainloader, optimizer, writer)
        if epoch % args.eval_val_every == 0:
            with torch.no_grad():
                test(epoch, model, testloader, evaluator_val, writer, args, logpath)
    print('Best AUC achieved is ', best_auc)
    print('Best HM achieved is ', best_hm)


def train_normal(epoch, model, trainloader, optimizer, writer):
    model.train()
    freeze(model.feat_extractor)

    train_loss = 0.0

    for idx, data in tqdm(enumerate(trainloader), total=len(trainloader), desc='Training', dynamic_ncols=True):
        data = [d.to(device) for d in data]

        loss, _ = model(data, epoch)

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        train_loss += loss.item()

    train_loss = train_loss / len(trainloader)
    writer.add_scalar('Loss/train_total', train_loss, epoch)

    print('Epoch: {}| Loss: {}'.format(epoch, round(train_loss, 2)))


def test(epoch, model, testloader, evaluator, writer, args, logpath):
    global best_auc, best_hm, best_unseen, best_obj

    def save_checkpoint(filename):
        state = {
            'net': model.state_dict(),
            'epoch': epoch,
            'AUC': stats['AUC']
        }
        torch.save(state, os.path.join(logpath, 'ckpt_{}.t7'.format(filename)))

    model.eval()

    all_attr_gt, all_obj_gt, all_pair_gt, all_pred = [], [], [], []

    for idx, data in tqdm(enumerate(testloader), total=len(testloader), desc='Testing', dynamic_ncols=True):
        # print(data)

        data = [d.to(device) for d in data]

        _, predictions = model(data, epoch)

        attr_truth, obj_truth, pair_truth = data[1], data[2], data[3]

        all_pred.append(predictions)
        all_attr_gt.append(attr_truth)
        all_obj_gt.append(obj_truth)
        all_pair_gt.append(pair_truth)

        # print(predictions)
        # print(attr_truth)
        # print(obj_truth)
        # device = 'cuda:0'), ('Load1', 'IR'): tensor([0.0020, 0.0039, 0.0002, 0.0148, 0.0041, 0.0056, 0.0552, 0.0012],device='cuda:0'),
        # ('Load1', 'OR'): tensor([0.2705, 0.2714, 0.2227, 0.2198, 0.1923, 0.2028, 0.1568, 0.2733],device='cuda:0'),
        # ('Load2', 'Health'): tensor([0.0034, 0.0015, 0.0005, 0.0023, 0.0085, 0.0027, 0.0030, 0.0018],device='cuda:0'),
        # ('Load2', 'IR'): tensor([0.0017, 0.0034, 0.0003, 0.0160, 0.0046, 0.0073, 0.0701, 0.0014],device='cuda:0'),
        # ('Load2', 'OR'): tensor([0.2282, 0.2354, 0.2965, 0.2382, 0.2140, 0.2664, 0.1991, 0.3028],device='cuda:0'),
        # ('Load3', 'Health'): tensor([0.0021, 0.0009, 0.0001, 0.0014, 0.0068, 0.0017, 0.0021, 0.0007],device='cuda:0'),
        # ('Load3', 'IR'): tensor([1.0318e-03, 2.0034e-03, 7.8168e-05, 9.8673e-03, 3.6392e-03, 4.5878e-03, 4.7808e-02, 5.6546e-04], device='cuda:0'),
        # ('Load3', 'OR'): tensor([0.1393, 0.1397, 0.0774, 0.1465, 0.1700, 0.1671, 0.1358, 0.1246],device='cuda:0'),
        # ('Load4', 'Health'): tensor([0.0050, 0.0022, 0.0006, 0.0031, 0.0146, 0.0033, 0.0036, 0.0017],device='cuda:0'),
        # ('Load4', 'IR'): tensor([0.0025, 0.0048, 0.0004, 0.0218, 0.0078, 0.0090, 0.0844, 0.0013],device='cuda:0'),
        # ('Load4', 'OR'): tensor([0.3402, 0.3332, 0.4008, 0.3241, 0.3659, 0.3276, 0.2397, 0.2890],device='cuda:0')}
        # tensor([3, 3, 3, 3, 3, 3, 3, 3], device='cuda:0')
        # tensor([2, 2, 2, 2, 2, 2, 2, 2], device='cuda:0')

    if args.cpu_eval:
        all_attr_gt, all_obj_gt, all_pair_gt = torch.cat(all_attr_gt), torch.cat(all_obj_gt), torch.cat(all_pair_gt)
    else:
        all_attr_gt, all_obj_gt, all_pair_gt = torch.cat(all_attr_gt).to('cpu'), torch.cat(all_obj_gt).to(
            'cpu'), torch.cat(all_pair_gt).to('cpu')

    # print(all_attr_gt.shape, all_obj_gt.shape, all_pair_gt.shape) [12000]

    all_pred_dict = {}
    if args.cpu_eval:
        for k in all_pred[0].keys():
            all_pred_dict[k] = torch.cat(
                [all_pred[i][k].to('cpu') for i in range(len(all_pred))])

    else:
        for k in all_pred[0].keys():
            # print(all_pred[0][k])
            # tensor([0.0316, 0.0102, 0.0155, 0.0316, 0.0159, 0.0046, 0.0142, 0.0628],
            #        device='cuda:0')

            all_pred_dict[k] = torch.cat(
                [all_pred[i][k] for i in range(len(all_pred))])
            # print(all_pred_dict[k].shape)[12000]

    results = evaluator.score_model(all_pred_dict, all_obj_gt, bias=args.bias, topk=args.topk)
    stats, acc_array, confusion_accuracy = evaluator.evaluate_predictions(results, all_attr_gt, all_obj_gt, all_pair_gt, all_pred_dict, topk=args.topk)
    # print(stats.keys())
    # dict_keys(['obj_oracle_match', 'obj_oracle_match_unbiased', 'closed_attr_match', 'closed_obj_match', 'closed_m
    #            atch', 'closed_seen_match', 'closed_unseen_match', 'closed_ca', 'closed_seen_ca', 'closed_unseen_ca', '
    #            closed_
    #            ub_attr_match', 'closed_ub_obj_match', 'closed_ub_match', 'closed_ub_seen_match', 'closed_ub_unseen_match
    #            ', 'closed_ub_ca', 'closed_ub_seen_ca', 'closed_ub_unseen_ca', 'best_unseen', 'AUC'])


    stats['a_epoch'] = epoch

    result = ''
    # print(stats['closed_seen_match'])
    for key in ['closed_obj_match','closed_seen_obj_match','closed_unseen_obj_match']:
        writer.add_scalar(key, stats[key], epoch)
        result = result + key + '  ' + str(round(stats[key], 4)) + '| '

    print(f'Test Epoch: {epoch}')
    print(result)

    if epoch > 0 and epoch % args.save_every == 0:
        save_checkpoint(epoch)

    if epoch == args.max_epochs:
        # 构造结果行
        final_result_row = {
            'experiment': get_dataset_name_from_experiment_name(args.name)+str(get_dataset_name_from_experiment_name(args.name)),
            'closed_obj_match': round(stats['closed_obj_match'], 4),
            'closed_seen_obj_match': round(stats['closed_seen_obj_match'], 4),
            'closed_unseen_obj_match': round(stats['closed_unseen_obj_match'], 4)
        }

        # 创建 result 目录
        result_dir = os.path.join(os.getcwd(), 'result')
        os.makedirs(result_dir, exist_ok=True)
        result_csv = os.path.join(result_dir, 'res.csv')

        # 写入 CSV（追加或创建）
        write_header = not os.path.exists(result_csv)
        with open(result_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=final_result_row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(final_result_row)

        import pandas as pd

        # 1. 保存 每个 attribute 下的 object 准确率矩阵
        dataset_name = get_dataset_name_from_experiment_name(args.name)
        obj_acc_dir = os.path.join('result', 'obj_acc_per_attr', dataset_name)
        os.makedirs(obj_acc_dir, exist_ok=True)

        # print("acc_array.shape:", acc_array.shape)
        # print("objs:", testloader.dataset.objs)

        # acc_df = pd.DataFrame(acc_array, columns=testloader.dataset.objs)

        acc_df = pd.DataFrame(acc_array, columns=["attr_id", "seen_obj_acc", "unseen_obj_acc", "total_obj_acc"])

        # acc_array: [#attr, #obj]
        acc_df.index = testloader.dataset.attrs
        # acc_df.to_csv(os.path.join(obj_acc_dir, f'task-{get_task_id(args.splitname)}.csv'))

        # 2. 保存 每个 attribute 下的 object 混淆矩阵（每个 attr 一个 csv）
        confusion_dir = os.path.join('result', 'confusion_max', dataset_name, f'task-{get_task_id(args.splitname)}')
        os.makedirs(confusion_dir, exist_ok=True)

        for attr_idx, matrix in confusion_accuracy.items():  # ✅ 正确使用字典
            attr_name = testloader.dataset.attrs[attr_idx]  # ⬅️ 根据 index 找属性名
            matrix_df = pd.DataFrame(matrix, index=testloader.dataset.objs, columns=testloader.dataset.objs)
            # matrix_df.to_csv(os.path.join(confusion_dir, f'{attr_name}.csv'))

    # if stats['AUC'] > best_auc:
    #     best_auc = stats['AUC']
    #     print('New best AUC ', best_auc)
    #     save_checkpoint('best_auc')

    # if stats['best_hm'] > best_hm:
    #     best_hm = stats['best_hm']
    #     print('New best HM ', best_hm)
    #     save_checkpoint('best_hm')

    # if stats['best_unseen'] > best_unseen:
    #     best_unseen = stats['best_unseen']
    #     print('New best_unseen ', best_unseen)
    #     save_checkpoint('best_unseen')

    if stats['closed_obj_match'] > best_obj:
        best_obj = stats['closed_obj_match']
        print('New best_obj ', best_obj)
        save_checkpoint('best_obj')

    with open(ospj(logpath, 'logs.csv'), 'a') as f:
        w = csv.DictWriter(f, stats.keys())
        if epoch == 0:
            w.writeheader()
        w.writerow(stats)


# if __name__ == '__main__':
#     config_list = [
#         'configs/German.yml',
#         'configs/BJTU-leftaxlebox.yml',
#         'configs/BJTU-gearbox.yml',
#         'configs/BJTU-motor.yml',
#         'configs/Canada.yml',
#         'configs/SWJTU.yml',
#         'configs/SF-ship.yml'
#     ]
#
#     # 每个数据集对应的 split ID 列表
#     split_id_map = {
#         'BJTU':   [8, 12, 20, 28],
#         'German': [3, 6, 9],
#         'SWJTU':  [6, 12, 18],
#         'Canada': [5, 10, 15]
#     }
#
#     for config_file in config_list:
#         # 在 split_id_map 中找到对应的 split ids
#         matched = False
#         for key in split_id_map:
#             if key in config_file:
#                 split_ids = split_id_map[key]
#                 matched = True
#                 break
#         if not matched:
#             print(f'❌ 配置文件 {config_file} 没有匹配到任何 split_id，请检查文件名是否包含 BJTU、German、SWJTU 或 Canada')
#             sys.exit(1)
#
#         for split_id in split_ids:
#             # 设置 sys.argv 以模拟命令行参数传入
#             sys.argv = [sys.argv[0], '--config', config_file]
#
#             args = parser.parse_args()
#             load_args(config_file, args)
#
#             # 修改 splitname 和实验名
#             args.splitname = f'compositional-split-natural-{split_id}'
#             args.name = f'{args.name}_s{split_id}'
#             setattr(args, "splitname", f'compositional-split-natural-{split_id}')
#
#             print(f'\n================ Running config: {config_file} | split: {args.splitname} ================\n')
#
#             try:
#                 main()
#             except KeyboardInterrupt:
#                 print('Early stopped for', config_file)
#             except Exception as e:
#                 print(f'Error running config {config_file}, split {split_id}: {e}')

import yaml

def modify_yaml_splitname(path, new_splitname):
    """直接修改 yml 文件中的 splitname 字段"""
    import yaml
    with open(path, 'r') as f:
        config_data = yaml.safe_load(f)

    config_data['splitname'] = new_splitname

    with open(path, 'w') as f:
        yaml.dump(config_data, f, default_flow_style=False)  # ✅ 保持正常 YAML 格式



if __name__ == '__main__':
    config_list = [
        'configs/German.yml',
        'configs/BJTU-leftaxlebox.yml',
        'configs/BJTU-gearbox.yml',
        'configs/BJTU-motor.yml',
        'configs/Canada.yml',
        'configs/SWJTU.yml',
    ]

    split_id_map = {
        'BJTU':   [8, 12, 20, 28],
        'German': [3, 6, 9],
        'SWJTU':  [6, 12, 18],
        'Canada': [5, 10, 15]
    }

    for config_file in config_list:
        matched = False
        for key in split_id_map:
            if key in config_file:
                split_ids = split_id_map[key]
                matched = True
                break
        if not matched:
            print(f'❌ 配置文件 {config_file} 没有匹配到任何 split_id，请检查文件名是否包含 BJTU、German、SWJTU 或 Canada')
            sys.exit(1)

        for split_id in split_ids:
            splitname = f'compositional-split-natural-{split_id}'

            # ✅ 修改 yml 中的 splitname 字

            # 设置命令行参数
            sys.argv = [sys.argv[0], '--config', config_file]
            args = parser.parse_args()
            load_args(config_file, args)

            # ✅ 自动加上 split ID 后缀的实验名
            args.name = f'{args.name}_s{split_id}'
            modify_yaml_splitname(config_file, splitname)

            print(f'\n================ Running config: {config_file} | split: {splitname} ================\n')

            try:
                # print('当前使用的数据目录:', args.data_dir)

                main()
            except KeyboardInterrupt:
                print('Early stopped for', config_file)
