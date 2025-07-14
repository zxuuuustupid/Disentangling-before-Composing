import sys
import torch
import os
import csv
import itertools
import random
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.dbc1 import DBC
from data import dataset as dset
from utils.common import Evaluator, device
from utils.utils import load_args
from flags import parser, DATA_FOLDER

# 保持 config 不变
sys.argv += ['--config', 'configs/BJTU-motor.yml']

TRIPLET_OUTPUT = './result/triplet/metrics.csv'
MAX_EPOCHS = 3
TOTAL_COMBOS = 50
TOTAL_WEIGHT = 1.5

os.makedirs(os.path.dirname(TRIPLET_OUTPUT), exist_ok=True)
device= 'cuda' if torch.cuda.is_available() else 'cpu'


# 生成 50 个满足 x+y+z = 1.5 的 (rec, rep, grad) 组合
def generate_triplets(n=50, total=1.5, step=0.05):
    values = [round(i * step, 2) for i in range(int(total / step) + 1)]
    candidates = [(x, y, total - x - y) for x in values for y in values if 0 <= total - x - y <= total]
    filtered = [t for t in candidates if t[2] >= 0]
    random.shuffle(filtered)
    return filtered[:n]

def freeze(m):
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
        p.grad = None

def train(epoch, model, trainloader, optimizer, device):
    model.train()
    total_loss = 0.0
    for idx, data in tqdm(enumerate(trainloader), total=len(trainloader), desc=f'Training Epoch {epoch}'):
        data = [d.to(device) for d in data]
        loss, _ = model(data, epoch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch} Train Loss: {total_loss / len(trainloader):.4f}")

@torch.no_grad()
# def evaluate(epoch, model, testloader, device):
#     model.eval()
#     correct, total = 0, 0
#     for idx, data in tqdm(enumerate(testloader), total=len(testloader), desc=f'Validating Epoch {epoch}'):
#         data = [d.to(device) for d in data]
#         _, pred_dict = model(data, epoch)
#         if pred_dict is not None:
#             targets = data[3]  # pair label
#             preds = torch.stack([pred_dict[pair] for pair in model.dset.pairs], dim=1)
#             pred_label = preds.argmax(dim=1)
#             correct += (pred_label == targets).sum().item()
#             total += targets.size(0)
#     acc = correct / total if total > 0 else 0
#     print(f"Eval Accuracy: {acc:.4f}")
#     return acc
def test(epoch, model, testloader, evaluator, writer, args, logpath):
    global best_auc, best_hm, best_unseen, best_obj

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

    stats['a_epoch'] = epoch

    result = ''
    # print(stats['closed_seen_match'])
    for key in ['closed_obj_match','closed_seen_obj_match','closed_unseen_obj_match']:
        # writer.add_scalar(key, stats[key], epoch)
        result = result + key + '  ' + str(round(stats[key], 4)) + '| '

    print(f'Test Epoch: {epoch}')
    print(result)
    return stats['closed_unseen_obj_match']  # 返回闭合对象匹配率作为准确率


def main(logpath=None):
    global args
    triplets = generate_triplets(n=TOTAL_COMBOS, total=TOTAL_WEIGHT, step=0.05)

    with open(TRIPLET_OUTPUT, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['lambda_rec', 'lambda_rep', 'lambda_grad', 'accuracy'])

        for (rec, rep, grad) in triplets:
            # print(f"\nRunning with rec={rec}, rep={rep}, grad={grad}")
            print(f"\nRunning with rec={rec:.3f}, rep={rep:.3f}, grad={grad:.3f}")

            args = parser.parse_args()
            load_args(args.config, args)
            args.lambda_rec = rec
            args.lambda_rep = rep
            args.lambda_grad = grad
            args.max_epochs = MAX_EPOCHS

            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            trainset = dset.CompositionDataset(
                root=os.path.join(DATA_FOLDER, args.data_dir).replace('\\', '/'),
                phase='train',
                split=args.splitname
            )
            testset = dset.CompositionDataset(
                root=os.path.join(DATA_FOLDER, args.data_dir).replace('\\', '/'),
                phase='test',
                split=args.splitname
            )
            trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
            testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

            model = DBC(trainset, args).to(device)
            freeze(model.feat_extractor)
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
            evaluator_val = Evaluator(testset, model)
            for epoch in range(args.max_epochs-1):
                train(epoch, model, trainloader, optimizer, device)

            acc = test(epoch, model, testloader, evaluator_val, writer, args, logpath)
            writer.writerow([rec, rep, grad, acc])
            f.flush()

if __name__ == '__main__':
    main()
