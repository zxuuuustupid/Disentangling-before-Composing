import os
import sys
import torch
import csv
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.dbc1 import DBC
from data import dataset as dset
from utils.common import Evaluator, device
from utils.utils import load_args
from flags import parser, DATA_FOLDER

ABLATION_CONFIGS = [
    'configs/BJTU-motor.yml',
    'configs/BJTU-gearbox.yml',
    'configs/BJTU-leftaxlebox.yml',
    'configs/Canada.yml',
    'configs/German.yml',
    'configs/SWJTU.yml'
]

ABLATION_COMBOS = [
    (0, 0, 0, 0),  # all zero
    (0, 1, 0, 0),  # only rep
    (0, 0, 1, 0),  # only grad
    (0, 1, 1, 0),  # rep + grad
    (1, 0, 0, 0),  # only rec
    (1, 1, 1, 0),  # only res
    (1, 1, 1, 1),  # all one
]


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
def test(epoch, model, testloader, evaluator, writer, args, logpath):
    model.eval()
    all_attr_gt, all_obj_gt, all_pair_gt, all_pred = [], [], [], []
    for idx, data in tqdm(enumerate(testloader), total=len(testloader), desc='Testing', dynamic_ncols=True):
        data = [d.to(device) for d in data]
        _, predictions = model(data, epoch)
        attr_truth, obj_truth, pair_truth = data[1], data[2], data[3]
        all_pred.append(predictions)
        all_attr_gt.append(attr_truth)
        all_obj_gt.append(obj_truth)
        all_pair_gt.append(pair_truth)

    all_attr_gt = torch.cat(all_attr_gt).cpu()
    all_obj_gt = torch.cat(all_obj_gt).cpu()
    all_pair_gt = torch.cat(all_pair_gt).cpu()

    all_pred_dict = {k: torch.cat([pred[k].cpu() for pred in all_pred]) for k in all_pred[0].keys()}
    results = evaluator.score_model(all_pred_dict, all_obj_gt, bias=args.bias, topk=args.topk)
    stats, acc_array, confusion_accuracy = evaluator.evaluate_predictions(results, all_attr_gt, all_obj_gt, all_pair_gt,
                                                                          all_pred_dict, topk=args.topk)

    acc = stats['closed_unseen_obj_match']
    print(f'Closed-unseen-object acc = {acc:.4f}')
    return acc


def ablation_run():
    os.makedirs('result/ablation', exist_ok=True)

    for config_file in ABLATION_CONFIGS:
        config_name = os.path.basename(config_file).replace('.yml', '')
        csv_path = os.path.join('result/ablation', f'{config_name}.csv')

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['lambda_rec', 'lambda_rep', 'lambda_grad', 'lambda_res', 'accuracy'])

            for rec, rep, grad, res in ABLATION_COMBOS:
                print(f'\nRunning {config_name} with rec={rec}, rep={rep}, grad={grad}, res={res}')

                # 清理 argv 再加载 config
                sys.argv = [sys.argv[0], '--config', config_file]
                args = parser.parse_args()
                load_args(args.config, args)

                args.lambda_rec = rec
                args.lambda_rep = rep
                args.lambda_grad = grad
                args.lambda_res = res
                args.max_epochs = 2  # 设短一点避免运行时间长

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
                evaluator = Evaluator(testset, model)

                for epoch in range(args.max_epochs - 1):
                    train(epoch, model, trainloader, optimizer, device)

                acc = test(epoch, model, testloader, evaluator, writer, args, None)
                writer.writerow([rec, rep, grad, res, acc])
                f.flush()


if __name__ == '__main__':
    ablation_run()
