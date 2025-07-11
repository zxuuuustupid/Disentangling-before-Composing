import sys
sys.argv += ['--logpath', 'logs/DBC/BJTU-gearbox/BJTU-gearbox9x5']
# sys.argv += ['--logpath', 'logs/DBC/German/German']
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from flags import DATA_FOLDER
from models.dbc import DBC

cudnn.benchmark = True

import tqdm
from tqdm import tqdm
import os
from os.path import join as ospj

from data import dataset as dset
from utils.common import Evaluator
from utils.utils import load_args
from flags import parser
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    args = parser.parse_args()
    logpath = args.logpath
    config = [os.path.join(logpath, _) for _ in os.listdir(logpath) if _.endswith('yml')][0]
    load_args(config, args)

    trainset = dset.CompositionDataset(
        root=os.path.join(DATA_FOLDER, args.data_dir),
        phase='train',
        split=args.splitname
    )
    testset = dset.CompositionDataset(
        root=os.path.join(DATA_FOLDER, args.data_dir),
        phase='test',
        split=args.splitname
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.workers)

    model = DBC(trainset, args)
    model = model.to(device)

    args.load = ospj(logpath, 'ckpt_best_obj.t7')
    # args.load = ospj(logpath, 'ckpt_best_auc.t7')
    # args.load = ospj(logpath, 'ckpt_best_unseen.t7')

    checkpoint = torch.load(args.load)
    model.load_state_dict(checkpoint['net'])
    model.eval()

    evaluator = Evaluator(testset, model)

    with torch.no_grad():
        test(model, testloader, evaluator, args)


def test(model, testloader, evaluator, args):
    model.eval()

    all_attr_gt, all_obj_gt, all_pair_gt, all_pred = [], [], [], []

    for idx, data in tqdm(enumerate(testloader), total=len(testloader), desc='Testing',dynamic_ncols=True  ):
        data = [d.to(device) for d in data]

        _, predictions = model(data, epoch=0)

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

    all_pred_dict = {}
    if args.cpu_eval:
        for k in all_pred[0].keys():
            all_pred_dict[k] = torch.cat(
                [all_pred[i][k].to('cpu') for i in range(len(all_pred))])
    else:
        for k in all_pred[0].keys():
            all_pred_dict[k] = torch.cat(
                [all_pred[i][k] for i in range(len(all_pred))])

    results = evaluator.score_model(all_pred_dict, all_obj_gt, bias=args.bias, topk=args.topk)
    stats = evaluator.evaluate_predictions(results, all_attr_gt, all_obj_gt, all_pair_gt, all_pred_dict, topk=args.topk)
    # print(stats)
    # print(results)

    result = ''
    for key in ['closed_attr_match', 'closed_obj_match']:
        # print(round(stats[key], 4))
        result = result + key + '  ' + str(round(stats[key], 4)) + '| '

    print('Final test results:', result)

    return results


if __name__ == '__main__':
    main()
