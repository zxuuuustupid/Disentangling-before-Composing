import sys

import torch
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from models.dbc import DBC
from data import dataset as dset
from utils.common import Evaluator
from utils.utils import save_args, load_args
from flags import parser, DATA_FOLDER
from models.dbc import visualize_attr_distribution_under_obj
sys.argv += ['--config', 'configs/German.yml']

def freeze(m):
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
        p.grad = None

def train(epoch, model, trainloader, optimizer):
    model.train()
    freeze(model.feat_extractor)
    total_loss = 0.0

    for idx, data in tqdm(enumerate(trainloader), total=len(trainloader), desc=f'Training Epoch {epoch}'):
        data = [d.to(device) for d in data]
        loss, _ = model(data, epoch)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        total_loss += loss.item()

        # 可视化
        if epoch == args.vis_epoch and idx == 0:
            with torch.no_grad():
                model.eval()
                img_feat = model.cached['img_feat']
                new_comp = model.cached['new_comp']
                attrs, objs = data[1], data[2]
                visualize_attr_distribution_under_obj(img_feat, new_comp[0], attrs, objs, obj_id=1, epoch=epoch)

    print(f"Epoch {epoch} Loss: {total_loss / len(trainloader):.4f}")


def main():
    global args, device
    args = parser.parse_args()
    load_args(args.config, args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainset = dset.CompositionDataset(
        root=os.path.join(DATA_FOLDER, args.data_dir).replace('\\', '/'),
        phase='train',
        split=args.splitname)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    model = DBC(trainset, args).to(device)
    freeze(model.feat_extractor)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    for epoch in range(args.max_epochs + 1):
        train(epoch, model, trainloader, optimizer)


if __name__ == '__main__':
    main()