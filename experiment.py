import sys
import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.dbc1 import DBC
from data import dataset as dset
from utils.utils import load_args
from flags import parser, DATA_FOLDER

sys.argv += ['--config', 'configs/BJTU-motor.yml']

# sys.argv += ['--config', 'configs/BJTU-leftaxlebox.yml']
# sys.argv += ['--config', 'configs/SWJTU.yml']
# sys.argv += ['--config', 'configs/Canada.yml']
# sys.argv += ['--config', 'configs/BJTU-gearbox.yml']

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
def evaluate(epoch, model, testloader, device):
    model.eval()
    print(f"Evaluating Epoch {epoch} ...")
    for idx, data in tqdm(enumerate(testloader), total=len(testloader), desc=f'Validating Epoch {epoch}'):
        data = [d.to(device) for d in data]
        _ = model(data, epoch)  # 忽略返回值，或收集 pred 做后处理也行


def main():
    global args
    args = parser.parse_args()
    load_args(args.config, args)
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

    for epoch in range(args.max_epochs + 1):
        train(epoch, model, trainloader, optimizer, device)
        evaluate(epoch, model, testloader, device)

if __name__ == '__main__':
    main()
