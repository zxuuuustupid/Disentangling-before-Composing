import sys
import torch
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from models.dbc import DBC, visualize_attr_distribution_under_obj
from data import dataset as dset
from utils.common import Evaluator
from utils.utils import save_args, load_args
from flags import parser, DATA_FOLDER

sys.argv += ['--config', 'configs/German.yml']

def freeze(m):
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
        p.grad = None

def extract_features(model, imgs):
    # imgs: tensor [B,C,H,W]
    model.eval()
    with torch.no_grad():
        feat = model.feat_extractor(imgs)[0]
        feat = model.img_embedder(feat)
        feat = model.img_avg_pool(feat).squeeze()
    model.train()
    return feat

def extract_features_from_loader(model, dataloader, obj_id, max_per_attr=100):
    all_feats_before = []
    all_feats_after = []
    all_attrs = []
    all_objs = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Extracting features for obj={obj_id}"):
            imgs, attrs, objs = batch[0].to(device), batch[1].to(device), batch[2].to(device)

            mask = (objs == obj_id)
            if mask.sum() == 0:
                continue

            imgs = imgs[mask]
            attrs = attrs[mask]
            objs = objs[mask]

            feat_before = extract_features(model, imgs)
            feat_after = feat_before.clone()  # 可换为重排后的版本

            all_feats_before.append(feat_before.cpu())
            all_feats_after.append(feat_after.cpu())
            all_attrs.append(attrs.cpu())
            all_objs.append(objs.cpu())

    if not all_feats_before:
        print(f"No samples found with object id {obj_id}")
        return

    feat_before = torch.cat(all_feats_before)
    feat_after = torch.cat(all_feats_after)
    attrs = torch.cat(all_attrs)
    objs = torch.cat(all_objs)

    selected_idx = []
    for attr_id in torch.unique(attrs):
        idx = (attrs == attr_id).nonzero(as_tuple=True)[0]
        if len(idx) > max_per_attr:
            idx = idx[torch.randperm(len(idx))[:max_per_attr]]
        selected_idx.append(idx)

    selected_idx = torch.cat(selected_idx)

    return feat_before[selected_idx], feat_after[selected_idx], attrs[selected_idx], objs[selected_idx]

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

    print(f"Epoch {epoch} Loss: {total_loss / len(trainloader):.4f}")

    # ✅ epoch=2 结束后统一进行一次可视化（全训练集）
    if epoch == 2:
        print("Generating visualization from full trainset for obj_id = 1 ...")
        feat_b, feat_a, attrs, objs = extract_features_from_loader(model, trainloader, obj_id=1, max_per_attr=100)
        visualize_attr_distribution_under_obj(feat_b, feat_a, attrs, objs, obj_id=1, epoch=epoch)


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
