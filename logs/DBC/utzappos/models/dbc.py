import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.common import MLP, Decoder
import numpy as np
import torch.autograd as autograd
from .backbone import Backbone
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Disentangler(nn.Module):
    def __init__(self, args):
        super(Disentangler, self).__init__()
        self.fc1 = nn.Linear(args.emb_dim, args.emb_dim)
        self.bn1_fc = nn.BatchNorm1d(args.emb_dim)

    def forward(self, x):
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        return x


class DBC(nn.Module):

    def __init__(self, dset, args): 
        super(DBC, self).__init__()
        self.args = args
        self.dset = dset
        def get_all_ids(relevant_pairs):
            attrs, objs = zip(*relevant_pairs)
            attrs = [dset.attr2idx[attr] for attr in attrs]
            objs = [dset.obj2idx[obj] for obj in objs]
            pairs = [a for a in range(len(relevant_pairs))]
            attrs = torch.LongTensor(attrs).to(device)
            objs = torch.LongTensor(objs).to(device)
            pairs = torch.LongTensor(pairs).to(device)
            return attrs, objs, pairs

        self.train_attrs, self.train_objs, self.train_pairs = get_all_ids(self.dset.train_pairs) 
        self.val_attrs, self.val_objs, self.val_pairs = get_all_ids(self.dset.pairs)

        self.train_forward = self.train_forward_closed

        self.feat_extractor = Backbone('resnet18')
        feat_dim = 512
        img_emb_modules = [
        nn.Conv2d(feat_dim, args.emb_dim, kernel_size=1, bias=False), 
        nn.BatchNorm2d(args.emb_dim),
        nn.ReLU()
        ]
        self.img_embedder = nn.Sequential(*img_emb_modules) 
        self.img_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.emb_dim=args.emb_dim

        self.D = nn.ModuleDict({'da': Disentangler(args), 'do': Disentangler(args)})

        self.attr_clf = MLP(args.emb_dim, len(dset.attrs), 1, relu = False)
        self.obj_clf = MLP(args.emb_dim, len(dset.objs), 1, relu = False)

        self.drop = args.drop
        self.decoder = Decoder(2*args.emb_dim, args.emb_dim) 
        self.recon_lossf = nn.MSELoss()
        self.res_epoch = args.res_epoch

        self.lambda_rep = args.lambda_rep
        self.lambda_grad = args.lambda_grad
        self.lambda_rec = args.lambda_rec
        self.lambda_res = args.lambda_res


    def train_forward_closed(self, x, epoch):
        img, attrs, objs, pairs = x[0], x[1], x[2], x[3]
        pos_attr_img, neg_objs, pos_obj_img, neg_attrs = x[4], x[5], x[6], x[7]
        neg_obj_pairs, neg_attr_pairs = x[10],x[11]
            
        img = self.feat_extractor(img)[0]
        img = self.img_embedder(img)
        img_feat = self.img_avg_pool(img).squeeze()
        
        pos_attr_img = self.feat_extractor(pos_attr_img)[0]
        pos_attr_img = self.img_embedder(pos_attr_img)
        pos_attr_img_feat = self.img_avg_pool(pos_attr_img).squeeze()

        pos_obj_img = self.feat_extractor(pos_obj_img)[0]
        pos_obj_img = self.img_embedder(pos_obj_img)
        pos_obj_img_feat = self.img_avg_pool(pos_obj_img).squeeze()


        img_da = self.D['da'](img_feat)
        img_do = self.D['do'](img_feat)

        pos_attr_img_da = self.D['da'](pos_attr_img_feat)
        pos_obj_img_do = self.D['do'](pos_obj_img_feat)

        neg_attr_img_da = self.D['da'](pos_obj_img_feat)
        neg_obj_img_do = self.D['do'](pos_attr_img_feat)

        loss_neg_attr = F.cross_entropy(self.attr_clf(neg_attr_img_da), neg_attrs)
        loss_neg_obj = F.cross_entropy(self.obj_clf(neg_obj_img_do), neg_objs)

        img_da_p = self.attr_clf(img_da)
        img_da_pp = self.attr_clf(pos_attr_img_da)
        img_do_p = self.obj_clf(img_do)
        img_do_pp = self.obj_clf(pos_obj_img_do)

        loss_attr = F.cross_entropy(img_da_p, attrs)
        loss_pos_attr = F.cross_entropy(img_da_pp, attrs)
        loss_obj = F.cross_entropy(img_do_p, objs)
        loss_pos_obj = F.cross_entropy(img_do_pp, objs)

        loss = loss_attr + loss_obj + loss_pos_attr + loss_pos_obj + loss_neg_attr + loss_neg_obj


        attr_one_hot = torch.nn.functional.one_hot(attrs, len(self.dset.attrs))
        obj_one_hot = torch.nn.functional.one_hot(objs, len(self.dset.objs))

        img_da_g = autograd.grad((img_da_p * attr_one_hot).sum(), img_feat, retain_graph=True)[0]
        img_da_g_p = autograd.grad((img_da_pp * attr_one_hot).sum(), pos_attr_img_feat, retain_graph=True)[0]

        img_do_g = autograd.grad((img_do_p * obj_one_hot).sum(), img_feat, retain_graph=True)[0]
        img_do_g_p = autograd.grad((img_do_pp * obj_one_hot).sum(), pos_obj_img_feat, retain_graph=True)[0]

        diff_attr = torch.abs(img_da_g - img_da_g_p)
        perct_attr = torch.sort(diff_attr)[0][:, int(self.drop * self.emb_dim)]
        perct_attr = perct_attr.unsqueeze(1).repeat(1, self.emb_dim)
        mask_attr = diff_attr.lt(perct_attr.to(device)).float()

        diff_obj = torch.abs(img_do_g - img_do_g_p)
        perct_obj = torch.sort(diff_obj)[0][:, int(self.drop * self.emb_dim)]
        perct_obj = perct_obj.unsqueeze(1).repeat(1, self.emb_dim)
        mask_obj = diff_obj.lt(perct_obj.to(device)).float()

        loss_rep_attr1= F.cross_entropy(self.attr_clf(self.D['da'](img_feat * mask_attr)), attrs)
        loss_rep_obj1 = F.cross_entropy(self.obj_clf(self.D['do'](img_feat * mask_obj)), objs)

        loss_rep_attr2 = F.cross_entropy(self.attr_clf(self.D['da'](pos_attr_img_feat * mask_attr)), attrs)
        loss_rep_obj2 = F.cross_entropy(self.obj_clf(self.D['do'](pos_obj_img_feat * mask_obj)), objs)

        loss += self.lambda_rep * (loss_rep_attr1 + loss_rep_obj1+ loss_rep_attr2 + loss_rep_obj2)

        attr_grads = []
        attr_env_loss = [loss_attr, loss_pos_attr] 
        attr_network = nn.Sequential(self.D['da'], self.attr_clf)
        for i in range(2):
            attr_env_grad = autograd.grad(attr_env_loss[i], attr_network.parameters(), create_graph=True)
            attr_grads.append(attr_env_grad)
        attr_penalty_value = 0
        for i in range(len(attr_grads[0])):
            attr_penalty_value += (attr_grads[0][i] - attr_grads[1][i]).pow(2).sum()
        loss_grad_attr = attr_penalty_value
        

        obj_grads = []
        obj_env_loss = [loss_obj, loss_pos_obj]
        obj_network = nn.Sequential(self.D['do'], self.obj_clf)
        for i in range(2):
            obj_env_grad = autograd.grad(obj_env_loss[i], obj_network.parameters(), create_graph=True) 
            obj_grads.append(obj_env_grad)
        obj_penalty_value = 0
        for i in range(len(obj_grads[0])):
            obj_penalty_value += (obj_grads[0][i] - obj_grads[1][i]).pow(2).sum()
        loss_grad_obj = obj_penalty_value

        loss += self.lambda_grad * (loss_grad_attr + loss_grad_obj)


        recon_ao = self.decoder(torch.cat((img_do, img_da), dim=1))
        recon_aoo = self.decoder(torch.cat((neg_obj_img_do, pos_attr_img_da), dim=1))
        recon_aao = self.decoder(torch.cat((pos_obj_img_do, neg_attr_img_da), dim=1))

        loss_rec_ao = self.recon_lossf(recon_ao, img_feat.detach())
        loss_rec_aoo = self.recon_lossf(recon_aoo, pos_attr_img_feat.detach())
        loss_rec_aao = self.recon_lossf(recon_aao, pos_obj_img_feat.detach())
        loss_rec = self.lambda_rec * (loss_rec_ao + loss_rec_aoo + loss_rec_aao)

        loss += loss_rec  

        if epoch >= self.res_epoch:
            attr_feat = [img_da,pos_attr_img_da,neg_attr_img_da]
            obj_feat = [img_do,pos_obj_img_do,neg_obj_img_do]
            a_y = [attrs,attrs,neg_attrs]
            o_y = [objs,objs,neg_objs]

            a = torch.randperm(3)
            o = torch.randperm(3)

            new_attr_feat = [0,0,0]
            new_obj_feat = [0,0,0]
            new_comp = [0,0,0]

            loss_swap_attr = 0.0
            loss_swap_obj = 0.0

            for i in range(3):
                new_attr_feat[i]=attr_feat[a[i]]
                new_obj_feat[i]=obj_feat[o[i]]

                new_comp[i]=self.decoder(torch.cat((new_attr_feat[i], new_obj_feat[i]), dim=1))

                loss_swap_attr+=F.cross_entropy(self.attr_clf(self.D['da'](new_comp[i])), a_y[a[i]])
                loss_swap_obj+=F.cross_entropy(self.obj_clf(self.D['do'](new_comp[i])), o_y[o[i]])

            loss_swap = self.lambda_res*(loss_swap_attr+loss_swap_obj)
            loss += loss_swap  

        return loss, None

   
    def val_forward(self, x):
        img = x[0]

        img = self.feat_extractor(img)[0]
        img = self.img_embedder(img)
        img = self.img_avg_pool(img).squeeze()
          
        img_da = self.D['da'](img)
        img_do = self.D['do'](img)
        attr_pred = F.softmax(self.attr_clf(img_da), dim =1)
        obj_pred = F.softmax(self.obj_clf(img_do), dim = 1)

        scores = {}
        for itr, (attr, obj) in enumerate(self.dset.pairs):
            attr_id, obj_id = self.dset.attr2idx[attr], self.dset.obj2idx[obj]
            scores[(attr, obj)] = attr_pred[:, attr_id] * obj_pred[:, obj_id]
        return None, scores

    def forward(self, x, epoch):
        if self.training:
            loss, pred = self.train_forward(x, epoch)
        else:
            with torch.no_grad():
                loss, pred = self.val_forward(x)
        return loss, pred


def sample(alpha,x1,x2,x3):
    if alpha<=1/3:
        return x1,0
    elif alpha<=2/3 and alpha>1/3:
        return x2,0
    else:
        return x3,1

