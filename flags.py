import argparse

DATA_FOLDER = "F:/Project/CZSL/code/Disentangling-before-Composing/Disentangling-before-Composing"

# DATA_FOLDER = "D:/zuoyichen/code"
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--config', default='configs/zappos.yml', help='path of the config file')
parser.add_argument('--dataset', default='zappos', help='utzappos|clothing|aoclevr')
parser.add_argument('--data_dir', default='ut-zap50k', help='local path to data root dir from ' + DATA_FOLDER)
parser.add_argument('--logpath', default=None, help='Path to dir where to logs are stored (test only)')
parser.add_argument('--splitname', default='compositional-split-natural', help="dataset split")
parser.add_argument('--cv_dir', default='logs/', help='dir to save checkpoints and logs to')
parser.add_argument('--name', default='temp', help='Name of exp used to name models')
parser.add_argument('--load', default=None, help='path to checkpoint to load from')
parser.add_argument('--test_set', default='val', help='val|test mode')
parser.add_argument('--test_batch_size', type=int, default=8, help="Batch size at test/eval time")
parser.add_argument('--cpu_eval', action='store_true', help='Perform test on cpu')
parser.add_argument('--seed', type=int, default=0, help='seed')

# Model parameters
parser.add_argument('--emb_dim', type=int, default=300, help='dimension of share embedding space')
parser.add_argument('--bias', type=float, default=0, help='Bias value for unseen concepts')
parser.add_argument('--drop', type=float, default=0.5, help='drop rate')
parser.add_argument('--res_epoch', type=float, default=1, help='res_epoch')
parser.add_argument('--lambda_rep', type=float, default=1 / 10, help='weight of rep losses at the representation level')
parser.add_argument('--lambda_grad', type=float, default=1 / 10, help='weight of grad losses at the gradient level')
parser.add_argument('--lambda_rec', type=float, default=1 / 10, help='weight of rec losses at the erd')
parser.add_argument('--lambda_res', type=float, default=1 / 10, help='weight of res losses at the erd')

# Hyperparameters
parser.add_argument('--topk', type=int, default=1, help="Compute topk accuracy")
parser.add_argument('--workers', type=int, default=8, help="Number of workers")
parser.add_argument('--batch_size', type=int, default=32, help="Training batch size")
parser.add_argument('--lr', type=float, default=5e-5, help="Learning rate")
parser.add_argument('--wd', type=float, default=5e-5, help="Weight decay")
parser.add_argument('--save_every', type=int, default=10000, help="Frequency of snapshots in epochs")
parser.add_argument('--eval_val_every', type=int, default=1, help="Frequency of eval in epochs")
parser.add_argument('--max_epochs', type=int, default=9, help="Max number of epochs")
