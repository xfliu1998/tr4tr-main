import argparse
import math
import random
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist
from apex.parallel import DistributedDataParallel
from experiment.train import *
from experiment.predict import predict
from utils.experiment_utils import *
from model.dataset import DeformDataset
from model.model import TR4TR
from model.loss import TR4TRLoss, cal_selfsupervised_loss

import warnings
warnings.filterwarnings("ignore")

# add everiment variable
os.environ["CUDA_VISBLE_DEVICES"] = '0,1'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '23456'
os.environ['RANK'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# load hyperparameter
cfg = load_config()
data_path = cfg['data']['dataset_path']
pretrained_model_path = cfg['data']['pretrained_model_path']
seed = cfg['data']['seed']
data_crop = cfg['data']['data_crop']
data_normal = cfg['data']['data_normal']
data_mask = cfg['data']['data_mask']
data_transform = cfg['data']['data_transform']
flow_reverse = cfg['data']['flow_reverse']

batch_size = cfg['model']['batch_size']
img_size = cfg['model']['img_size']
attention_type = cfg['model']['attention_type']
patch_size = cfg['model']['patch_size']
num_point = cfg['model']['num_point']
pretrained_model = cfg['model']['pretrained_model']
num_layers = cfg['model']['num_layers']
num_heads = cfg['model']['num_heads']
mlp_ratio = cfg['model']['mlp_ratio']
qkv_bias = cfg['model']['qkv_bias']
qk_scale = cfg['model']['qk_scale']
drop_rate = cfg['model']['drop_rate']
attn_drop_rate = cfg['model']['attn_drop_rate']
drop_path_rate = cfg['model']['drop_path_rate']
space_pos = cfg['model']['space_pos']
time_pos = cfg['model']['time_pos']
query_pos = cfg['model']['query_pos']

w_mask = cfg['loss']['w_mask']
w_tr = cfg['loss']['w_tr']
w_depth = cfg['loss']['w_depth']
w_sf = cfg['loss']['w_sf']
w_drloc = cfg['loss']['w_drloc']
w_rev = cfg['loss']['w_rev']
drloc_mode = cfg['loss']['drloc_mode']
sample_size = cfg['loss']['sample_size']
alpha = cfg['loss']['alpha']
loss_type = cfg['loss']['loss_type']

use_amp = cfg['train']['use_amp']
epochs = cfg['train']['epochs']
accum_iter = cfg['train']['accum_iter']
lr = cfg['train']['learning_rate']
lr_decay_type = cfg['train']['lr_decay_type']
warmup_step = cfg['train']['warmup_step']
weight_decay = cfg['train']['weight_decay']
optimizer_name = cfg['train']['optimizer_name']
momentum = cfg['train']['momentum']

img_size = list(map(int, img_size[1:-1].split(',')))

# random seed
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

loss_log = 'loss_log_seed%d_bs%d_img%d.csv' % (seed, batch_size, img_size[0])


def load_data(data_mode):
    dataset = DeformDataset(data_path, data_mode, img_size[:2], data_crop, data_normal, data_transform,
                            flow_reverse, data_mask, num_point)  # train 12470; val 683
    # data_single = next(iter(dataset))

    # use num_workers to set parallel
    if data_mode == 'train' or data_mode == 'val':
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=True,
                            num_workers=4, pin_memory=True, sampler=sampler)
    else:
        loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                            num_workers=0, pin_memory=True)
    return loader


def init_model():
    model = TR4TR(batch_size=batch_size, img_size=img_size, attention_type=attention_type, patch_size=patch_size,
                   num_point=num_point, num_layers=num_layers, num_heads=num_heads, mlp_ratio=mlp_ratio,
                   qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                   drop_path_rate=drop_path_rate, pretrained_model=pretrained_model, pretrained_model_path=pretrained_model_path,
                   drloc_mode=drloc_mode, sample_size=sample_size, space_pos=space_pos, time_pos=time_pos, query_pos=query_pos)
    return model


def main_worker(local_rank, args):
    global_rank = local_rank + args.node_rank * args.nproc_per_node
    world_size = args.nnode * args.nproc_per_node
    dist.init_process_group(backend="nccl", init_method='env://', rank=global_rank, world_size=world_size)

    train_loader = load_data(data_mode='train')
    val_loader = load_data(data_mode='val')

    model = init_model()
    para_num = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (para_num / 1e6))

    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)
    # model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    model = DistributedDataParallel(model)

    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # load pre-trained optimizer
    if pretrained_model != '':
        pretrained_dict = torch.load(pretrained_model_path + pretrained_model, map_location='cpu')['optimizer']
        optimizer.load_state_dict(pretrained_dict)
        for param_group in optimizer.param_groups:  # modify lr
            param_group["lr"] = lr
        del pretrained_dict

    if lr_decay_type == 'warmup_step':
        t, T = warmup_step, epochs
        lr_lambda = lambda epoch: (0.9 * epoch / t + 0.1) if epoch < t \
            else 0.1 if 0.5 * (1 + math.cos(math.pi * (epoch - t) / (T - t))) < 0.1 \
            else 0.5 * (1 + math.cos(math.pi * (epoch - t) / (T - t)))
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif lr_decay_type == 'consine_anneal':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif lr_decay_type == 'linear':
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.2, total_iters=epochs*3117)
    else:  # step
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(1e9), gamma=0.1, last_epoch=-1)

    if use_amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    criterion_sup = TR4TRLoss(model=model, w_tr=w_tr, w_depth=w_depth, w_sf=w_sf, alpha=alpha, loss_type=loss_type)
    criterion_sup.cuda(local_rank)
    if drloc_mode != '':
        criterion_ssup = cal_selfsupervised_loss
    else:
        criterion_ssup = False

    train(epochs=epochs, accum_iter=accum_iter, train_loader=train_loader, val_loader=val_loader, model=model, criterion_sup=criterion_sup,
          criterion_ssup=criterion_ssup, w_mask=w_mask, drloc_mode=drloc_mode, w_drloc=w_drloc, w_rev=w_rev, optimizer=optimizer,
          scheduler=scheduler, use_amp=use_amp, local_rank=local_rank, ngpus_per_node=args.nproc_per_node, loss_log=loss_log)

    dist.destroy_process_group()


def main():
    if args.mode == 'train':
        print('------------training and evaluating---------------')
        mp.spawn(main_worker, nprocs=args.nproc_per_node, args=(args,))

    elif args.mode == 'predict':
        print('------------predicting---------------')
        test_loader = load_data(data_mode='val_')
        model = init_model()
        predict(model, test_loader, pretrained_model)

    else:
        print('Please specify the mode: train or predict')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--nproc_per_node', default=2, type=int, help='nums of process/gpu')
    parser.add_argument('--nnode', default=1, type=int, help='nums of node')
    parser.add_argument('--node_rank', default=0, type=int)
    parser.add_argument("--mode", default='train', help='train or predict')

    args = parser.parse_args()

    main()
