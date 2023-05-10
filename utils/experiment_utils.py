import os
import yaml
import matplotlib.pyplot as plt
import torch
from torch import nn
from utils.model_utils import UNet


def plot_curve(curve_list, curve_name, output_path):
    """
    plot output data curve
    :param curve_list:
    :param curve_name:
    :param output_path:
    :return:
    """
    plt.plot(range(len(curve_list)), curve_list, linewidth=3)
    plt.xlabel('epoch')
    plt.ylabel(curve_name)
    plt.savefig(output_path + '{}_epoch.pdf'.format(curve_name))


def load_config():
    """
    load hyperparameter file
    :return:
    """
    # get current file path
    cur_path = os.path.dirname(os.path.realpath(__file__))
    # get yaml file path
    with open("config.yaml", 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg


def mask_utils(source, target, mask):
    """
    get the mask of the source and target
    :param source:
    :param target:
    :param mask:
    :return:
    """
    cfg = load_config()
    pretrained_model_path = cfg['data']['pretrained_model_path']
    masknet_pretrain = cfg['data']['masknet_pretrain']
    mask_loss = None
    mask_net = UNet()
    if masknet_pretrain:
        masknet_pretrained_dict = torch.load(pretrained_model_path + masknet_pretrain, map_location='cpu')['state_dict']
        mask_net.load_state_dict(masknet_pretrained_dict, strict=False)
    else:
        mask_pred = mask_net(source[..., :3].permute(0, 3, 1, 2))
        mask_gt = torch.where(mask[..., :1].permute(0, 3, 1, 2), 1, 0)
        mask_loss = nn.BCELoss(mask_pred, mask_gt)
    source_mask = mask_net(source[..., :3].permute(0, 3, 1, 2))
    target_mask = mask_net(target[..., :3].permute(0, 3, 1, 2))
    source = source_mask.permute(0, 2, 3, 1) * source
    target = target_mask.permute(0, 2, 3, 1) * target
    return source, target, mask_loss