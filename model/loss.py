import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from visdom import Visdom
from munch import Munch

viz = Visdom()


class L2Loss(nn.Module):

    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, pred, gt):
        if len(pred) == 0:
            return torch.tensor(float('nan'), device=pred.device, dtype=pred.dtype)
        diff = pred - gt
        diff = diff.flatten(0, 1)
        return torch.norm(diff, p=2, dim=1)  # Euclidean distance


class TR4TRLoss(nn.Module):

    def __init__(self, model='', w_tr=1, w_depth=1, w_sf=1, alpha=0, loss_type='L2'):
        super(TR4TRLoss, self).__init__()
        self.model = model
        self.w_tr = w_tr        # tracking and reconstruction coefficient
        self.w_depth = w_depth  # depth coefficient
        self.w_sf = w_sf        # scene flow coefficient
        self.alpha = alpha      # parameter regularization coefficient
        self.loss_type = loss_type
        if loss_type == 'L1':
            self.loss_fun = nn.L1Loss(reduction='none')
        elif loss_type == 'L2':
            self.loss_fun = L2Loss()
        elif loss_type == 'MSE':
            self.loss_fun = nn.MSELoss(reduction='none')
        else:
            self.loss_fun = nn.SmoothL1Loss()

    def cal_loss(self, point_set, targets):
        source_point_pred, target_point_pred = point_set[..., :3], point_set[..., 3:]
        [source_rgbd, target_rgbd, point_cloud, scene_flow_gt, scene_flow_mask] = targets
        source_color, target_color = source_rgbd[..., :3], target_rgbd[..., :3]
        source_point_gt, target_point_gt = point_cloud[..., :3], point_cloud[..., 3:]
        batch_size, num_points = point_set.shape[0], point_set.shape[1]

        # calculate the matching point id according to source point
        cost_dist_i = torch.cdist(source_point_pred.to(source_point_gt.dtype), source_point_gt, p=2)
        point_gt_id = np.array([list(linear_sum_assignment(cost_dist_i[i].cpu().detach().numpy())[1]) for i in range(batch_size)])
        for i in range(batch_size):
            source_point_gt[i], target_point_gt[i] = source_point_gt[i, point_gt_id[i]], target_point_gt[i, point_gt_id[i]]

        source_point_loss_all = self.loss_fun(source_point_pred, source_point_gt)
        target_point_loss_all = self.loss_fun(target_point_pred, target_point_gt)
        scene_flow_loss_all = self.loss_fun(target_point_pred - source_point_pred,
                                            target_point_gt - source_point_gt)
        source_point_depth_loss_all = self.loss_fun(source_point_pred[..., 2:],
                                                    source_point_gt[..., 2:])
        target_point_depth_loss_all = self.loss_fun(target_point_pred[..., 2:],
                                                    target_point_gt[..., 2:])
        source_point_loss = torch.mean(source_point_loss_all)
        target_point_loss = torch.mean(target_point_loss_all)
        scene_flow_loss = torch.mean(scene_flow_loss_all)
        source_point_depth_loss = torch.mean(source_point_depth_loss_all)
        target_point_depth_loss = torch.mean(target_point_depth_loss_all)
        total_loss = self.w_tr * (source_point_loss + target_point_loss) + \
                     self.w_depth * (source_point_depth_loss + target_point_depth_loss) + \
                     self.w_sf * scene_flow_loss

        # L2 loss regularization
        if self.alpha != 0:
            for name, param in self.model.named_parameters():
                if 'bias' not in name:
                    total_loss += self.alpha * torch.sum(torch.pow(param, 2))

        # visualize the training process with visdom
        viz.scatter(source_point_gt[0], win='source point gt',
                    opts={'title': "source points gt", 'markersize': 1})
        viz.scatter(target_point_gt[0], win='target point gt',
                    opts={'title': "target points gt", 'markersize': 1})
        viz.scatter(source_point_pred[0], win='source point pred',
                    opts={'title': "source point pred", 'markersize': 1})
        viz.scatter(target_point_pred[0], win='target point pred',
                    opts={'title': "target point pred", 'markersize': 1})
        images = torch.stack([source_color[0].permute(2, 0, 1),
                              target_color[0].permute(2, 0, 1),
                              scene_flow_gt[0].permute(2, 0, 1) * 1000], dim=0)
        viz.images(images, win='source & target', opts={'title': "source / target / scene_flow"})

        print('source_point: %f, target_point: %f, source_depth: %f, target_depth: %f, '
              'scene_flow: %f, total_loss: %f'
              % (source_point_loss.cpu().detach().numpy(),
                 target_point_loss.cpu().detach().numpy(),
                 source_point_depth_loss.cpu().detach().numpy(),
                 target_point_depth_loss.cpu().detach().numpy(),
                 scene_flow_loss.cpu().detach().numpy(),
                 total_loss.cpu().detach().numpy()))

        return source_point_loss_all, target_point_loss_all, source_point_depth_loss_all, \
               target_point_depth_loss_all, scene_flow_loss_all, total_loss

    def forward(self, outputs, targets, evaluate):
        source_point_loss, target_point_loss, source_point_depth_loss, target_point_depth_loss, \
        scene_flow_loss, total_loss = self.cal_loss(outputs, targets)
        if evaluate:
            return source_point_loss, target_point_loss, source_point_depth_loss, \
                   target_point_depth_loss, scene_flow_loss, total_loss
        else:
            return total_loss


# Copyright (c) Microsoft Corporation.
def relative_constraint_l1(deltaxy, predxy):
    return F.l1_loss(deltaxy, predxy)


def relative_constraint_ce(deltaxy, predxy):
    # predx, predy = torch.chunk(predxy, chunks=2, dim=1)
    predx, predy = predxy[:, :, 0], predxy[:, :, 1]
    targetx, targety = deltaxy[:, 0].long(), deltaxy[:, 1].long()
    return F.cross_entropy(predx, targetx) + F.cross_entropy(predy, targety)


def variance_aware_regression(pred, beta, target, labels, lambda_var=0.001):
    EPSILON = 1e-8
    # Variance aware regression.
    pred_titled = pred.unsqueeze(0).t().repeat(1, labels.size(1))
    pred_var = torch.sum((labels-pred_titled)**2*beta, dim=1) + EPSILON
    pred_log_var = torch.log(pred_var)
    squared_error = (pred - target)**2
    return torch.mean(torch.exp(-pred_log_var) * squared_error + lambda_var * pred_log_var)


# based on the codes: https://github.com/google-research/google-research/blob/master/tcc/tcc/losses.py
def relative_constraint_cbr(deltaxy, predxy, loss_type="regression_mse_var"):
    predx, predy = predxy[:, :, 0], predxy[:, :, 1]
    num_classes = predx.size(1)
    targetx, targety = deltaxy[:, 0].long(), deltaxy[:, 1].long()    # [N, ], [N, ]
    betax, betay = F.softmax(predx, dim=1), F.softmax(predy, dim=1)  # [N, C], [N, C]
    labels = torch.arange(num_classes).unsqueeze(0).to(predxy.device)  # [1, C]
    true_idx = targetx  # torch.sum(targetx*labels, dim=1)      # [N, ]
    true_idy = targety  # torch.sum(targety*labels, dim=1)      # [N, ]

    pred_idx = torch.sum(betax*labels, dim=1)        # [N, ]
    pred_idy = torch.sum(betay*labels, dim=1)        # [N, ]

    if loss_type in ["regression_mse", "regression_mse_var"]:
        if "var" in loss_type:
            # Variance aware regression.
            lossx = variance_aware_regression(pred_idx, betax, true_idx, labels)
            lossy = variance_aware_regression(pred_idy, betay, true_idy, labels)
        else:
            lossx = torch.mean((pred_idx - true_idx)**2)
            lossy = torch.mean((pred_idy - true_idy)**2)
        loss = lossx + lossy
        return loss
    else:
        raise NotImplementedError("We only support regression_mse and regression_mse_var now.")


def cal_selfsupervised_loss(outs, drloc_mode='l1', lambda_drloc=0.0):
    loss, all_losses = 0.0, Munch()
    if drloc_mode == "l1":  # l1 regression constraint
        reld_criterion = relative_constraint_l1
    elif drloc_mode == "ce":  # cross entropy constraint
        reld_criterion = relative_constraint_ce
    elif drloc_mode == "cbr":  # cycle-back regression constaint: https://arxiv.org/pdf/1904.07846.pdf
        reld_criterion = relative_constraint_cbr
    else:
        raise NotImplementedError("We only support l1, ce and cbr now.")

    loss_drloc = 0.0
    for deltaxy, drloc, plane_size in zip(outs.deltaxy, outs.drloc, outs.plz):
        loss_drloc += reld_criterion(deltaxy, drloc) * lambda_drloc
    all_losses.drloc = loss_drloc.item()
    loss += loss_drloc

    return loss, all_losses
