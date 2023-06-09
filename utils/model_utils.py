import torch
import torch.nn as nn
import math
import warnings


# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright 2020 Ross Wightman
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def init_weights_(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


def randn_sampling(maxint, sample_size, batch_size):
    return torch.randint(maxint, size=(batch_size, sample_size, 2))


def collect_samples(feats, pxy, batch_size):
    return torch.stack([feats[i, :, pxy[i][:,0], pxy[i][:,1]] for i in range(batch_size)], dim=0)


def collect_samples_faster(feats, pxy, batch_size):
    n,c,h,w = feats.size()
    feats = feats.view(n, c, -1).permute(1,0,2).reshape(c, -1)  # [n, c, h, w] -> [n, c, hw] -> [c, nhw]
    pxy = ((torch.arange(n).long().to(pxy.device) * h * w).view(n, 1) + pxy[:,:,0]*h + pxy[:,:,1]).view(-1)  # [n, m, 2] -> [nm]
    return (feats[:,pxy]).view(c, n, -1).permute(1,0,2)


def collect_positions(batch_size, N):
    all_positions = [[i,j]  for i in range(N) for j in range(N)]
    pts = torch.tensor(all_positions) # [N*N, 2]
    pts_norm = pts.repeat(batch_size,1,1)  # [B, N*N, 2]
    rnd = torch.stack([torch.randperm(N*N) for _ in range(batch_size)], dim=0) # [B, N*N]
    pts_rnd = torch.stack([pts_norm[idx,r] for idx, r in enumerate(rnd)],dim=0) # [B, N*N, 2]
    return pts_norm, pts_rnd


# Copyright (c) Microsoft Corporation.
class DenseRelativeLoc(nn.Module):
    def __init__(self, in_dim, out_dim=2, sample_size=32, drloc_mode="l1", use_abs=False):
        super(DenseRelativeLoc, self).__init__()
        self.sample_size = sample_size
        self.in_dim = in_dim
        self.drloc_mode = drloc_mode
        self.use_abs = use_abs

        if self.drloc_mode == "l1":
            self.out_dim = out_dim
            self.layers = nn.Sequential(
                nn.Linear(in_dim * 2, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, self.out_dim)
            )
        elif self.drloc_mode in ["ce", "cbr"]:
            self.out_dim = out_dim if self.use_abs else out_dim * 2 - 1
            self.layers = nn.Sequential(
                nn.Linear(in_dim * 2, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512)
            )
            self.unshared = nn.ModuleList()
            for _ in range(2):
                self.unshared.append(nn.Linear(512, self.out_dim))
        else:
            raise NotImplementedError("We only support l1, ce and cbr now.")

    def forward_features(self, x, mode="part"):
        # x, feature map with shape: [B, C, H, W]
        B, C, H, W = x.size()

        if mode == "part":
            pxs = randn_sampling(H, self.sample_size, B).detach()
            pys = randn_sampling(H, self.sample_size, B).detach()

            deltaxy = (pxs - pys).float().to(x.device)  # [B, sample_size, 2]

            ptsx = collect_samples_faster(x, pxs, B).transpose(1, 2).contiguous()  # [B, sample_size, C]
            ptsy = collect_samples_faster(x, pys, B).transpose(1, 2).contiguous()  # [B, sample_size, C]
        else:
            pts_norm, pts_rnd = collect_positions(B, H)
            ptsx = x.view(B, C, -1).transpose(1, 2).contiguous()  # [B, H*W, C]
            ptsy = collect_samples(x, pts_rnd, B).transpose(1, 2).contiguous()  # [B, H*W, C]

            deltaxy = (pts_norm - pts_rnd).float().to(x.device)  # [B, H*W, 2]

        pred_feats = self.layers(torch.cat([ptsx, ptsy], dim=2))
        return pred_feats, deltaxy, H

    def forward(self, x, normalize=False):
        pred_feats, deltaxy, H = self.forward_features(x)
        deltaxy = deltaxy.view(-1, 2)  # [B*sample_size, 2]

        if self.use_abs:
            deltaxy = torch.abs(deltaxy)
            if normalize:
                deltaxy /= float(H - 1)
        else:
            deltaxy += (H - 1)
            if normalize:
                deltaxy /= float(2 * (H - 1))

        if self.drloc_mode == "l1":
            predxy = pred_feats.view(-1, self.out_dim)  # [B*sample_size, Output_size]
        else:
            predx, predy = self.unshared[0](pred_feats), self.unshared[1](pred_feats)
            predx = predx.view(-1, self.out_dim)  # [B*sample_size, Output_size]
            predy = predy.view(-1, self.out_dim)  # [B*sample_size, Output_size]
            predxy = torch.stack([predx, predy], dim=2)  # [B*sample_size, Output_size, 2]
        return predxy, deltaxy

    def flops(self):
        fps = self.in_dim * 2 * 512 * self.sample_size
        fps += 512 * 512 * self.sample_size
        fps += 512 * self.out_dim * self.sample_size
        if self.drloc_mode in ["ce", "cbr"]:
            fps += 512 * 512 * self.sample_size
            fps += 512 * self.out_dim * self.sample_size
        return fps


class DownsampleLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownsampleLayer, self).__init__()
        self.Conv_BN_ReLU_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.Conv_BN_ReLU_2(x)
        out_2 = self.downsample(out)
        return out, out_2


class UpSampleLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpSampleLayer, self).__init__()
        self.Conv_BN_ReLU_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch*2),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch*2, out_channels=out_ch*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch*2),
            nn.ReLU()
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels=out_ch*2, out_channels=out_ch, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x, out):
        x_out = self.Conv_BN_ReLU_2(x)
        x_out = self.upsample(x_out)
        cat_out = torch.cat((x_out, out), dim=1)
        return cat_out


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        out_channels = [2**(i+6) for i in range(5)]  # [64, 128, 256, 512, 1024]

        self.d1 = DownsampleLayer(3, out_channels[0])  # 3-64
        self.d2 = DownsampleLayer(out_channels[0], out_channels[1])    # 64-128
        self.d3 = DownsampleLayer(out_channels[1], out_channels[2])    # 128-256
        self.d4 = DownsampleLayer(out_channels[2], out_channels[3])    # 256-512

        self.u1 = UpSampleLayer(out_channels[3], out_channels[3])     # 512-1024-512
        self.u2 = UpSampleLayer(out_channels[4], out_channels[2])     # 1024-512-256
        self.u3 = UpSampleLayer(out_channels[3], out_channels[1])     # 512-256-128
        self.u4 = UpSampleLayer(out_channels[2], out_channels[0])     # 256-128-64

        self.o = nn.Sequential(
            nn.Conv2d(out_channels[1], out_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(),
            nn.Conv2d(out_channels[0], out_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(),
            nn.Conv2d(out_channels[0], 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
            # BCELoss
        )

    def forward(self, x):
        out_1, out1 = self.d1(x)     # (3, 240, 320) -> (64, 240, 320) (64, 120, 160)
        out_2, out2 = self.d2(out1)  # (64, 120, 160) -> (128, 120, 160) (128, 60, 80)
        out_3, out3 = self.d3(out2)  # (128, 60, 80) -> (256, 60, 80) (256, 30, 40)
        out_4, out4 = self.d4(out3)  # (256, 30, 40) -> (512, 30, 40) (512, 15, 20)
        out5 = self.u1(out4, out_4)  # (512, 15, 20) (512, 30, 40) -> (1024, 30, 40)
        out6 = self.u2(out5, out_3)  # (1024, 30, 40) (256, 60, 80) -> (512, 60, 80)
        out7 = self.u3(out6, out_2)  # (512, 60, 80) (128, 120, 160) -> (256, 120, 160)
        out8 = self.u4(out7, out_1)  # (256, 120, 160) (64, 240, 320) -> (128, 240. 320)
        out = self.o(out8)           # (128, 240, 320) -> (1, 240, 320)
        return out