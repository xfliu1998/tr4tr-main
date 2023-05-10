import torch.nn.functional as F
from functools import partial
from einops import rearrange
from utils.model_utils import *
from utils.experiment_utils import *
from munch import Munch


class CNN(nn.Module):
    def __init__(self, input_channel, output_channel, input_frames, output_frames):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=1)
        self.conv2 = nn.Conv3d(in_channels=input_frames, out_channels=output_frames, kernel_size=1)

    def forward(self, x):
        b, h = x.shape[0], x.shape[1]
        x = rearrange(x, 'b h w c t -> (b t) c h w')
        x = self.conv1(x)
        x = rearrange(x, '(b t) c h w -> b t c h w', b=b)
        x = self.conv2(x)
        x = rearrange(x, 'b t c h w -> b h w c t', h=h)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads  # 1536//12 = 128
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
           self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
           self.proj = nn.Linear(dim, dim)
           self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape
        if self.with_qkv:
           qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
           q, k, v = qkv[0], qkv[1], qkv[2]
        else:
           qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
           q, k, v = qkv, qkv, qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
           x = self.proj(x)
           x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        # Temporal Attention Parameters
        self.temporal_norm1 = norm_layer(dim)
        self.temporal_attn = Attention(
          dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.temporal_fc = nn.Linear(dim, dim)

        # drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, B, T, W):
        # x = (b, 400*t+1, 1536)  W=20
        num_spatial_tokens = (x.size(1) - 1) // T  # 400
        H = num_spatial_tokens // W    # 20
        # Temporal
        xt = x[:, 1:, :]  # (b, 400*2, 1536)
        xt = rearrange(xt, 'b (h w t) m -> (b h w) t m', b=B, h=H, w=W, t=T)  # -> (b*400, t, 1536)
        res_temporal = self.drop_path(self.temporal_attn(self.temporal_norm1(xt)))
        res_temporal = rearrange(res_temporal, '(b h w) t m -> b (h w t) m', b=B, h=H, w=W, t=T)
        res_temporal = self.temporal_fc(res_temporal)  # (b, 400*t, 1536)
        xt = x[:, 1:, :] + res_temporal  # (b, 400*t, 1536)

        # Spatial
        init_cls_token = x[:, 0, :].unsqueeze(1)    # (b, 1, 1536)
        cls_token = init_cls_token.repeat(1, T, 1)  # (b, t, 1536)
        cls_token = rearrange(cls_token, 'b t m -> (b t) m', b=B, t=T).unsqueeze(1)  # (b*t, 1, 1536)
        xs = xt  # (b, 400*t, 1536)
        xs = rearrange(xs, 'b (h w t) m -> (b t) (h w) m', b=B, h=H, w=W, t=T)  # (b*t, 400, 1536)
        xs = torch.cat((cls_token, xs), 1)  # (b*t, 401, 1536)
        res_spatial = self.drop_path(self.attn(self.norm1(xs)))   # (b*t, 401, 1536)

        # Taking care of CLS token
        cls_token = res_spatial[:, 0, :]   # (b*t, 1, 1536)
        cls_token = rearrange(cls_token, '(b t) m -> b t m', b=B, t=T)  # (b, t, 1536)
        cls_token = torch.mean(cls_token, 1, True)   # averaging for every frame   # (b, 1, 1536)
        res_spatial = res_spatial[:, 1:, :]  # (b*t, 400, 1536)
        res_spatial = rearrange(res_spatial, '(b t) (h w) m -> b (h w t) m', b=B, h=H, w=W, t=T)
        res = res_spatial  # (b, 400*t, 1536)
        x = xt  # (b, 400*t, 1536)

        # Mlp
        x = torch.cat((init_cls_token, x), 1) + torch.cat((cls_token, res), 1)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x  # (b, 400*t+1, 1536)


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=(320, 320, 6, 2), patch_size=16, in_channels=6, embed_dim=1536):
        super(PatchEmbed, self).__init__()
        num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)  # 400
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, H, W, C, T = x.shape  # batch, height, width, channel, time
        x = rearrange(x, 'b h w c t -> (b t) c h w')
        x = self.proj(x)  # (b t) c h w -> (b*t, embed_dim, 20, 20)
        W = x.size(-1)  # 20
        x = x.flatten(2).transpose(1, 2)   # (b*t, 400, embed_dim)
        return x, T, W


# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright 2020 Ross Wightman
# Modified TimeSformer definition
class TimeSformer(nn.Module):

    def __init__(self, batch_size=64, img_size=(320, 320, 6, 2), patch_size=16, num_layers=12, num_heads=12,
                 mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 drloc_mode='l1', sample_size=32, norm_layer=nn.LayerNorm, space_pos=True, time_pos=True, **kwargs):
        super(TimeSformer, self).__init__()
        self.batch_size = batch_size
        self.img_size = img_size
        h, w, in_channels, num_frames = self.img_size  # 320 320 6 2
        self.patch_size = patch_size  # 16
        self.embed_dim = patch_size * patch_size * img_size[2]  # 16*16*6=1536
        self.num_heads = num_heads
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_channels=in_channels, embed_dim=self.embed_dim)
        num_patches = self.patch_embed.num_patches   # 20*20 = 400
        self.use_drloc = (drloc_mode != '')
        self.space_pos = space_pos
        self.time_pos = time_pos

        # Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))  # (1, 1, 1536)
        if self.space_pos:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))  # (1, 401, 1536)
            trunc_normal_(self.pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        if self.time_pos:
            self.time_embed = nn.Parameter(torch.zeros(1, num_frames, self.embed_dim))  # (1, 2, 1536)
        self.time_drop = nn.Dropout(p=drop_rate)

        # Attention Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]  # stochastic num_layers decay rule
        self.blocks = nn.ModuleList([
            Block(dim=self.embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(num_layers)])
        self.norm = norm_layer(self.embed_dim)

        if self.use_drloc:
            self.drloc = DenseRelativeLoc(
                in_dim=self.embed_dim,
                out_dim=2 if drloc_mode == "l1" else 14,
                sample_size=sample_size,
                drloc_mode=drloc_mode,
                use_abs=False
            )

        trunc_normal_(self.cls_token, std=.02)
        self.apply(init_weights_)

        # initialization of temporal attention weights
        i = 0
        for m in self.blocks.modules():
            m_str = str(m)
            if 'Block' in m_str:
                if i > 0:
                    nn.init.constant_(m.temporal_fc.weight, 0)
                    nn.init.constant_(m.temporal_fc.bias, 0)
                i += 1

    def forward(self, x):
        B, H, W, C, T = x.shape  # batch, height, width, channel, time
        x, T, W = self.patch_embed(x)  # x: (b*t, 400, embed_dim)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)  # (b*t, 1, 1536)
        x = torch.cat((cls_tokens, x), dim=1)   # (b*t, 401, 1536)

        if self.space_pos:
            # resizing the positional embeddings in case they don't match the input at inference
            if x.size(1) != self.pos_embed.size(1):
                pos_embed = self.pos_embed
                cls_pos_embed = pos_embed[0, 0, :].unsqueeze(0).unsqueeze(1)
                other_pos_embed = pos_embed[0, 1:, :].unsqueeze(0).transpose(1, 2)
                P = int(other_pos_embed.size(2) ** 0.5)
                H = x.size(1) // W
                other_pos_embed = other_pos_embed.reshape(1, x.size(2), P, P)
                new_pos_embed = F.interpolate(other_pos_embed, size=(H, W), mode='nearest')
                new_pos_embed = new_pos_embed.flatten(2)
                new_pos_embed = new_pos_embed.transpose(1, 2)
                new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
                x = x + new_pos_embed
            else:
                x = x + self.pos_embed
            x = self.pos_drop(x)    # (b*t, 401, 1536)

        # Time Embeddings
        cls_tokens = x[:B, 0, :].unsqueeze(1)  # (b, 1, 1536)
        x = x[:, 1:]  # (b*t, 400, 1536)
        x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)    # (b*t, 400, 1536) -> (b*400, 2, 1536)
        if self.time_pos:
            # Resizing time embeddings in case they don't match
            if T != self.time_embed.size(1):
                time_embed = self.time_embed.transpose(1, 2)
                new_time_embed = F.interpolate(time_embed, size=(T), mode='nearest')
                new_time_embed = new_time_embed.transpose(1, 2)
                x = x + new_time_embed
            else:
                x = x + self.time_embed
        x = self.time_drop(x)   # (b*400, 2, 1536)
        x = rearrange(x, '(b n) t m -> b (n t) m', b=B, t=T)  # (b*400, 2, 1536) -> (b, 400*2, 1536)
        x = torch.cat((cls_tokens, x), dim=1)  # (b, 400*2+1, 1536)

        # Attention blocks
        for blk in self.blocks:
            x = blk(x, B, T, W)  # (b, 400*t+1, 1536)

        outs = Munch()
        sup = self.norm(x)
        outs.sup = sup

        # SSUP
        if self.use_drloc:
            x_last = x[:, 1:]  # B, L, C
            x_last = x_last.transpose(1, 2)  # [B, C, L]
            B, C, HWT = x_last.size()
            # H = W = int(math.sqrt(HWT))
            H = H // self.patch_size
            W = HWT // (H * T)
            x_last = x_last.view(B, C, H, W, T)  # [B, C, H, W]
            x_last = rearrange(x_last, 'b c h w t -> (b t) c h w')

            drloc_feats, deltaxy = self.drloc(x_last)
            outs.drloc = [drloc_feats]
            outs.deltaxy = [deltaxy]
            outs.plz = [H]  # plane size
        return outs


class PointDecoder(nn.Module):

    def __init__(self, num_heads, num_layers, num_point, embed_dim, query_pos):
        super(PointDecoder, self).__init__()
        self.query_pos = query_pos
        if self.query_pos:
            self.pos_embed = nn.Parameter(torch.zeros(num_point, embed_dim))  # (num_point, 1536)
        self.query_embed = nn.Parameter(torch.zeros(num_point, embed_dim))  # (num_point, 1536)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads)
        decoder_norm = nn.LayerNorm(embed_dim)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=num_layers, norm=decoder_norm)
        self.head = nn.Linear(embed_dim, 6)
        trunc_normal_(self.query_embed, std=.02)
        self.apply(init_weights_)

    def forward(self, x):
        # x: (b, 400*t+1, 1536)
        batch_size = x.shape[0]
        query_embed = self.query_embed.unsqueeze(1).repeat(1, batch_size, 1)  # (num_point, b, 1536)
        if self.query_pos:
            pos_embed = self.pos_embed.unsqueeze(1).repeat(1, batch_size, 1)  # (num_point, b, 1536)
            x = self.transformer_decoder(query_embed + pos_embed, x.permute(1, 0, 2))  # (num_points, b, 1536)
        else:
            x = self.transformer_decoder(query_embed, x.permute(1, 0, 2))  # (num_points, b, 1536)
        output = self.head(x)  # (num_points, b, 6)
        return output.permute(1, 0, 2)  # (b, num_points, 6)


class TR4TR(nn.Module):

    def __init__(self, batch_size=2, img_size=(320, 320, 6, 2), attention_type='patch', patch_size=16, num_point=2048,
                 num_layers=6, num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6), pretrained_model='', pretrained_model_path='',
                 drloc_mode='l1', sample_size=32, space_pos=True, time_pos=True, query_pos=True, **kwargs):
        """
        :param batch_size: train parameter
        :param img_size: the size of the image (height, width, channels, times)
        :param attention_type: patch
        :param patch_size: the size of the every crop image
        :param num_point: output point pairs numbers
        :param num_layers: layer number of transformer
        :param num_heads: head number of transformer
        :param mlp_ratio:
        :param qkv_bias: attention bias
        :param qk_scale: scaling factor of the attention
        :param drop_rate: dropout rate
        :param attn_drop_rate: attention drop rate
        :param drop_path_rate:
        :param norm_layer: layer norm
        :param pretrained_model: pre-trained model name
        :param pretrained_model_path: pre-trained model path
        :param drloc_mode: self-supervised item mode
        :param sample_size: self-supervised item parameter
        :param space_pos: w/o space position
        :param time_pos: w/o time position
        :param query_pos: w/o query position
        :param kwargs:
        """
        super(TR4TR, self).__init__()
        self.pretrained_model = pretrained_model

        if attention_type == 'patch':
            self.timeSformer = TimeSformer(batch_size=batch_size, img_size=img_size, patch_size=patch_size,
                                           num_layers=num_layers, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                           qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate,
                                           attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                                           drloc_mode=drloc_mode, sample_size=sample_size, norm_layer=norm_layer,
                                           space_pos=space_pos, time_pos=time_pos, **kwargs)

        embed_dim = self.timeSformer.embed_dim
        self.transformer_decoder = PointDecoder(num_heads, num_layers, num_point, embed_dim, query_pos)

        # load pre-training parameters
        if self.pretrained_model != '':
            pretrained_dict = torch.load(pretrained_model_path + pretrained_model, map_location='cpu')['state_dict']
            self.timeSformer.load_state_dict({k[19:]: v for k, v in pretrained_dict.items()}, strict=False)  # module.timesformer
            self.transformer_decoder.load_state_dict({k[27:]: v for k, v in pretrained_dict.items()}, strict=False)
            del pretrained_dict

    def forward(self, x):
        # x: B, H, W, C, T
        # if self.pretrained_model != '':
        #     x = self.cnn(x)
        outs = self.timeSformer(x)
        # sup: (b num_point, 6)  (source_x, source_y, source_z, target_x, target_y, target_z) the unit is m
        outs.sup = self.transformer_decoder(outs.sup)
        return outs

