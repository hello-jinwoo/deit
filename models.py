# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import numpy as np
import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_


__all__ = [
    'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
    'deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224',
    'deit_base_distilled_patch16_224', 'deit_base_patch16_384',
    'deit_base_distilled_patch16_384',
]

""" sinusoid position embedding """
def get_sinusoid_encoding_table(n_seq, d_hidn):
    def cal_angle(position, i_hidn):
        return position / 10000 ** (2 * (i_hidn // 2) / d_hidn)
    def get_posi_angle_vec(position):
        return [cal_angle(position, i_hidn) for i_hidn in range(d_hidn)]

    sinusoid_table = torch.tensor([get_posi_angle_vec(i_seq) for i_seq in range(n_seq)])
    sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])  # even index sin 
    sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])  # odd index cos

    return sinusoid_table

def get_area_encoding(num_patches, encoding_dim=192, mode='aaud', n_extra_tokens=1, img_size=224):
    assert mode in ['aaud', 'naive']
    num_patches_per_row = int(num_patches ** 0.5)
    patch_pos = torch.arange(num_patches_per_row, dtype=float) / num_patches_per_row # [0, 1/14, 2/14, ... 13/14]
    patch_pos_pair = torch.cartesian_prod(patch_pos, patch_pos) # [[0, 0], [0, 1/14], ... [n/14, m/14], ... [13/14, 13/14]]
    patch_area_pair = torch.zeros(len(patch_pos_pair), 4)
    patch_area_pair[:, 0] = patch_pos_pair[:, 0]
    patch_area_pair[:, 1] = patch_pos_pair[:, 0] + 1 / num_patches_per_row
    patch_area_pair[:, 2] = patch_pos_pair[:, 1]
    patch_area_pair[:, 3] = patch_pos_pair[:, 1] + 1 / num_patches_per_row
    pos_embed = nn.Parameter(torch.zeros(1, num_patches + n_extra_tokens, encoding_dim), requires_grad=False)
    if mode == 'aaud':
        pos_embed[0, n_extra_tokens:, :] = get_aaud_for_patch(patch_area_pair, encoding_dim)
    if mode == 'naive':
        patch_size = img_size // num_patches
        sin_table = get_sinusoid_encoding_table(img_size ** 2, encoding_dim)
        sin_table_2d = torch.reshape(sin_table, (img_size, img_size, encoding_dim))
        pos_embed[0, n_extra_tokens:, :] = get_naive_ae_for_patch((patch_area_pair * img_size).int(),
                                                                   encoding_dim, 
                                                                   sin_table_2d)

    return pos_embed

def get_aaud_for_patch(pos, encoding_dim=192):
    """
    pos_info : (num_of_patches, 4)
    ae : (num_of_patches, encoding_dim)
    """
    pos -= 0.5
    x_start, x_end, y_start, y_end = pos[:, 0], pos[:, 1], pos[:, 2], pos[:, 3]
    x_start, x_end, y_start, y_end = x_start[:, None], x_end[:, None], y_start[:, None], y_end[:, None]

    # IN PROGRESS: experiments on scale of coefficient
    # # scale_v0
    # x_coeff = 1 / ((x_end - x_start) * 4 * np.pi ** 2)
    # y_coeff = 1 / ((y_end - y_start) * 4 * np.pi ** 2)
    # scale_v1
    # x_coeff = 1 / ((x_end - x_start) * 4)
    # y_coeff = 1 / ((y_end - y_start) * 4)
    # scale_v2
    # x_coeff = 1 / ((x_end - x_start) * 4 * np.pi)
    # y_coeff = 1 / ((y_end - y_start) * 4 * np.pi)
    # # scale_v3
    # x_coeff = 1 / ((x_end - x_start) * np.pi ** 2)
    # y_coeff = 1 / ((y_end - y_start) * np.pi ** 2)
    # scale_v5
    x_coeff = 1 / ((x_end - x_start) * np.pi)
    y_coeff = 1 / ((y_end - y_start) * np.pi)
    # # scale_v6
    # x_coeff = 1 / (x_end - x_start)
    # y_coeff = 1 / (y_end - y_start)
    
    x_theta_1 = np.pi * x_start
    x_theta_2 = np.pi * x_end
    y_theta_1 = np.pi * y_start
    y_theta_2 = np.pi * y_end

    m = torch.arange(encoding_dim // 8, dtype=pos.dtype, device=pos.device)[None, :] + 1 # degrees for fourier series 
    x_a_m_1 = x_coeff * torch.sin(m * x_theta_1) / m
    x_a_m_2 = x_coeff * torch.sin(m * x_theta_2) / m
    x_b_m_1 = x_coeff * torch.cos(m * x_theta_1) / m
    x_b_m_2 = x_coeff * torch.cos(m * x_theta_2) / m
    y_a_m_1 = y_coeff * torch.sin(m * y_theta_1) / m
    y_a_m_2 = y_coeff * torch.sin(m * y_theta_2) / m
    y_b_m_1 = y_coeff * torch.cos(m * y_theta_1) / m
    y_b_m_2 = y_coeff * torch.cos(m * y_theta_2) / m

    # # scale_v7
    # x_a_m = (x_a_m + 0.15) / 0.3
    # x_b_m = (x_b_m + 0.15) / 0.3
    # y_a_m = (y_a_m + 0.15) / 0.3
    # y_b_m = (y_b_m + 0.15) / 0.3

    ae = torch.cat([x_a_m_1, x_a_m_2, x_b_m_1, x_b_m_2, y_a_m_1, y_a_m_2, y_b_m_1, y_b_m_2], dim=-1)

    return ae

def get_naive_ae_for_patch(pos, encoding_dim=192, sin_table_2d=None):
    """
    pos_info : (num_of_patches, 4)
    sin_table_2d = (img_size, img_size, encoding_dim)
    ae : (num_of_patches, encoding_dim)
    """
    x_start, x_end, y_start, y_end = pos[:, 0], pos[:, 1], pos[:, 2], pos[:, 3]

    # TODO: parallelize
    ae = torch.zeros(len(pos), encoding_dim)
    for i in range(len(pos)):
        ae[i] = torch.mean(sin_table_2d[x_start[i]: x_end[i], y_start[i]: y_end[i]], axis=(0, 1))

    return ae


class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        if self.training:
            return x, x_dist
        else:
            # during inference, return the average of both classifier predictions
            return (x + x_dist) / 2


# default
@register_model
def deit_tiny_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


# without pos embed
@register_model
def deit_tiny_patch16_224_without_pos(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()

    # without pos_embed (implemented as fixed zero embedding)
    num_patches = model.patch_embed.num_patches
    model.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, model.embed_dim), requires_grad=False)
    
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


# with sinusoidal pos embed
@register_model
def deit_tiny_patch16_224_with_sin(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()

    # sinusoidal positional embedding
    num_patches = model.patch_embed.num_patches
    pos_encoding = get_sinusoid_encoding_table(num_patches + 1, model.embed_dim)
    pos_emb = nn.Parameter(pos_encoding[None, ...], requires_grad=False)
    model.pos_embed = pos_emb

    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


# with aaud area embed
@register_model
def deit_tiny_patch16_224_with_aaud(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()

    # area encoding
    num_patches = model.patch_embed.num_patches
    model.pos_embed = get_area_encoding(num_patches, 
                                       model.embed_dim, 
                                       mode='aaud',
                                       n_extra_tokens=1)

    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


# with naive area embed
@register_model
def deit_tiny_patch16_224_with_naive_ae(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()

    # area encoding
    num_patches = model.patch_embed.num_patches
    model.pos_embed = get_area_encoding(num_patches, 
                                        model.embed_dim, 
                                        mode='naive', 
                                        n_extra_tokens=1,
                                        img_size=224)

    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


# with sin and naive area embed
@register_model
def deit_tiny_patch16_224_with_sin_and_naive(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()

    num_patches = model.patch_embed.num_patches
    # sinusoidal encoding
    sin_pos_encoding = get_sinusoid_encoding_table(num_patches + 1, model.embed_dim)
    model.pos_embed = nn.Parameter(sin_pos_encoding[None, ...], requires_grad=False)
    # area encoding    
    model.pos_embed += get_area_encoding(num_patches, 
                                         model.embed_dim, 
                                         mode='naive', 
                                         n_extra_tokens=1,
                                         img_size=224)
    # model.pos_embed /= 2.

    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


# with sin and aaud area embed
@register_model
def deit_tiny_patch16_224_with_sin_and_aaud(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()

    num_patches = model.patch_embed.num_patches
    # sinusoidal encoding
    sin_pos_encoding = get_sinusoid_encoding_table(num_patches + 1, model.embed_dim)
    model.pos_embed = nn.Parameter(sin_pos_encoding[None, ...], requires_grad=False)
    # area encoding    
    model.pos_embed += get_area_encoding(num_patches, 
                                        model.embed_dim, 
                                        mode='aaud',
                                        n_extra_tokens=1)
    # model.pos_embed /= 2.

    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_small_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


# without pos embed
@register_model
def deit_small_patch16_224_without_pos(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()

    # without pos_embed (implemented as fixed zero embedding)
    num_patches = model.patch_embed.num_patches
    model.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, model.embed_dim), requires_grad=False)

    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


# with sinusoidal pos embed
@register_model
def deit_small_patch16_224_with_sin(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()

    # sinusoidal positional embedding
    num_patches = model.patch_embed.num_patches
    pos_encoding = get_sinusoid_encoding_table(num_patches + 1, model.embed_dim)
    pos_emb = nn.Parameter(pos_encoding[None, ...], requires_grad=False)
    model.pos_embed = pos_emb

    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


# with naive area embed
@register_model
def deit_small_patch16_224_with_naive(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()

    # area encoding
    num_patches = model.patch_embed.num_patches
    model.pos_embed = get_area_encoding(num_patches, 
                                        model.embed_dim, 
                                        mode='naive', 
                                        n_extra_tokens=1,
                                        img_size=224)

    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


# with aaud area embed
@register_model
def deit_small_patch16_224_with_aaud(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()

    # area encoding
    num_patches = model.patch_embed.num_patches
    model.pos_embed = get_area_encoding(num_patches, 
                                       model.embed_dim, 
                                       mode='aaud',
                                       n_extra_tokens=1)

    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


# without pos embed
@register_model
def deit_base_patch16_224_without_pos(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    # without pos_embed (implemented as fixed zero embedding)
    num_patches = model.patch_embed.num_patches
    model.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, model.embed_dim), requires_grad=False)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


# with sinusoidal pos embed
@register_model
def deit_base_patch16_224_with_sin(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()

    # sinusoidal positional embedding
    num_patches = model.patch_embed.num_patches
    pos_encoding = get_sinusoid_encoding_table(num_patches + 1, model.embed_dim)
    pos_emb = nn.Parameter(pos_encoding[None, ...], requires_grad=False)
    model.pos_embed = pos_emb

    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


# with naive area embed
@register_model
def deit_base_patch16_224_with_naive(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()

    # area encoding
    num_patches = model.patch_embed.num_patches
    model.pos_embed = get_area_encoding(num_patches, 
                                        model.embed_dim, 
                                        mode='naive', 
                                        n_extra_tokens=1,
                                        img_size=224)

    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


# with aaud area embed
@register_model
def deit_base_patch16_224_with_aaud(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()

    # area encoding
    num_patches = model.patch_embed.num_patches
    model.pos_embed = get_area_encoding(num_patches, 
                                       model.embed_dim, 
                                       mode='aaud',
                                       n_extra_tokens=1)

    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_tiny_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_small_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_distilled_patch16_384(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model
