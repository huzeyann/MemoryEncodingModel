from torch import nn, Tensor
from einops import rearrange
from typing import Dict

import torch
from config import AutoConfig

from timm.layers.norm import LayerNorm2d
from timm.models.convnext import ConvNeXtBlock
from timm.layers.mlp import Mlp


class SimpleConvBlocks(nn.Module):
    def __init__(
        self,
        in_chs,
        out_chs,
        depth=3,
        kernel_size=5,
        max_dim=1024,
        stride=1,
        padding="same",
        norm_layer=LayerNorm2d,
        act=nn.SiLU,
        groups=1,
        bias=False,
        conv1x1=False,
        reduce_dim=False,
        skip_connection=True,
    ):
        super(SimpleConvBlocks, self).__init__()

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * depth

        if reduce_dim:
            ch1 = min(in_chs, out_chs)
        else:
            ch1 = max(in_chs, out_chs)

        # ch1 = min(ch1, max_dim)

        layers = []
        self.reduce_dim = False
        if in_chs > max_dim:
            self.reduce_block = nn.Conv2d(in_chs, max_dim, 1, bias=False)
            in_chs = max_dim
            ch1 = max_dim
            self.reduce_dim = True

        # norm_shape = None
        # if norm_layer == nn.BatchNorm2d:
        #     norm_shape = ch1
        # if norm_layer == nn.LayerNorm:
        #     norm_shape = [ch1, 16, 16]  # not elegant

        for i in range(depth - 1):
            block = nn.Sequential(
                nn.Conv2d(
                    in_chs if i == 0 else ch1,
                    ch1,
                    kernel_size[i],
                    stride,
                    padding,
                    groups,
                    bias=bias,
                ),
                norm_layer(ch1),
                act(inplace=True),
            )
            layers.append(block)
        if not conv1x1:
            block = nn.Sequential(
                nn.Conv2d(
                    ch1, out_chs, kernel_size[-1], stride, padding, groups, bias=bias
                ),
                act(inplace=True),
            )
            layers.append(block)
        if conv1x1:
            block = nn.Sequential(
                nn.Conv2d(
                    ch1, ch1, kernel_size[-1], stride, padding, groups, bias=bias
                ),
                norm_layer(ch1),
                act(inplace=True),
                nn.Conv2d(ch1, out_chs, 1, bias=bias),
                act(inplace=True),
            )
            layers.append(block)

        self.block = nn.Sequential(*layers)

        self.skip_connection = skip_connection
        self.depth = depth

    def forward(self, x):
        if self.reduce_dim:
            x = self.reduce_block(x)

        for i, b in enumerate(self.block):
            x_prev = x
            x_next = b(x)
            if i < self.depth - 1:
                x = x_next + x_prev if self.skip_connection else x_next
            else:
                x = x_next
        return x


class ConvBlocks(nn.Module):
    def __init__(
        self,
        in_chs,
        out_chs,
        max_dim=1024,
        depth=3,
        kernel_size=5,
    ):
        super().__init__()

        dim = min(in_chs, max_dim)

        self.blocks = []
        for i in range(depth):
            _in_chs = in_chs if i == 0 else dim
            norm_layer = None  # defaults to LayerNorm
            # if i == depth - 1 and skip_last_norm:
            # norm_layer = nn.Identity
            self.blocks.append(
                ConvNeXtBlock(_in_chs, dim, kernel_size,
                              norm_layer=norm_layer),
            )
        self.blocks.append(nn.Conv2d(dim, out_chs, 3, padding="same"))
        self.blocks.append(nn.GELU())
        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x: Tensor):
        return self.blocks(x)


class DictConvBlocks(nn.Module):
    def __init__(
        self,
        layers=[5, 11, 17, 23],
        in_dims=[1024, 1024, 1024, 1024],
        out_dim=256,
        max_dim=1024,
        kernel_sizes=[5, 5, 5, 5],
        depths=[3, 3, 3, 3],
        block=ConvBlocks,
    ):
        super().__init__()

        self.blocks_dict = nn.ModuleDict()
        for i, layer in enumerate(layers):
            self.blocks_dict[str(layer)] = block(
                in_dims[i],
                out_dim,
                max_dim=max_dim,
                depth=depths[i],
                kernel_size=kernel_sizes[i],
            )

    def forward(self, x: Dict[str, Tensor]):
        for layer, block in self.blocks_dict.items():
            x[layer] = block(x[layer])
        return x


def build_conv_blocks(cfg: AutoConfig):
    return DictConvBlocks(
        layers=cfg.MODEL.BACKBONE.LAYERS,
        in_dims=cfg.MODEL.BACKBONE.FEATURE_DIMS,
        out_dim=cfg.MODEL.CONV_HEAD.WIDTH,
        max_dim=cfg.MODEL.CONV_HEAD.MAX_DIM,
        kernel_sizes=cfg.MODEL.CONV_HEAD.KERNEL_SIZES,
        depths=cfg.MODEL.CONV_HEAD.DEPTHS,
        block=SimpleConvBlocks if cfg.MODEL.CONV_HEAD.SIMPLE else ConvBlocks,
    )


class ClassTokenMLPs(nn.Module):
    def __init__(
        self,
        layers=[5, 11, 17, 23],
        in_dims=[1024, 1024, 1024, 1024],
        out_dim=256,
    ):
        super().__init__()

        self.mlp_dict = nn.ModuleDict()
        for i, layer in enumerate(layers):
            self.mlp_dict[str(layer)] = Mlp(
                in_features=in_dims[i], out_features=out_dim
            )

    def forward(self, x: Dict[str, Tensor]):
        for layer, mlp in self.mlp_dict.items():
            x[layer] = mlp(x[layer])
        return x


def build_class_token_mlp(cfg: AutoConfig):
    return ClassTokenMLPs(
        layers=cfg.MODEL.BACKBONE.LAYERS,
        in_dims=cfg.MODEL.BACKBONE.CLS_DIMS,
        out_dim=cfg.MODEL.CONV_HEAD.WIDTH,
    )


def build_class_token_mlp_prev(cfg: AutoConfig):
    return ClassTokenMLPs(
        layers=cfg.MODEL.BACKBONE_SMALL.LAYERS,
        in_dims=cfg.MODEL.BACKBONE_SMALL.CLS_DIMS,
        out_dim=cfg.MODEL.BACKBONE_SMALL.WIDTH,
    )

class PreviousFeatureMLPs(nn.Module):
    def __init__(
        self,
        feat_dim=1024,
        c_dim=256,
        t_dim=128,
        out_dim=256,
    ):
        super().__init__()

        self.mlp = Mlp(
            in_features=feat_dim + c_dim + t_dim,
            out_features=out_dim,
        )
        
    def forward(self, x: Tensor, c: Tensor, t: Tensor):
        bsz, tsz, _ = x.shape
        c = rearrange(c, "(b t) c -> b t c", b=bsz)
        t = rearrange(t, "(b t) c -> b t c", b=bsz)
        x = torch.cat([x, c, t], dim=-1)
        return self.mlp(x)

def build_prev_feat_mlp(cfg: AutoConfig):
    return PreviousFeatureMLPs(
        feat_dim=cfg.MODEL.PREV_FEAT.DIM,
        c_dim=cfg.MODEL.COND.DIM,
        t_dim=cfg.MODEL.BACKBONE_SMALL.T_DIM,
        out_dim=cfg.MODEL.BACKBONE_SMALL.WIDTH,
    )

class SubjectPreviousFrameCompress(nn.Module):
    def __init__(
        self,
        num_time_steps,
        in_width,
        merge_width,
        subject_list,
        hidden_ratio=4,
    ):
        super().__init__()
        self.subject_list = subject_list
        self.hidden_dim = int(merge_width * hidden_ratio)
        self.t = num_time_steps

        self.subject_layer = nn.ModuleDict()
        for subject in self.subject_list:
            self.subject_layer[subject] = nn.Sequential(
                nn.Linear(int(in_width * num_time_steps), self.hidden_dim),
                nn.GELU(),
            )
        self.merge_layer = Mlp(self.hidden_dim, out_features=merge_width)

    def forward(
        self,
        x: Tensor,  # [(B T), C]
        subject: str,
    ):
        x = rearrange(x, "b t c -> b (t c)", t=self.t)
        x = self.subject_layer[subject](x)
        x = self.merge_layer(x)
        return x


def build_prev_compress(cfg: AutoConfig):
    return SubjectPreviousFrameCompress(
        num_time_steps=cfg.DATASET.N_PREV_FRAMES-1,
        in_width=cfg.MODEL.BACKBONE_SMALL.WIDTH,
        merge_width=cfg.MODEL.BACKBONE_SMALL.MERGE_WIDTH,
        subject_list=cfg.DATASET.SUBJECT_LIST,
        hidden_ratio=4,
    )
