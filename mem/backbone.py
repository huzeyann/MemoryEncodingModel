# %%
import copy
from functools import partial
import os
from typing import Any, Dict, List, Optional, Tuple, Union
from einops import rearrange
from filelock import FileLock

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from config import AutoConfig
from registry import Registry

from torchvision.models import list_models, get_model
from torchvision.models.feature_extraction import (
    create_feature_extractor,
    get_graph_node_names,
)

import math

import open_clip

from open_clip.transformer import VisionTransformer, Transformer, ResidualAttentionBlock
from open_clip.timm_model import TimmModel

from timm.models.convnext import ConvNeXt

import torch.nn.functional as F

# from xformers.ops import memory_efficient_attention


BACKBONES = Registry()


class LoRALinearLayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()

        if rank > min(in_features, out_features):
            raise ValueError(
                f"LoRA rank {rank} must be less or equal than {min(in_features, out_features)}"
            )

        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        return up_hidden_states.to(orig_dtype)

    @property
    def weight(self):
        return self.up.weight @ self.down.weight

    @property
    def bias(self):
        return 0


class MonkeyLoRALinear(nn.Module):
    def __init__(self, fc: nn.Linear, rank=4, lora_scale=1):
        super().__init__()
        if rank > min(fc.in_features, fc.out_features):
            raise ValueError(
                f"LoRA rank {rank} must be less or equal than {min(fc.in_features, fc.out_features)}"
            )
        if not isinstance(fc, nn.Linear):
            raise ValueError(
                f"MonkeyLoRALinear only support nn.Linear, but got {type(fc)}"
            )

        self.fc = fc
        self.rank = rank
        self.lora_scale = lora_scale

        in_features = fc.in_features
        out_features = fc.out_features
        self.fc_lora = LoRALinearLayer(in_features, out_features, rank)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc(hidden_states) + self.lora_scale * self.fc_lora(
            hidden_states
        )
        return hidden_states

    @property
    def weight(self):
        return self.fc.weight + self.lora_scale * self.fc_lora.weight

    @property
    def bias(self):
        return self.fc.bias


class AdaLNZeroPatch(nn.Module):
    def __init__(self, embed_dim, d_c=64, adaln_scale=1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.d_c = d_c
        self.adaln_scale = adaln_scale

        # for condition (behavior data)
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(self.d_c, 6 * self.embed_dim, bias=False),
            nn.Tanh(),
        )

        nn.init.zeros_(self.adaLN_modulation[0].weight)

    def forward(self, c):
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = (
            self.adaLN_modulation(c) * self.adaln_scale
        ).chunk(6, dim=1)

        scale_msa = scale_msa + 1
        gate_msa = gate_msa + 1
        scale_mlp = scale_mlp + 1
        gate_mlp = gate_mlp + 1

        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLNLoRACLIPResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        block: ResidualAttentionBlock,
        rank=4,
        lora_scale=1.0,
        d_c=64,
        adaln_scale=1.0,
    ):
        super().__init__()
        self.block = block
        self.lora_scale = lora_scale
        self.rank = rank
        self.d_c = d_c

        self.embed_dim = self.block.attn.embed_dim

        ### these are nn.Parameter, can not be monkey-patched
        # patch qkv
        self.w_clone = self.block.attn.in_proj_weight.clone()
        self.w_clone.requires_grad_(False)
        self.attn_in_proj_lora = LoRALinearLayer(
            self.embed_dim, 3 * self.embed_dim, rank=rank
        )

        ### these are nn.Linear, can be monkey-patched
        self.block.attn.out_proj = MonkeyLoRALinear(
            self.block.attn.out_proj, rank=rank, lora_scale=lora_scale
        )
        self.block.mlp[0] = MonkeyLoRALinear(
            self.block.mlp[0], rank=rank, lora_scale=lora_scale
        )
        self.block.mlp[2] = MonkeyLoRALinear(
            self.block.mlp[2], rank=rank, lora_scale=lora_scale
        )

        # for condition (behavior data)
        self.adaLN = AdaLNZeroPatch(self.embed_dim, d_c=d_c, adaln_scale=adaln_scale)

    def forward(
        self,
        q_x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        # lora patch qkv
        self.block.attn.in_proj_weight.data = (
            self.w_clone.to(q_x.device)
            + self.lora_scale * self.attn_in_proj_lora.weight
        )

        # conditioning can be None
        bsz = q_x.shape[1]
        if c is None:
            c = torch.zeros(bsz, self.d_c, device=q_x.device, dtype=q_x.dtype)

        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN(c)

        # attention with adaLN, LoRA is applied to weight
        x = q_x + gate_msa.unsqueeze(0) * self.block.ls_1(
            self.block.attention(
                self.modulate(self.block.ln_1(q_x), shift_msa, scale_msa),
                attn_mask=attn_mask,
            )
        )
        x = x + gate_mlp.unsqueeze(0) * self.block.ls_2(
            self.block.mlp(self.modulate(self.block.ln_2(x), shift_mlp, scale_mlp))
        )

        return x

    @staticmethod
    def modulate(x, shift, scale):
        return x * scale.unsqueeze(0) + shift.unsqueeze(0)


def maxavg_globalpool2d(x):
    out = torch.cat([F.adaptive_avg_pool2d(x, 1), F.adaptive_max_pool2d(x, 1)], dim=1)
    out = out.squeeze(-1).squeeze(-1)
    return out


@BACKBONES.register("adaln_lora_clip_vit")
class AdaLNLoRACLIPViT(nn.Module):
    def __init__(
        self, lora_scale=1.0, rank=4, d_c=64, adaln_scale=1.0, ver="ViT-L-14", data='datacomp_xl_s13b_b90k', **kwargs
    ) -> None:
        super().__init__()

        model, _, _ = open_clip.create_model_and_transforms(
            ver, pretrained=data
        )
        self.vision_model: VisionTransformer = model.visual
        self.vision_model.requires_grad_(False)

        self.vision_model = self.inject_lora_and_adaln_clip_vit(
            self.vision_model,
            lora_scale=lora_scale,
            rank=rank,
            d_c=d_c,
            adaln_scale=adaln_scale,
        )

    @staticmethod
    def inject_lora_and_adaln_clip_vit(
        model: VisionTransformer, lora_scale=1.0, rank=4, d_c=64, adaln_scale=1.0
    ):
        transformer: Transformer = model.transformer
        for _i in range(len(transformer.resblocks)):
            block = transformer.resblocks[_i]
            lora_block = AdaLNLoRACLIPResidualAttentionBlock(
                block,
                rank=rank,
                lora_scale=lora_scale,
                d_c=d_c,
                adaln_scale=adaln_scale,
            )
            model.transformer.resblocks[_i] = lora_block
        return model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vision_model(x)

    def get_intermediate_layers(
        self,
        x,
        n: List[int] = [5, 11, 17, 23],
        c: Optional[torch.Tensor] = None,
        reshape=True,
        attn_mask=None,
    ):
        ##############################
        ### patchify ###
        ##############################

        # to patches - whether to use dual patchnorm - https://arxiv.org/abs/2302.01327v1
        if self.vision_model.input_patchnorm:
            # einops - rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)')
            x = x.reshape(
                x.shape[0],
                x.shape[1],
                self.vision_model.grid_size[0],
                self.vision_model.patch_size[0],
                self.vision_model.grid_size[1],
                self.vision_model.patch_size[1],
            )
            x = x.permute(0, 2, 4, 1, 3, 5)
            x = x.reshape(
                x.shape[0],
                self.vision_model.grid_size[0] * self.vision_model.grid_size[1],
                -1,
            )
            x = self.vision_model.patchnorm_pre_ln(x)
            x = self.vision_model.conv1(x)
        else:
            x = self.vision_model.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat(
            [
                self.vision_model.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.vision_model.positional_embedding.to(x.dtype)

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        x = self.vision_model.patch_dropout(x)
        x = self.vision_model.ln_pre(x)

        ##############################
        ### transformer ###
        ##############################

        x = x.permute(1, 0, 2)  # NLD -> LND

        output_dict = {}
        cls_dict = {}
        for i, r in enumerate(self.vision_model.transformer.resblocks):
            x = r(x, c=c, attn_mask=attn_mask)  # [1+p**2, B, D]
            if i not in n:
                continue
            x_save = x.clone()
            if reshape == True:
                x_save = x_save[1:, :, :]  # [p**2, B, D]
                p = int(np.sqrt(x_save.shape[0]))
                x_save = rearrange(x_save, "(p1 p2) b d -> b d p1 p2", p1=p, p2=p)
            output_dict[str(i)] = x_save
            if i == len(self.vision_model.transformer.resblocks) - 1:
                cls_dict[str(i)] = x[0, :, :]  # [B, D]
            else:
                cls_dict[str(i)] = maxavg_globalpool2d(x_save)

        return output_dict, cls_dict


@BACKBONES.register("clip_vit_l")
def clip_vit_l(**kwargs):
    ver='ViT-L-14'
    data='datacomp_xl_s13b_b90k'
    return AdaLNLoRACLIPViT(ver=ver, data=data, **kwargs)

@BACKBONES.register("clip_vit_b")
def clip_vit_l(**kwargs):
    ver='ViT-B-16'
    data='datacomp_l_s1b_b8k'
    return AdaLNLoRACLIPViT(ver=ver, data=data, **kwargs)

@BACKBONES.register("clip_vit_s")
def clip_vit_l(**kwargs):
    ver='ViT-B-32'
    data='datacomp_m_s128m_b4k'
    return AdaLNLoRACLIPViT(ver=ver, data=data, **kwargs)

# @BACKBONES.register("eva_clip_l")
# def eva_clip_l(**kwargs):
#     ver='hf-hub:timm/eva02_large_patch14_clip_224.merged2b_s4b_b131k'
#     data=None
#     return AdaLNLoRACLIPViT(ver=ver, data=data, **kwargs)

# @BACKBONES.register("eva_clip_b")
# def eva_clip_b(**kwargs):
#     ver='hf_hub:timm/eva02_base_patch14_224.mim_in22k'
#     data=None
#     return AdaLNLoRACLIPViT(ver=ver, data=data, **kwargs)

from timm.models.convnext import ConvNeXtStage, ConvNeXt, ConvNeXtBlock


class AdaLNCovNeXtBlock(nn.Module):
    def __init__(self, block: ConvNeXtBlock, d_c=64, adaln_scale=1.0) -> None:
        super().__init__()
        self.block = block
        self.d_c = d_c
        self.embed_dim = block.norm.weight.shape[0]

        # for condition (behavior data)
        self.adaLN = AdaLNZeroPatch(self.embed_dim, d_c=d_c, adaln_scale=adaln_scale)

    def forward(self, x, c: Optional[torch.Tensor] = None):
        # conditioning can be None
        bsz = x.shape[0]
        if c is None:
            c = torch.zeros(bsz, self.d_c, device=x.device, dtype=x.dtype)

        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN(c)

        shortcut = x
        x = self.block.conv_dw(x)
        if self.block.use_conv_mlp:
            x = self.block.norm(x)
            x = self.modulate(x, shift_mlp, scale_mlp)
            x = self.block.mlp(x)
            x = x * gate_mlp.unsqueeze(1).unsqueeze(1)
        else:
            x = x.permute(0, 2, 3, 1)
            x = self.block.norm(x)
            x = self.modulate(x, shift_mlp, scale_mlp)
            x = self.block.mlp(x)
            x = x * gate_mlp.unsqueeze(1).unsqueeze(1)
            x = x.permute(0, 3, 1, 2)
        if self.block.gamma is not None:
            x = x.mul(self.block.gamma.reshape(1, -1, 1, 1))

        x = self.block.drop_path(x) + self.block.shortcut(shortcut)
        return x

    @staticmethod
    def modulate(x, shift, scale):
        return x * scale.unsqueeze(1).unsqueeze(1) + shift.unsqueeze(1).unsqueeze(1)


class AdaLNConvNeXtStage(nn.Module):
    def __init__(self, stage: ConvNeXtStage):
        super().__init__()
        self.stage = stage

    def forward(self, x, c: Optional[torch.Tensor] = None):
        x = self.stage.downsample(x)
        # x = self.stage.blocks(x)
        for block in self.stage.blocks:
            x = block(x, c=c)
        return x


@BACKBONES.register("adaln_lora_clip_convnext")
class AdaLNLoRACLIPConvNeXt(nn.Module):
    def __init__(
        self, lora_scale=1.0, rank=4, d_c=64, adaln_scale=1.0, ver='convnext_xxlarge', data='laion2b_s34b_b82k_augreg_soup', **kwargs
    ) -> None:
        super().__init__()

        model, _, preprocess = open_clip.create_model_and_transforms(
            ver,
            pretrained=data,
        )
        self.vision_model: ConvNeXt = model.visual.trunk
        self.vision_model.requires_grad_(False)

        self.vision_model = self.inject_lora_and_adaln_clip_convnext(
            self.vision_model,
            lora_scale=lora_scale,
            rank=rank,
            d_c=d_c,
            adaln_scale=adaln_scale,
        )

    @staticmethod
    def inject_lora_and_adaln_clip_convnext(
        model: ConvNeXt, lora_scale=1.0, rank=4, d_c=64, adaln_scale=1.0
    ):
        for _stage_i in range(len(model.stages)):
            stage = model.stages[_stage_i]
            for _block_i in range(len(stage.blocks)):
                block = stage.blocks[_block_i]
                block.mlp.fc1 = MonkeyLoRALinear(
                    block.mlp.fc1, rank=rank, lora_scale=lora_scale
                )
                block.mlp.fc2 = MonkeyLoRALinear(
                    block.mlp.fc2, rank=rank, lora_scale=lora_scale
                )
                block = AdaLNCovNeXtBlock(block, d_c=d_c, adaln_scale=adaln_scale)
                model.stages[_stage_i].blocks[_block_i] = block
            model.stages[_stage_i] = AdaLNConvNeXtStage(model.stages[_stage_i])
        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vision_model(x)

    def get_intermediate_layers(
        self,
        x,
        n: List[str] = [0, 1, 2, 3],
        c: Optional[torch.Tensor] = None,
    ):
        x = self.vision_model.stem(x)

        output_dict = {}
        cls_dict = {}
        for i, stage in enumerate(self.vision_model.stages):
            x = stage(x, c=c)
            if i not in n:
                continue
            output_dict[str(i)] = x
            fake_cls_token = torch.cat(
                [F.adaptive_avg_pool2d(x, 1), F.adaptive_max_pool2d(x, 1)], dim=1
            )
            cls_dict[str(i)] = fake_cls_token
        return output_dict, cls_dict

"""
('convnext_base', 'laion400m_s13b_b51k'),
('convnext_base_w', 'laion2b_s13b_b82k'),
('convnext_base_w', 'laion2b_s13b_b82k_augreg'),
('convnext_base_w', 'laion_aesthetic_s13b_b82k'),
('convnext_base_w_320', 'laion_aesthetic_s13b_b82k'),
('convnext_base_w_320', 'laion_aesthetic_s13b_b82k_augreg'),
('convnext_large_d', 'laion2b_s26b_b102k_augreg'),
('convnext_large_d_320', 'laion2b_s29b_b131k_ft'),
('convnext_large_d_320', 'laion2b_s29b_b131k_ft_soup'),
('convnext_xxlarge', 'laion2b_s34b_b82k_augreg'),
('convnext_xxlarge', 'laion2b_s34b_b82k_augreg_rewind'),
('convnext_xxlarge', 'laion2b_s34b_b82k_augreg_soup'),
"""
@BACKBONES.register("clip_convnext_xxlarge")
def clip_convnext_xxlarge(**kwargs):
    ver='convnext_xxlarge'
    data='laion2b_s34b_b82k_augreg_soup'
    return AdaLNLoRACLIPConvNeXt(ver=ver, data=data, **kwargs)

@BACKBONES.register("clip_convnext_large")
def clip_convnext_large(**kwargs):
    ver='convnext_large_d_320'
    data='laion2b_s29b_b131k_ft_soup'
    return AdaLNLoRACLIPConvNeXt(ver=ver, data=data, **kwargs)

@BACKBONES.register("clip_convnext_base")
def clip_convnext_large(**kwargs):
    ver='convnext_base_w_320'
    data='laion_aesthetic_s13b_b82k_augreg'
    return AdaLNLoRACLIPConvNeXt(ver=ver, data=data, **kwargs)

from dinov2.models.vision_transformer import DinoVisionTransformer

from dinov2.layers.attention import MemEffAttention, Attention
from dinov2.layers.block import NestedTensorBlock, Block
from dinov2.layers.block import drop_add_residual_stochastic_depth


class AdaLNDiNOBlock(nn.Module):
    def __init__(self, block: Block, d_c=64, adaln_scale=1.0):
        super().__init__()
        self.block = block
        self.embed_dim = block.norm1.weight.shape[0]
        self.d_c = d_c

        self.adaLN = AdaLNZeroPatch(self.embed_dim, d_c=d_c, adaln_scale=adaln_scale)

    def forward(self, x, c: Optional[torch.Tensor] = None):
        # conditioning can be None
        bsz = x.shape[0]
        if c is None:
            c = torch.zeros(bsz, self.d_c, device=x.device, dtype=x.dtype)

        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN(c)

        def attn_residual_func(x: Tensor) -> Tensor:
            return self.block.ls1(
                self.block.attn(
                    self.modulate(self.block.norm1(x), shift_msa, scale_msa)
                )
            ) * gate_msa.unsqueeze(1)

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.block.ls2(
                self.block.mlp(self.modulate(self.block.norm2(x), shift_mlp, scale_mlp))
            ) * gate_mlp.unsqueeze(1)

        if self.block.training and self.block.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.block.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.block.sample_drop_ratio,
            )
        elif self.block.training and self.block.sample_drop_ratio > 0.0:
            x = x + self.block.drop_path1(attn_residual_func(x))
            x = x + self.block.drop_path1(ffn_residual_func(x))  # FIXME: drop_path2
        else:
            x = x + attn_residual_func(x)
            x = x + ffn_residual_func(x)
        return x

    @staticmethod
    def modulate(x, shift, scale):
        return x * scale.unsqueeze(1) + shift.unsqueeze(1)


@BACKBONES.register("adaln_lora_dinov2_vit")
class AdaLNLoRADiNOv2ViT(nn.Module):
    def __init__(
        self, lora_scale=1.0, rank=4, d_c=64, adaln_scale=1.0, ver='dinov2_vitl14', **kwargs
    ) -> None:
        super().__init__()

        vision_model = torch.hub.load("facebookresearch/dinov2", ver)
        self.vision_model: DinoVisionTransformer = vision_model
        self.vision_model.requires_grad_(False)

        self.vision_model = self.inject_lora_and_adaln_dinov2(
            self.vision_model,
            lora_scale=lora_scale,
            rank=rank,
            d_c=d_c,
            adaln_scale=adaln_scale,
        )

    @staticmethod
    def inject_lora_and_adaln_dinov2(
        model: DinoVisionTransformer, lora_scale=1.0, rank=4, d_c=64, adaln_scale=1.0
    ):
        for _i in range(len(model.blocks)):
            block: Block = model.blocks[_i]
            attn: Attention = block.attn
            block.attn.qkv = MonkeyLoRALinear(
                attn.qkv, rank=rank, lora_scale=lora_scale
            )
            block.attn.proj = MonkeyLoRALinear(
                attn.proj, rank=rank, lora_scale=lora_scale
            )
            block.mlp.fc1 = MonkeyLoRALinear(
                block.mlp.fc1, rank=rank, lora_scale=lora_scale
            )
            block.mlp.fc2 = MonkeyLoRALinear(
                block.mlp.fc2, rank=rank, lora_scale=lora_scale
            )
            model.blocks[_i] = AdaLNDiNOBlock(block, d_c=d_c, adaln_scale=adaln_scale)
        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vision_model(x)

    def get_intermediate_layers(
        self,
        x,
        n: List[str] = [0, 1, 2, 3],
        c: Optional[torch.Tensor] = None,
        reshape=True,
        masks=None,
    ):
        x = self.vision_model.prepare_tokens_with_masks(x, masks)

        output_dict = {}
        cls_dict = {}
        for i, blk in enumerate(self.vision_model.blocks):
            x = blk(x, c=c)
            if i not in n:
                continue
            saved_x = x.clone()
            if reshape:
                saved_x = saved_x[:, 1:, :]  # remove cls token, [B, N, C]
                p = int(np.sqrt(saved_x.shape[1]))
                saved_x = rearrange(saved_x, "b (p1 p2) c -> b c p1 p2", p1=p, p2=p)
            output_dict[str(i)] = saved_x
            if i == len(self.vision_model.blocks) - 1:
                cls_dict[str(i)] = x[:, 0, :]  # [B, C]
            else:
                cls_dict[str(i)] = maxavg_globalpool2d(saved_x)
        return output_dict, cls_dict

@BACKBONES.register("dinov2_vit_l")
def dinov2_vit_l(**kwargs):
    ver='dinov2_vitl14'
    return AdaLNLoRADiNOv2ViT(ver=ver, **kwargs)

@BACKBONES.register("dinov2_vit_b")
def dinov2_vit_b(**kwargs):
    ver='dinov2_vitb14'
    return AdaLNLoRADiNOv2ViT(ver=ver, **kwargs)

@BACKBONES.register("dinov2_vit_s")
def dinov2_vit_s(**kwargs):
    ver='dinov2_vits14'
    return AdaLNLoRADiNOv2ViT(ver=ver, **kwargs)

def clean_state_dict(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if ".module." in k:
            k = k.replace(".module.", ".")
        new_state_dict[k] = v
    return new_state_dict


def build_backbone(cfg: AutoConfig):
    # home = os.path.expanduser("~")
    # lock_path = os.path.join(home, ".cache", "download.lock")
    # with FileLock(lock_path):
    return BACKBONES[cfg.MODEL.BACKBONE.NAME](
        lora_scale=cfg.MODEL.BACKBONE.LORA.SCALE,
        rank=cfg.MODEL.BACKBONE.LORA.RANK,
        d_c=cfg.MODEL.COND.DIM,
        adaln_scale=cfg.MODEL.BACKBONE.ADAPTIVE_LN.SCALE,
    )
    
def build_backbone_prev(cfg: AutoConfig):
    return BACKBONES[cfg.MODEL.BACKBONE_SMALL.NAME](
        lora_scale=cfg.MODEL.BACKBONE_SMALL.LORA.SCALE,
        rank=cfg.MODEL.BACKBONE_SMALL.LORA.RANK,
        d_c=cfg.MODEL.COND.DIM,
        adaln_scale=cfg.MODEL.BACKBONE_SMALL.ADAPTIVE_LN.SCALE,
    )
    
class SubjectTimeEmbed(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    Each subject is running at a different clock speed, so we need to a subject-layer
    """
    def __init__(self, hidden_size, subject_list, frequency_embedding_size=256):
        super().__init__()
        self.subject_list = subject_list
        self.subject_layers = nn.ModuleDict()
        self.frequency_embedding_size = frequency_embedding_size

        for subject in subject_list:
            self.subject_layers[subject] = nn.Linear(frequency_embedding_size, hidden_size, bias=True)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, subject):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.subject_layers[subject](t_freq)
        t_emb = self.mlp(t_emb)
        return t_emb

def build_time_emd(cfg: AutoConfig):
    return SubjectTimeEmbed(
        hidden_size=cfg.MODEL.BACKBONE_SMALL.T_DIM,
        subject_list=cfg.DATASET.SUBJECT_LIST,
    )
    

# %%
# if __name__ == "__main__":
#     from config_utils import get_cfg_defaults

#     visual = AdaLNLoRACLIPViT(rank=4, d_c=1)

#     # out_dict = visual(torch.randn(3, 3, 224, 224))
#     out_dict = visual.get_intermediate_layers(
#         torch.randn(3, 3, 224, 224), [0, 1, 2, 3, 4, 5, 6, 7, 8]
#     )[0]

#     for k, v in out_dict.items():
#         print(k, v.shape)

#     import torch
#     from PIL import Image
#     import open_clip

#     # model, _, preprocess = open_clip.create_model_and_transforms(
#     #     "ViT-L-14", pretrained="datacomp_xl_s13b_b90k"
#     # )
#     # model, _, preprocess = open_clip.create_model_and_transforms(
#     #     "ViT-L-14", pretrained="eva02_large_patch14_clip_224"
#     # )
    
#     model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:timm/eva02_large_patch14_clip_224.merged2b_s4b_b131k')
#     tokenizer = open_clip.get_tokenizer('hf-hub:timm/eva02_large_patch14_clip_224.merged2b_s4b_b131k')
#     preprocess = preprocess_val
    
#     print(preprocess)
#     tokenizer = open_clip.get_tokenizer("ViT-L-14")

#     # verify that the patched model is working
#     model.visual = visual

#     image = preprocess(Image.open("CLIP.png")).unsqueeze(0)
#     text = tokenizer(["a diagram", "a dog", "a cat"])

#     with torch.no_grad(), torch.cuda.amp.autocast():
#         image_features = model.encode_image(image)
#         text_features = model.encode_text(text)
#         image_features /= image_features.norm(dim=-1, keepdim=True)
#         text_features /= text_features.norm(dim=-1, keepdim=True)

#         text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

#     print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]

#     print(sum(p.numel() for p in visual.parameters() if p.requires_grad))

# if __name__ == "__main__":
#     model = eva_clip_l()

# if __name__ == "__main__":
#     import torch
#     from PIL import Image
#     import open_clip

#     model, _, preprocess = open_clip.create_model_and_transforms(
#         "convnext_xxlarge",
#         pretrained="laion2b_s34b_b82k_augreg_soup",
#     )
#     tokenizer = open_clip.get_tokenizer("convnext_xxlarge")

#     # patch
#     lora_trunk = AdaLNLoRACLIPConvNeXt(rank=4, d_c=1)
#     model.visual.trunk = lora_trunk

#     image = preprocess(Image.open("CLIP.png")).unsqueeze(0)
#     print(preprocess)
#     text = tokenizer(["a diagram", "a dog", "a cat"])

#     with torch.no_grad(), torch.cuda.amp.autocast():
#         image_features = model.encode_image(image)
#         text_features = model.encode_text(text)
#         image_features /= image_features.norm(dim=-1, keepdim=True)
#         text_features /= text_features.norm(dim=-1, keepdim=True)

#         text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

#     print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]

#     out_dict = lora_trunk.get_intermediate_layers(
#         torch.randn(3, 3, 256, 256), [0, 1, 2, 3]
#     )

#     for k, v in out_dict.items():
#         print(k, v.shape)

#     print(sum(p.numel() for p in lora_trunk.parameters() if p.requires_grad))

# if __name__ == "__main__":
#     dinov2_vitl14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")

#     # patch
#     patched_model = AdaLNLoRADiNOv2ViT(rank=4, d_c=1)
#     patched_model = patched_model.cuda()

#     out_dict, cls_dict = patched_model.get_intermediate_layers(
#         torch.randn(3, 3, 224, 224).cuda(), [0, 8, 23]
#     )

#     for k, v in out_dict.items():
#         print(k, v.shape, cls_dict[k].shape)


def get_shape(model, input_size, n=[5, 11]):
    model = BACKBONES[model]()
    model.eval()
    model = model.cuda()
    input = torch.randn(1, 3, input_size, input_size).cuda()
    out_dict, cls_dict = model.get_intermediate_layers(input, n)
    for k, v in out_dict.items():
        print(k, v.shape, cls_dict[k].shape)
    
    return model

BACKBONEC = {
    'clip_vit_l': (224, [5, 11, 17, 23], [1024, 1024, 1024, 1024], [2048, 2048, 2048, 1024]),
    'clip_vit_b': (224, [2, 5, 8, 11], [768, 768, 768, 768], [1536, 1536, 1536, 768]),
    'clip_vit_s': (224, [2, 5, 8, 11], [768, 768, 768, 768], [1536, 1536, 1536, 768]),
    'dinov2_vit_l': (224, [5, 11, 17, 23], [1024, 1024, 1024, 1024], [2048, 2048, 2048, 1024]),
    'dinov2_vit_b': (224, [2, 5, 8, 11], [768, 768, 768, 768], [1536, 1536, 1536, 768]),
    'dinov2_vit_s': (224, [2, 5, 8, 11], [384, 384, 384, 384], [768, 768, 768, 384]),
}