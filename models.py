from functools import partial
import logging
from torch import nn, Tensor
from einops import rearrange, repeat
from typing import Dict, Optional, Union

import torch
import torch.nn.functional as F
from config import AutoConfig

from backbone import (
    SubjectTimeEmbed,
    build_backbone,
    AdaLNLoRADiNOv2ViT,
    AdaLNLoRACLIPViT,
    AdaLNLoRACLIPConvNeXt,
    build_backbone_prev,
    build_time_emd,
)
from blocks import (
    PreviousFeatureMLPs,
    SubjectPreviousFrameCompress,
    build_class_token_mlp_prev,
    build_conv_blocks,
    build_class_token_mlp,
    DictConvBlocks,
    ClassTokenMLPs,
    build_prev_compress,
    build_prev_feat_mlp,
)
from behav_embed import build_behavior_embed, SubjectBehaviorEmbed
from topyneck import (
    build_coords_mlp,
    CachedCoordsMLP,
    build_voxelouts_weight,
    CoordsMLPLinearWeight,
    VoxelNonShareLinearWeight,
)

from timm.layers.mlp import Mlp
from timm.layers.norm import LayerNorm


def build_each_subject(fn, subject_list):
    return nn.ModuleDict({subject: fn() for subject in subject_list})


def _stack(d):
    return torch.stack(list(d.values()), dim=-1)


class MemVoxelWiseEncodingModel(nn.Module):
    def __init__(
        self,
        cfg: AutoConfig,
        n_voxel_dict: Dict[str, int],
    ):
        super().__init__()
        self.subject_list = cfg.DATASET.SUBJECT_LIST
        if list(n_voxel_dict.keys()) != self.subject_list:
            logging.warning(
                f"subjects in config and voxel dict are not matched, using voxel dict"
            )
            self.subject_list = list(n_voxel_dict.keys())
        self.layers = cfg.MODEL.BACKBONE.LAYERS
        self.layers_small = cfg.MODEL.BACKBONE_SMALL.LAYERS
        self.n_layers = len(self.layers)
        r = cfg.MODEL.WIDTH_RATIO
        cfg.MODEL.BACKBONE_SMALL.WIDTH = int(cfg.MODEL.BACKBONE_SMALL.WIDTH * r)
        cfg.MODEL.BACKBONE_SMALL.MERGE_WIDTH = int(
            cfg.MODEL.BACKBONE_SMALL.MERGE_WIDTH * r
        )
        cfg.MODEL.CONV_HEAD.WIDTH = int(cfg.MODEL.CONV_HEAD.WIDTH * r)
        cfg.MODEL.COND.PASSTHROUGH_DIM = int(cfg.MODEL.COND.PASSTHROUGH_DIM * r)
        self.cfg = cfg

        # current frame 0
        self.behav_embed: SubjectBehaviorEmbed = build_behavior_embed(cfg)
        self.backbone: AdaLNLoRADiNOv2ViT = build_backbone(cfg)
        self.conv_blocks: DictConvBlocks = build_conv_blocks(cfg)
        self.cls_blocks: ClassTokenMLPs = build_class_token_mlp(cfg)
        self.layer_selector: Dict[str, CachedCoordsMLP] = build_each_subject(
            partial(
                build_coords_mlp,
                cfg=cfg,
                in_dim=cfg.POSITION_ENCODING.IN_DIM,
                out_dim=self.n_layers,
                act_fn=partial(nn.Softmax, dim=-1),
            ),
            self.subject_list,
        )
        self.retina_mapper: Dict[str, CachedCoordsMLP] = build_each_subject(
            partial(
                build_coords_mlp,
                cfg=cfg,
                in_dim=cfg.POSITION_ENCODING.IN_DIM,
                out_dim=2,
                act_fn=nn.Tanh,
            ),
            self.subject_list,
        )
        self.mu_sigma = cfg.MODEL.RETINA_MAPPER.CONSTANT_SIGMA
        self.behav_pt: SubjectBehaviorEmbed = build_behavior_embed(
            cfg, cfg.MODEL.COND.PASSTHROUGH_DIM
        )

        # previous frame -1
        self.behav_embed_prev: SubjectBehaviorEmbed = build_behavior_embed(cfg)
        self.backbone_prev: AdaLNLoRADiNOv2ViT = build_backbone_prev(cfg)
        self.cls_blocks_prev: ClassTokenMLPs = build_class_token_mlp_prev(cfg)
        self.behav_pt_prev: SubjectBehaviorEmbed = build_behavior_embed(
            cfg, cfg.MODEL.COND.PASSTHROUGH_DIM
        )

        # previous frame -2:-32
        self.behav_embed_prev_feat: SubjectBehaviorEmbed = build_behavior_embed(cfg)
        self.time_emb: SubjectTimeEmbed = build_time_emd(cfg)
        self.prev_feat_embed: PreviousFeatureMLPs = build_prev_feat_mlp(cfg)
        self.prev_compress: SubjectPreviousFrameCompress = build_prev_compress(cfg)

        # voxel-wise output
        d_model = (
            self.cfg.MODEL.CONV_HEAD.WIDTH
            + self.cfg.MODEL.BACKBONE_SMALL.WIDTH
            + self.cfg.MODEL.BACKBONE_SMALL.MERGE_WIDTH
            + self.cfg.MODEL.COND.PASSTHROUGH_DIM * 2
        )
        self.voxel_outs_weight: Dict[
            str, Union[VoxelNonShareLinearWeight, CoordsMLPLinearWeight]
        ] = nn.ModuleDict(
            {
                subject: build_voxelouts_weight(cfg, n_voxel_dict[subject], d_model)
                for subject in self.subject_list
            }
        )

    def forward(
        self,
        x: Tensor,  # [B, C, H, W]
        subject: str,
        coords: Tensor,  # [N, 3]
        bhv: Optional[Tensor] = None,  # [B, D_B]
        prev_img: Optional[Tensor] = None,  # [B, C, H, W]
        prev_feats: Optional[Tensor] = None,  # [B, T, D]
        prev_bhvs: Optional[Tensor] = None,  # [B, T, D_B]
        voxel_indices: Optional[Tensor] = None,
        chunk_size=25600,
    ):
        #############################
        ### current frame 0 ###
        #############################
        c = self.behav_embed(bhv, subject=subject)  # [B, D_C]
        x_retina_grid, x_cls_dict = self.backbone.get_intermediate_layers(
            x, n=self.layers, c=c
        )
        x_retina_grid = self.conv_blocks(x_retina_grid)
        # {layer: [B, D, H/k, W/k], ...}
        x_cls_dict = self.cls_blocks(x_cls_dict)
        x_cls = _stack(x_cls_dict)  # [B, D, 4]
        c_pt0 = self.behav_pt(bhv, subject=subject)  # [B, D_C]

        #############################
        ### previous frame -1 ###
        #############################
        c = self.behav_embed_prev(prev_bhvs[:, 0, :], subject=subject)  # [B, D_C]
        _, x_cls_dict_prev = self.backbone_prev.get_intermediate_layers(
            prev_img, n=self.layers_small, c=c
        )
        x_cls_dict_prev = self.cls_blocks_prev(x_cls_dict_prev)
        x_cls_prev = _stack(x_cls_dict_prev).mean(-1)  # [B, D']
        c_pt1 = self.behav_pt_prev(prev_bhvs[:, 0, :], subject=subject)  # [B, D_C]

        #############################
        ### previous frame -2:-32 ###
        #############################
        bhvs = rearrange(prev_bhvs[:, 1:, :], "b t d -> (b t) d")
        c = self.behav_embed_prev_feat(bhvs, subject=subject)
        t = torch.arange(
            prev_feats.shape[1], dtype=torch.float32, device=prev_feats.device
        )
        t = self.time_emb(t, subject)  # [T, D_T]
        bsz = prev_feats.shape[0]
        t = repeat(t, "t d -> (b t) d", b=bsz)
        x_p = self.prev_feat_embed(prev_feats, c, t)  # [(B T), D']
        x_p = self.prev_compress(x_p, subject)  # [B, D]

        #############################
        ### voxel-wise prediction ###
        #############################

        # divide voxels into chunks
        n_voxels = coords.shape[0]
        if voxel_indices is None or voxel_indices == ...:
            voxel_indices = torch.arange(n_voxels, device=coords.device)
        voxel_indices_chunks = torch.split(voxel_indices, chunk_size)

        out_ys, reg_layers = [], []
        for voxel_indices_chunk in voxel_indices_chunks:
            out_y, reg_layer = self._forward_voxels(
                x_retina_grid,
                x_cls,
                x_cls_prev,
                x_p,
                c_pt0,
                c_pt1,
                subject,
                coords,
                voxel_indices_chunk,
            )
            out_ys.append(out_y)
            reg_layers.append(reg_layer)

        out_y = torch.cat(out_ys, dim=1)  # [B, N]
        reg_layer = torch.cat(reg_layers, dim=0).mean()  # [1]

        if self.training:
            return out_y, reg_layer
        else:
            return out_y

    def _forward_voxels(
        self,
        x_retina_grid: Dict[str, Tensor],  # {layer: [B, D, H/k, W/k], ...}
        x_cls: Tensor,  # [B, D, 4]
        x_cls_prev: Tensor,  # [B, D']
        x_p: Tensor,  # [B, D']
        c_pt0: Tensor,  # [B, D"]
        c_pt1: Tensor,  # [B, D"]
        subject: str,
        coords: Tensor,
        voxel_indices: Tensor,
    ):
        w_layer = self.layer_selector[subject](coords, voxel_indices)  # [N, 4]

        # regularization
        def entropy(x):
            return (x * x.log()).sum(dim=1)

        if self.training and next(self.layer_selector.parameters()).requires_grad:
            reg_layer = entropy(w_layer)  # [N]
        else:
            reg_layer = torch.zeros_like(w_layer[:, 0])  # [N]

        x_cls = repeat(x_cls, "b d l -> b n d l", n=1)
        _w_layer = repeat(w_layer, "n l -> b n d l", b=1, d=1)

        x_cls = (x_cls * _w_layer).sum(dim=-1)  # [B, N, D]

        mu = self.retina_mapper[subject](coords, voxel_indices)  # [N, 2]
        mu = mu * (1 - self.mu_sigma)
        if self.training:
            norm = torch.normal(0, torch.ones_like(mu) * self.mu_sigma)
            mu = mu + norm
        bsz = x_cls.shape[0]
        mu = repeat(mu, "n d -> b n d", b=bsz)
        mu = rearrange(mu, "b n (d c) -> b n d c", d=1, c=2)

        _w_layer = repeat(w_layer, "n l -> b n l", b=1)
        x_retina = None  # [B, N, D]
        for i, layer in zip(range(self.n_layers), self.layers):
            x = x_retina_grid[str(layer)]
            _x_retina = F.grid_sample(
                x,
                mu,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )  # [B, C, N, D] (C=D_model, D=1, N=N_voxels)
            _x_retina = rearrange(_x_retina, "b c n d -> b n (c d)")
            _x_retina = _x_retina * _w_layer[:, :, i : i + 1]
            if x_retina is None:
                x_retina = _x_retina
            else:
                x_retina += _x_retina
        # x_retina: [B, N, D]

        x_y = x_retina + x_cls  # [B, N, D]  # T=0
        x_cls_prev = repeat(x_cls_prev, "b d -> b n d", n=x_y.shape[1])  # T=-1
        x_p = repeat(x_p, "b d -> b n d", n=x_y.shape[1])  # T=-2:-32, x and bhv
        c_pt0 = repeat(c_pt0, "b d -> b n d", n=x_y.shape[1])  # bhv T=0
        c_pt1 = repeat(c_pt1, "b d -> b n d", n=x_y.shape[1])  # bhv T=-1
        x_y = torch.cat([x_y, x_cls_prev, x_p, c_pt0, c_pt1], dim=-1)  # [B, N, DDD]

        w, b = self.voxel_outs_weight[subject](coords, voxel_indices)  # [N, DDD], [N]
        out_y = (x_y * w.unsqueeze(0)).mean(-1) + b.unsqueeze(0)  # [B, N]

        return out_y, reg_layer  # [B, N], [N]


class DevMemVoxelWiseEncodingModel(nn.Module):
    def __init__(
        self,
        cfg: AutoConfig,
        n_voxel_dict: Dict[str, int],
    ):
        super().__init__()
        self.subject_list = cfg.DATASET.SUBJECT_LIST
        if list(n_voxel_dict.keys()) != self.subject_list:
            logging.warning(
                f"subjects in config and voxel dict are not matched, using voxel dict"
            )
            self.subject_list = list(n_voxel_dict.keys())
        self.layers = cfg.MODEL.BACKBONE.LAYERS
        self.layers_small = cfg.MODEL.BACKBONE_SMALL.LAYERS
        self.n_layers = len(self.layers)
        r = cfg.MODEL.WIDTH_RATIO
        cfg.MODEL.BACKBONE_SMALL.WIDTH = int(cfg.MODEL.BACKBONE_SMALL.WIDTH * r)
        cfg.MODEL.BACKBONE_SMALL.MERGE_WIDTH = int(
            cfg.MODEL.BACKBONE_SMALL.MERGE_WIDTH * r
        )
        cfg.MODEL.CONV_HEAD.WIDTH = int(cfg.MODEL.CONV_HEAD.WIDTH * r)
        cfg.MODEL.COND.PASSTHROUGH_DIM = int(cfg.MODEL.COND.PASSTHROUGH_DIM * r)
        self.cfg = cfg

        # current frame 0
        self.behav_embed: SubjectBehaviorEmbed = build_behavior_embed(cfg)
        self.backbone: AdaLNLoRADiNOv2ViT = build_backbone(cfg)
        if self.cfg.EXPERIMENTAL.BACKBONE_NOGRAD:
            self.behav_embed.requires_grad_(False)
            self.backbone.requires_grad_(False)
        self.conv_blocks: DictConvBlocks = build_conv_blocks(cfg)
        self.cls_blocks: ClassTokenMLPs = build_class_token_mlp(cfg)
        self.layer_selector: Dict[str, CachedCoordsMLP] = build_each_subject(
            partial(
                build_coords_mlp,
                cfg=cfg,
                in_dim=cfg.POSITION_ENCODING.IN_DIM,
                out_dim=self.n_layers,
                act_fn=partial(nn.Softmax, dim=-1),
            ),
            self.subject_list,
        )
        self.retina_mapper: Dict[str, CachedCoordsMLP] = build_each_subject(
            partial(
                build_coords_mlp,
                cfg=cfg,
                in_dim=cfg.POSITION_ENCODING.IN_DIM,
                out_dim=2,
                act_fn=nn.Tanh,
            ),
            self.subject_list,
        )
        self.mu_sigma = cfg.MODEL.RETINA_MAPPER.CONSTANT_SIGMA
        self.behav_pt: SubjectBehaviorEmbed = build_behavior_embed(
            cfg, cfg.MODEL.COND.PASSTHROUGH_DIM
        )

        # previous frame -1
        self.behav_embed_prev: SubjectBehaviorEmbed = build_behavior_embed(cfg)
        self.backbone_prev: AdaLNLoRADiNOv2ViT = build_backbone_prev(cfg)
        if self.cfg.EXPERIMENTAL.BACKBONE_NOGRAD:
            self.behav_embed_prev.requires_grad_(False)
            self.backbone_prev.requires_grad_(False)
        self.cls_blocks_prev: ClassTokenMLPs = build_class_token_mlp_prev(cfg)
        self.behav_pt_prev: SubjectBehaviorEmbed = build_behavior_embed(
            cfg, cfg.MODEL.COND.PASSTHROUGH_DIM
        )

        # previous frame -2:-32
        self.behav_embed_prev_feat: SubjectBehaviorEmbed = build_behavior_embed(cfg)
        self.time_emb: SubjectTimeEmbed = build_time_emd(cfg)
        self.prev_feat_embed: PreviousFeatureMLPs = build_prev_feat_mlp(cfg)
        self.prev_compress: SubjectPreviousFrameCompress = build_prev_compress(cfg)

        # voxel-wise output
        d_model = (
            self.cfg.MODEL.CONV_HEAD.WIDTH
            + self.cfg.MODEL.BACKBONE_SMALL.WIDTH
            + self.cfg.MODEL.BACKBONE_SMALL.MERGE_WIDTH
            + self.cfg.MODEL.COND.PASSTHROUGH_DIM * 2
        )
        self.voxel_outs_weight: Dict[
            str, Union[VoxelNonShareLinearWeight, CoordsMLPLinearWeight]
        ] = nn.ModuleDict(
            {
                subject: build_voxelouts_weight(cfg, n_voxel_dict[subject], d_model)
                for subject in self.subject_list
            }
        )

        self._behav_only_mlp = Mlp(self.cfg.MODEL.COND.DIM, out_features=d_model)

    def forward(
        self,
        x: Tensor,  # [B, C, H, W]
        subject: str,
        coords: Tensor,  # [N, 3]
        bhv: Optional[Tensor] = None,  # [B, D_B]
        prev_img: Optional[Tensor] = None,  # [B, C, H, W]
        prev_feats: Optional[Tensor] = None,  # [B, T, D]
        prev_bhvs: Optional[Tensor] = None,  # [B, T, D_B]
        voxel_indices: Optional[Tensor] = None,
        chunk_size=25600,
    ):
        if self.cfg.EXPERIMENTAL.USE_BHV == False:
            bhv *= 0
            prev_bhvs *= 0
        if self.cfg.EXPERIMENTAL.BEHV_ONLY:
            return self._forward_bhvs(
                bhv, subject, coords, voxel_indices=voxel_indices, chunk_size=chunk_size
            )

        #############################
        ### current frame 0 ###
        #############################
        if self.cfg.EXPERIMENTAL.BACKBONE_NOGRAD:
            c = self.behav_embed(bhv, subject=subject)  # [B, D_C]
            with torch.no_grad():
                x_retina_grid, x_cls_dict = self.backbone.get_intermediate_layers(
                    x, n=self.layers, c=c
                )
        else:
            c = self.behav_embed(bhv, subject=subject)  # [B, D_C]
            x_retina_grid, x_cls_dict = self.backbone.get_intermediate_layers(
                x, n=self.layers, c=c
            )
        if self.cfg.EXPERIMENTAL.USE_RETINA_MAPPER:
            x_retina_grid = self.conv_blocks(x_retina_grid)
            # {layer: [B, D, H/k, W/k], ...}
        else:
            x_retina_grid = None
        x_cls_dict = self.cls_blocks(x_cls_dict)
        x_cls = _stack(x_cls_dict)  # [B, D, 4]
        if self.cfg.EXPERIMENTAL.USE_LAYER_SELECTOR == False:
            x_cls = x_cls.mean(-1)
        c_pt0 = self.behav_pt(bhv, subject=subject)  # [B, D_C]
        if self.cfg.EXPERIMENTAL.USE_BHV_PASSTHROUGH == False:
            c_pt0 = c_pt0 * 0

        if self.cfg.EXPERIMENTAL.USE_PREV_FRAME:
            #############################
            ### previous frame -1 ###
            #############################
            if self.cfg.EXPERIMENTAL.BACKBONE_NOGRAD:
                with torch.no_grad():
                    c = self.behav_embed_prev(prev_bhvs[:, 0, :], subject=subject)  # [B, D_C]
                    _, x_cls_dict_prev = self.backbone_prev.get_intermediate_layers(
                        prev_img, n=self.layers_small, c=c
                    )
            else:
                c = self.behav_embed_prev(prev_bhvs[:, 0, :], subject=subject)
                _, x_cls_dict_prev = self.backbone_prev.get_intermediate_layers(
                    prev_img, n=self.layers_small, c=c
                )
            x_cls_dict_prev = self.cls_blocks_prev(x_cls_dict_prev)
            x_cls_prev = _stack(x_cls_dict_prev).mean(-1)  # [B, D']
            c_pt1 = self.behav_pt_prev(prev_bhvs[:, 0, :], subject=subject)  # [B, D_C]

            #############################
            ### previous frame -2:-32 ###
            #############################
            bhvs = rearrange(prev_bhvs[:, 1:, :], "b t d -> (b t) d")
            c = self.behav_embed_prev_feat(bhvs, subject=subject)
            t = torch.arange(
                prev_feats.shape[1], dtype=torch.float32, device=prev_feats.device
            )
            t = self.time_emb(t, subject)  # [T, D_T]
            bsz = prev_feats.shape[0]
            t = repeat(t, "t d -> (b t) d", b=bsz)
            x_p = self.prev_feat_embed(prev_feats, c, t)  # [(B T), D']
            x_p = self.prev_compress(x_p, subject)  # [B, D]
        else:
            bsz = x_cls.shape[0]
            kw = {"device": x_cls.device, "dtype": x_cls.dtype}
            x_cls_prev = torch.zeros((bsz, self.cfg.MODEL.BACKBONE_SMALL.WIDTH), **kw)
            c_pt1 = torch.zeros((bsz, self.cfg.MODEL.COND.PASSTHROUGH_DIM), **kw)
            x_p = torch.zeros((bsz, self.cfg.MODEL.BACKBONE_SMALL.MERGE_WIDTH), **kw)

        #############################
        ### voxel-wise prediction ###
        #############################

        # divide voxels into chunks
        n_voxels = coords.shape[0]
        if voxel_indices is None or voxel_indices == ...:
            voxel_indices = torch.arange(n_voxels, device=coords.device)
        voxel_indices_chunks = torch.split(voxel_indices, chunk_size)

        out_ys, reg_layers = [], []
        for voxel_indices_chunk in voxel_indices_chunks:
            out_y, reg_layer = self._forward_voxels(
                x_retina_grid,
                x_cls,
                x_cls_prev,
                x_p,
                c_pt0,
                c_pt1,
                subject,
                coords,
                voxel_indices_chunk,
            )
            out_ys.append(out_y)
            reg_layers.append(reg_layer)

        out_y = torch.cat(out_ys, dim=1)  # [B, N]
        reg_layer = torch.cat(reg_layers, dim=0).mean()  # [1]

        if self.training:
            return out_y, reg_layer
        else:
            return out_y

    def _forward_voxels(
        self,
        x_retina_grid: Dict[str, Tensor],  # {layer: [B, D, H/k, W/k], ...}
        x_cls: Tensor,  # [B, D, 4]
        x_cls_prev: Tensor,  # [B, D']
        x_p: Tensor,  # [B, D']
        c_pt0: Tensor,  # [B, D"]
        c_pt1: Tensor,  # [B, D"]
        subject: str,
        coords: Tensor,
        voxel_indices: Tensor,
    ):
        N = len(voxel_indices)
        if self.cfg.EXPERIMENTAL.USE_LAYER_SELECTOR:
            w_layer = self.layer_selector[subject](coords, voxel_indices)  # [N, 4]

            # regularization
            def entropy(x):
                return (x * x.log()).sum(dim=1)

            if self.training and next(self.layer_selector.parameters()).requires_grad:
                reg_layer = entropy(w_layer)  # [N]
            else:
                reg_layer = torch.zeros_like(w_layer[:, 0])  # [N]

            x_cls = repeat(x_cls, "b d l -> b n d l", n=1)
            _w_layer = repeat(w_layer, "n l -> b n d l", b=1, d=1)

            x_cls = (x_cls * _w_layer).sum(dim=-1)  # [B, N, D]
        else:
            w_layer = None
            reg_layer = torch.zeros(N, dtype=x_cls.dtype, device=x_cls.device)
            x_cls = repeat(x_cls, "b d -> b n d", n=N)

        if self.cfg.EXPERIMENTAL.USE_RETINA_MAPPER:
            mu = self.retina_mapper[subject](coords, voxel_indices)  # [N, 2]
            mu = mu * (1 - self.mu_sigma)
            if self.training:
                norm = torch.normal(0, torch.ones_like(mu) * self.mu_sigma)
                mu = mu + norm
            bsz = x_cls.shape[0]
            mu = repeat(mu, "n d -> b n d", b=bsz)
            mu = rearrange(mu, "b n (d c) -> b n d c", d=1, c=2)

            if self.cfg.EXPERIMENTAL.USE_LAYER_SELECTOR:
                _w_layer = repeat(w_layer, "n l -> b n l", b=1)
            x_retina = None  # [B, N, D]
            for i, layer in zip(range(self.n_layers), self.layers):
                x = x_retina_grid[str(layer)]
                _x_retina = F.grid_sample(
                    x,
                    mu,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )  # [B, C, N, D] (C=D_model, D=1, N=N_voxels)
                _x_retina = rearrange(_x_retina, "b c n d -> b n (c d)")
                if self.cfg.EXPERIMENTAL.USE_LAYER_SELECTOR:
                    _x_retina = _x_retina * _w_layer[:, :, i : i + 1]
                if x_retina is None:
                    x_retina = _x_retina
                else:
                    x_retina += _x_retina
            # x_retina: [B, N, D]

            x_y = x_retina + x_cls  # [B, N, D]  # T=0
        else:
            x_y = x_cls  # [B, N, D]  # T=0
        x_cls_prev = repeat(x_cls_prev, "b d -> b n d", n=x_y.shape[1])  # T=-1
        x_p = repeat(x_p, "b d -> b n d", n=x_y.shape[1])  # T=-2:-32, x and bhv
        c_pt0 = repeat(c_pt0, "b d -> b n d", n=x_y.shape[1])  # bhv T=0
        c_pt1 = repeat(c_pt1, "b d -> b n d", n=x_y.shape[1])  # bhv T=-1
        x_y = torch.cat([x_y, x_cls_prev, x_p, c_pt0, c_pt1], dim=-1)  # [B, N, DDD]

        w, b = self.voxel_outs_weight[subject](coords, voxel_indices)  # [N, DDD], [N]
        out_y = (x_y * w.unsqueeze(0)).mean(-1) + b.unsqueeze(0)  # [B, N]

        return out_y, reg_layer  # [B, N], [N]

    def _forward_bhvs(
        self,
        bhv,
        subject,
        coords,
        voxel_indices=None,
        chunk_size=25600,
    ):
        c = self.behav_embed(bhv, subject=subject)  # [B, D_C]
        x = self._behav_only_mlp(c)
        # divide voxels into chunks
        n_voxels = coords.shape[0]
        if voxel_indices is None or voxel_indices == ...:
            voxel_indices = torch.arange(n_voxels, device=coords.device)
        voxel_indices_chunks = torch.split(voxel_indices, chunk_size)

        out_ys, reg_layers = [], []
        for voxel_indices_chunk in voxel_indices_chunks:
            n = voxel_indices_chunk.shape[0]
            _x = repeat(x, "b c -> b n c", n=n)

            w, b = self.voxel_outs_weight[subject](
                coords, voxel_indices_chunk
            )  # [N, DDD], [N]
            out_y = (_x * w.unsqueeze(0)).mean(-1) + b.unsqueeze(0)  # [B, N]

            reg_layer = torch.zeros(n, dtype=x.dtype, device=x.device)  # [N]

            out_ys.append(out_y)
            reg_layers.append(reg_layer)

        out_y = torch.cat(out_ys, dim=1)  # [B, N]
        reg_layer = torch.cat(reg_layers, dim=0).mean()  # [1]

        if self.training:
            return out_y, reg_layer
        else:
            return out_y


class DevVoxelWiseEncodingModel(nn.Module):
    def __init__(
        self,
        cfg: AutoConfig,
        n_voxel_dict: Dict[str, int],
    ):
        super().__init__()
        self.subject_list = self.cfg.DATASET.SUBJECT_LIST
        if list(n_voxel_dict.keys()) != self.subject_list:
            logging.warning(
                f"subjects in config and voxel dict are not matched, using voxel dict"
            )
            self.subject_list = list(n_voxel_dict.keys())
        self.layers = cfg.MODEL.BACKBONE.LAYERS
        self.n_layers = len(self.layers)
        r = cfg.MODEL.WIDTH_RATIO
        cfg.MODEL.BACKBONE_SMALL.WIDTH = int(cfg.MODEL.BACKBONE_SMALL.WIDTH * r)
        cfg.MODEL.BACKBONE_SMALL.MERGE_WIDTH = int(
            cfg.MODEL.BACKBONE_SMALL.MERGE_WIDTH * r
        )
        cfg.MODEL.CONV_HEAD.WIDTH = int(cfg.MODEL.CONV_HEAD.WIDTH * r)
        cfg.MODEL.COND.PASSTHROUGH_DIM = int(cfg.MODEL.COND.PASSTHROUGH_DIM * r)
        self.cfg = cfg

        self.behav_embed: SubjectBehaviorEmbed = build_behavior_embed(cfg)
        self._behav_only_mlp = Mlp(
            self.cfg.MODEL.COND.DIM, out_features=self.cfg.MODEL.CONV_HEAD.WIDTH
        )
        self.backbone: AdaLNLoRADiNOv2ViT = build_backbone(cfg)
        if self.cfg.EXPERIMENTAL.STRAIGHT_FORWARD:
            if not self.cfg.EXPERIMENTAL.STRAIGHT_FORWARD_BUT_KEEP_BACKBONE_GRAD:
                self.behav_embed.requires_grad_(False)
                self.backbone.requires_grad_(False)
        self.conv_blocks: DictConvBlocks = build_conv_blocks(cfg)
        self.cls_blocks: ClassTokenMLPs = build_class_token_mlp(cfg)
        self.cls_blocks_prev1: ClassTokenMLPs = build_class_token_mlp_prev(cfg)
        self.cls_blocks_prev2: ClassTokenMLPs = build_class_token_mlp_prev(cfg)

        self.layer_selector: Dict[str, CachedCoordsMLP] = build_each_subject(
            partial(
                build_coords_mlp,
                cfg=cfg,
                in_dim=cfg.POSITION_ENCODING.IN_DIM,
                out_dim=self.n_layers,
                act_fn=partial(nn.Softmax, dim=-1),
            ),
            self.subject_list,
        )

        self.retina_mapper: Dict[str, CachedCoordsMLP] = build_each_subject(
            partial(
                build_coords_mlp,
                cfg=cfg,
                in_dim=cfg.POSITION_ENCODING.IN_DIM,
                out_dim=2,
                act_fn=nn.Tanh,
            ),
            self.subject_list,
        )
        self.mu_sigma = cfg.MODEL.RETINA_MAPPER.CONSTANT_SIGMA

        # TODO: maybe this should not be cached
        self.use_bottle_neck = cfg.MODEL.BOTTLENECK.RANK > 0
        if self.use_bottle_neck:
            self.down_shape = (cfg.MODEL.CONV_HEAD.WIDTH, cfg.MODEL.BOTTLENECK.RANK)
            self.bottleneck_down: Dict[str, CachedCoordsMLP] = build_each_subject(
                partial(
                    build_coords_mlp,
                    cfg=cfg,
                    in_dim=cfg.POSITION_ENCODING.IN_DIM,
                    out_dim=self.down_shape[0] * self.down_shape[1],
                    act_fn=nn.Identity,
                ),
                self.subject_list,
            )
            # self.up_shape = (cfg.MODEL.BOTTLENECK.RANK, cfg.MODEL.BOTTLENECK.OUT_DIM)
            # self.bottleneck_up : Dict[str, CachedCoordsMLP] = build_each_subject(
            #     partial(
            #         build_coords_mlp,
            #         cfg=cfg,
            #         in_dim=cfg.POSITION_ENCODING.IN_DIM,
            #         out_dim=self.up_shape[0] * self.up_shape[1],
            #         act_fn=nn.Identity,
            #     ),
            #     self.subject_list,
            # )
            # self.bottleneck_act = nn.Identity()

        d_model = self.cfg.MODEL.CONV_HEAD.WIDTH
        self.voxel_outs_weight: Dict[
            str, Union[VoxelNonShareLinearWeight, CoordsMLPLinearWeight]
        ] = nn.ModuleDict(
            {
                subject: build_voxelouts_weight(cfg, n_voxel_dict[subject], d_model)
                for subject in self.subject_list
            }
        )

    def forward(
        self,
        x: Tensor,  # [B, C, H, W]
        subject: str,
        coords: Tensor,  # [N, 3]
        bhv: Optional[Tensor] = None,  # [B, D_B]
        prev_img: Optional[Tensor] = None,  # [B, C, H, W]
        prev_feats: Optional[Tensor] = None,  # [B, T, D]
        prev_bhvs: Optional[Tensor] = None,  # [B, T, D_B]
        voxel_indices: Optional[Tensor] = None,
        chunk_size=8096,
    ):
        if self.cfg.EXPERIMENTAL.BEHV_ONLY:
            return self._forward_bhvs(bhv, subject, coords, voxel_indices=voxel_indices)

        if self.cfg.EXPERIMENTAL.STRAIGHT_FORWARD:
            if self.cfg.EXPERIMENTAL.STRAIGHT_FORWARD_BUT_KEEP_BACKBONE_GRAD:
                c = self.behav_embed(bhv, subject=subject)  # [B, D_C]
                _, _x_cls_dict = self.backbone.get_intermediate_layers(
                    x, n=self.layers, c=c
                )
                x_retina_grid = None
            else:
                with torch.no_grad():
                    c = self.behav_embed(bhv, subject=subject)  # [B, D_C]
                    _, _x_cls_dict = self.backbone.get_intermediate_layers(
                        x, n=self.layers, c=c
                    )
                    x_retina_grid = None
        else:
            c = self.behav_embed(bhv, subject=subject)  # [B, D_C]
            x_retina_grid, _x_cls_dict = self.backbone.get_intermediate_layers(
                x, n=self.layers, c=c
            )
            x_retina_grid = self.conv_blocks(x_retina_grid)
        # x_retina_grid: {layer: [B, D, H/k, W/k], ...}
        __x_cls_dict = {k: v.clone() for k, v in _x_cls_dict.items()}
        x_cls_dict = self.cls_blocks(__x_cls_dict)
        x_cls = _stack(x_cls_dict)  # [B, D, 4]

        # #### previous frame don't enjoy RetinaMapper and LayerSelector ###
        # TODO: add time embedding to previous frames

        # divide voxels into chunks
        n_voxels = coords.shape[0]
        if voxel_indices is None or voxel_indices == ...:
            voxel_indices = torch.arange(n_voxels, device=coords.device)
        voxel_indices_chunks = torch.split(voxel_indices, chunk_size)

        out_ys, reg_layers = [], []
        for voxel_indices_chunk in voxel_indices_chunks:
            if not self.cfg.EXPERIMENTAL.STRAIGHT_FORWARD:
                out_y, reg_layer = self._forward_voxels(
                    x_retina_grid,
                    x_cls,
                    # x_cls_prev1,
                    # x_cls_prev2,
                    subject,
                    coords,
                    voxel_indices_chunk,
                )
            else:
                out_y, reg_layer = self._straight_forward_voxels_forward(
                    x_cls,
                    subject,
                    coords,
                    voxel_indices_chunk,
                )
            out_ys.append(out_y)
            reg_layers.append(reg_layer)

        out_y = torch.cat(out_ys, dim=1)  # [B, N]
        reg_layer = torch.cat(reg_layers, dim=0).mean()  # [1]

        if self.training:
            return out_y, reg_layer
        else:
            return out_y

    def _forward_bhvs(
        self,
        bhv,
        subject,
        coords,
        voxel_indices=None,
        chunk_size=40960,
    ):
        c = self.behav_embed(bhv, subject=subject)  # [B, D_C]
        x = self._behav_only_mlp(c)
        # divide voxels into chunks
        n_voxels = coords.shape[0]
        if voxel_indices is None or voxel_indices == ...:
            voxel_indices = torch.arange(n_voxels, device=coords.device)
        voxel_indices_chunks = torch.split(voxel_indices, chunk_size)

        out_ys, reg_layers = [], []
        for voxel_indices_chunk in voxel_indices_chunks:
            n = voxel_indices_chunk.shape[0]
            x = repeat(x, "b c -> b n c", n=n)

            w, b = self.voxel_outs_weight[subject](
                coords, voxel_indices_chunk
            )  # [N, DDD], [N]
            out_y = (x * w.unsqueeze(0)).mean(-1) + b.unsqueeze(0)  # [B, N]

            reg_layer = torch.zeros(n, dtype=x.dtype, device=x.device)  # [N]

            out_ys.append(out_y)
            reg_layers.append(reg_layer)

        out_y = torch.cat(out_ys, dim=1)  # [B, N]
        reg_layer = torch.cat(reg_layers, dim=0).mean()  # [1]

        if self.training:
            return out_y, reg_layer
        else:
            return out_y

    def _forward_voxels(
        self,
        x_retina_grid: Dict[str, Tensor],  # {layer: [B, D, H/k, W/k], ...}
        x_cls: Tensor,  # [B, D, 4]
        # x_cls_prev1: Tensor,  # [B, D, 4]
        # x_cls_prev2: Tensor,  # [B, D, 4]
        subject: str,
        coords: Tensor,
        voxel_indices: Tensor,
    ):
        w_layer = self.layer_selector[subject](coords, voxel_indices)  # [N, 4]

        # regularization
        def entropy(x):
            return (x * x.log()).sum(dim=1)

        if self.training:
            reg_layer = entropy(w_layer)  # [N]
        else:
            reg_layer = torch.zeros_like(w_layer[:, 0])  # [N]

        x_cls = repeat(x_cls, "b d l -> b n d l", n=1)
        _w_layer = repeat(w_layer, "n l -> b n d l", b=1, d=1)

        x_cls = (x_cls * _w_layer).sum(dim=-1)  # [B, N, D]

        mu = self.retina_mapper[subject](coords, voxel_indices)  # [N, 2]
        mu = mu * (1 - self.mu_sigma)
        if self.training:
            norm = torch.normal(0, torch.ones_like(mu) * self.mu_sigma)
            mu = mu + norm
        bsz = x_cls.shape[0]
        mu = repeat(mu, "n d -> b n d", b=bsz)
        mu = rearrange(mu, "b n (d c) -> b n d c", d=1, c=2)

        _w_layer = repeat(w_layer, "n l -> b n l", b=1)
        x_retina = None  # [B, N, D]
        for i, layer in zip(range(self.n_layers), self.layers):
            x = x_retina_grid[str(layer)]
            _x_retina = F.grid_sample(
                x,
                mu,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )  # [B, C, N, D] (C=D_model, D=1, N=N_voxels)
            _x_retina = rearrange(_x_retina, "b c n d -> b n (c d)")
            _x_retina = _x_retina * _w_layer[:, :, i : i + 1]
            if x_retina is None:
                x_retina = _x_retina
            else:
                x_retina += _x_retina
        # x_retina: [B, N, D]

        x_y = x_retina + x_cls  # [B, N, D]
        # x_cls_prev1 = repeat(x_cls_prev1.mean(-1), "b d -> b n d", n=x_y.shape[1])
        # x_cls_prev2 = repeat(x_cls_prev2.mean(-1), "b d -> b n d", n=x_y.shape[1])
        # x_y = torch.cat([x_y, x_cls_prev1, x_cls_prev2], dim=-1)  # [B, N, DDD]

        if self.use_bottle_neck:
            w_a = self.bottleneck_down[subject](coords, voxel_indices)  # [N, D*R]
            w_a = rearrange(
                w_a, "n (d r) -> n d r", d=self.down_shape[0], r=self.down_shape[1]
            )
            # w_b = self.bottleneck_up[subject](coords, voxel_indices)  # [N, R*D]
            # w_b = rearrange(w_b, "n (r d) -> n r d", d=self.up_shape[1], r=self.up_shape[0])

            x_y = torch.einsum("bnd,ndr->bnr", x_y, w_a)  # [B, N, R]
            # x_y = torch.einsum("bnr,nrd->bnd", x_y, w_b)  # [B, N, D]
            # x_y = self.bottleneck_act(x_y)  # [B, N, D]

        w, b = self.voxel_outs_weight[subject](coords, voxel_indices)  # [N, DDD], [N]
        out_y = (x_y * w.unsqueeze(0)).mean(-1) + b.unsqueeze(0)  # [B, N]

        return out_y, reg_layer  # [B, N], [N]

    def _straight_forward_voxels_forward(
        self,
        x_cls: Tensor,  # [B, D, 4]
        subject: str,
        coords: Tensor,
        voxel_indices: Tensor,
    ):
        n = voxel_indices.shape[0]
        x_cls = x_cls.mean(-1)
        x_cls = repeat(x_cls, "b d -> b n d", n=n)

        w, b = self.voxel_outs_weight[subject](coords, voxel_indices)  # [N, DDD], [N]
        out_y = (x_cls * w.unsqueeze(0)).mean(-1) + b.unsqueeze(0)  # [B, N]

        reg_layer = torch.zeros(n, dtype=x_cls.dtype, device=x_cls.device)  # [N]
        return out_y, reg_layer  # [B, N], [N]


if __name__ == "__main__":
    coords = torch.rand(1000, 3)
    n_voxels = {"sub-01": 1000}
    from config_utils import get_cfg_defaults

    cfg = get_cfg_defaults()
    cfg.MODEL.BACKBONE.NAME = "adaln_lora_clip_vit"
    cfg.MODEL.VOXEL_OUTS.SHARED.USE = True
    cfg.DATASET.SUBJECT_LIST = ["sub-01"]
    model = DevVoxelWiseEncodingModel(cfg, n_voxels)
    x = torch.rand(2, 3, 224, 224)
    out = model.forward(x, "sub-01", coords)
    pass
