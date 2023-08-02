from collections import OrderedDict
from functools import partial
import os
from copy import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from torch_lazy.nn import LazyBatchNorm, LazyLayerNorm

# +
from einops import einsum, rearrange, repeat

from torch import Tensor, nn

from config import AutoConfig
from point_pe import point_position_encoding

from timm.layers.mlp import Mlp


class PositionalEncoding(nn.Module):
    def __init__(self, max_steps=1000, features=32, periods=10000):
        super().__init__()
        self.pe = partial(
            point_position_encoding,
            max_steps=max_steps,
            features=features,
            periods=periods,
        )

    @torch.no_grad()
    def forward(self, x):
        return self.pe(x)


def coords_mlp(
    in_dim,
    out_dim,
    hidden_dim=256,
    depth=3,
    act_fn=nn.GELU,
    max_steps=100,
    features=32,
    periods=100,
    fi_act_fn=nn.Identity,
):
    assert depth >= 2
    modules = []
    modules.append(
        PositionalEncoding(max_steps=max_steps, features=features, periods=periods)
    )
    in_dim = in_dim * features * 2
    for i in range(depth - 1):
        modules.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
        modules.append(act_fn())
    modules.append(nn.Linear(hidden_dim, out_dim))
    modules.append(fi_act_fn())
    return nn.Sequential(*modules)


class CachedCoordsMLP(nn.Module):
    # caching greatly improves speed, since number of voxels is huge
    def __init__(self, in_dim, out_dim, hidden_dim=256, depth=3, act_fn=nn.Identity):
        super().__init__()
        self.mlp = coords_mlp(
            in_dim, out_dim, hidden_dim=hidden_dim, depth=depth, fi_act_fn=act_fn
        )
        self.cache = None

    def forward(self, coords, voxel_indices):
        if self.training and self.is_req_grad:
            self.cache = None
            return self.mlp(coords[voxel_indices])
        else:
            with torch.no_grad():
                if self.cache is None:
                    self.cache = self.mlp(coords)
                return self.cache[voxel_indices]

    @property
    def is_req_grad(self):
        return next(self.parameters()).requires_grad


def build_coords_mlp(
    cfg: AutoConfig, in_dim, out_dim, act_fn=partial(nn.Softmax, dim=-1)
):
    return CachedCoordsMLP(
            in_dim,
            out_dim,
            hidden_dim=cfg.MODEL.COORDS_MLP.WIDTH,
            depth=cfg.MODEL.COORDS_MLP.DEPTH,
            act_fn=act_fn,
        )


class VoxelNonShareLinearWeight(nn.Module):
    def __init__(self, d_model, n_voxels, **kwargs):
        super().__init__()
        dummy = nn.Linear(d_model, n_voxels)
        self.weight = nn.Parameter(dummy.weight)  # (n_voxels, d_model)
        self.bias = nn.Parameter(dummy.bias)  # (n_voxels,)

    def forward(self, coords, voxel_indices=..., *args, **kwargs):
        w = self.weight[voxel_indices]  # (n_voxels, d_model)
        b = self.bias[voxel_indices]  # (n_voxels,)
        return w, b


class CoordsMLPLinearWeight(nn.Module):
    def __init__(self, d_model, n_voxels, in_dim=3, hidden_dim=256, depth=3, **kwargs):
        super().__init__()
        self.w_mlp = CachedCoordsMLP(
            in_dim, d_model, hidden_dim=hidden_dim, depth=depth
        )
        self.b = nn.Parameter(torch.zeros(n_voxels))

    def forward(self, coords, voxel_indices=..., *args, **kwargs):
        w = self.w_mlp(coords, voxel_indices)  # (n_voxels, d_model)
        b = self.b[voxel_indices]  # (n_voxels,)
        return w, b


def build_voxelouts_weight(cfg: AutoConfig, n_voxels, d_model):
    kwargs = {
        "d_model": d_model,
        "n_voxels": n_voxels,
        "in_dim": cfg.POSITION_ENCODING.IN_DIM,
        "hidden_dim": cfg.MODEL.COORDS_MLP.WIDTH,
        "depth": cfg.MODEL.COORDS_MLP.DEPTH,
    }
    if cfg.MODEL.VOXEL_OUTS.SHARED.USE:
        kwargs["hidden_dim"] = cfg.MODEL.VOXEL_OUTS.SHARED.MLP.WIDTH
        kwargs["depth"] = cfg.MODEL.VOXEL_OUTS.SHARED.MLP.DEPTH
        return CoordsMLPLinearWeight(**kwargs)
    else:
        return VoxelNonShareLinearWeight(**kwargs)


class LinearBlock(nn.Module):
    def __init__(self, in_planes, n):
        super(LinearBlock, self).__init__()
        dummy = nn.Linear(in_planes, n)
        self.weight = nn.Parameter(dummy.weight.unsqueeze(0))
        self.bias = nn.Parameter(dummy.bias.unsqueeze(0))

    def forward(self, x, voxel_indices=None):
        voxel_indices = ... if voxel_indices is None else voxel_indices
        out = (x * self.weight[:, voxel_indices, :]).mean(dim=-1)  # mean is critical
        out += self.bias[:, voxel_indices]
        return out


class VoxelOutBlock(nn.Module):
    # although this code runs for depth > 1, it is not tested
    def __init__(self, in_planes, n, planes=32, depth=1):
        super(VoxelOutBlock, self).__init__()
        planes = in_planes if planes is None else planes
        self.weight = nn.ParameterList()
        self.bias = nn.ParameterList()
        self.act = nn.GELU()
        # self.act = nn.Identity()
        self.depth = depth
        for i in range(depth):
            o = planes if i < depth - 1 else 1
            weight = []
            bias = []
            for j in range(o):
                dummy = nn.Linear(
                    in_planes if i == 0 else planes,
                    n,
                )
                weight.append(dummy.weight.unsqueeze(0).clone())
                bias.append(dummy.bias.unsqueeze(0).clone())
            weight = torch.cat(weight, dim=0)
            bias = torch.cat(bias, dim=0)
            weight = rearrange(weight, "o n i -> n i o", n=n, o=o)
            bias = rearrange(bias, "o n -> n o", n=n, o=o)
            self.weight.append(nn.Parameter(weight))
            self.bias.append(nn.Parameter(bias))

    def forward(self, x, voxel_indices=None):
        voxel_indices = ... if voxel_indices is None else voxel_indices
        for ww, bb in zip(self.weight, self.bias):
            w = ww[voxel_indices]
            b = bb[voxel_indices]
            x = einsum(x, w, "b n i, n i o -> b n o")
            x /= w.shape[1]  # mean is critical
            x += b[None, ...]
            if x.shape[-1] != 1:
                x = self.act(x)
        x = x.squeeze(-1)
        return x


class NeuronProjector(nn.Module):
    def __init__(
        self,
        cfg: AutoConfig,
        layer_list: List[str],
        neuron_coords: Tensor,
        act_fn=nn.GELU,
    ):
        super().__init__()
        self.cfg = cfg
        self.layer_list = layer_list
        self.neuron_coords = neuron_coords
        self.neuron_coords.requires_grad = False
        self.act_fn = act_fn

        self.projectors = nn.ModuleDict()
        self.eye_shifters = nn.ModuleDict()

        if self.cfg.MODEL.NEURON_PROJECTOR.SEPARATE_LAYERS:
            for layer in self.layer_list:
                k = layer.replace(".", "_")
                self.projectors[k] = self.build_neuron_projector(
                    neuron_coords.shape[-1]
                )
                self.eye_shifters[k] = self.build_eye_shifter()
        else:
            shared_projector = self.build_neuron_projector(neuron_coords.shape[-1])
            shared_eye_shifter = self.build_eye_shifter()
            for layer in self.layer_list:
                k = layer.replace(".", "_")
                self.projectors[k] = shared_projector
                self.eye_shifters[k] = shared_eye_shifter

        self.layer_gate = self.build_layer_gate(
            neuron_coords.shape[-1], len(layer_list)
        )

    def forward(self, batch_size, eye_coords=None, voxel_indices=None):
        if next(self.projectors.parameters()).requires_grad:
            grids, coord_inp, (reg_mu1, reg_mu2, reg_mu3) = self._forward(
                batch_size, eye_coords, voxel_indices
            )
        else:
            with torch.no_grad():
                grids, coord_inp, (reg_mu1, reg_mu2, reg_mu3) = self._forward(
                    batch_size, eye_coords, voxel_indices
                )

        if next(self.layer_gate.parameters()).requires_grad:
            gate = self.layer_gate(coord_inp)
        else:
            with torch.no_grad():
                gate = self.layer_gate(coord_inp)

        return grids, gate, (reg_mu1, reg_mu2, reg_mu3)

    def _forward(
        self,
        batch_size,
        eye_coords=None,
        voxel_indices=None,
    ):
        if self.neuron_coords.device != self.device:
            self.neuron_coords = self.neuron_coords.to(self.device)

        voxel_indices = ... if voxel_indices is None else voxel_indices
        coord_inp = self.neuron_coords[voxel_indices]

        # gate = self.layer_gate(coord_inp)
        # gate = 1.

        grids = {}
        for layer in self.layer_list:
            k = layer.replace(".", "_")

            mu = self.projectors[k](coord_inp)

            if self.training and next(self.projectors.parameters()).requires_grad:
                reg_mu1 = torch.cdist(mu, mu, p=2)
                reg_mu1 = 1.0 / (reg_mu1 + 1e-3)
                reg_mu1 = reg_mu1.mean()
                reg_mu2 = torch.sqrt((mu**2).sum(dim=-1)).mean()
                reg_mu3 = mu[:, 0].mean() ** 2 + mu[:, 1].mean() ** 2
            else:
                reg_mu1 = torch.tensor(0.0)
                reg_mu2 = torch.tensor(0.0)
                reg_mu3 = torch.tensor(0.0)

            mu = repeat(mu, "n c -> b n c", b=batch_size)

            if self.training:
                norm = torch.normal(
                    0,
                    torch.ones_like(mu) * self.cfg.MODEL.NEURON_PROJECTOR.SIGMA_SCALE,
                )
                mu = mu + norm

            if eye_coords is not None:
                shift = self.eye_shifters[k](eye_coords)
                shift = repeat(shift, "b c -> b n c", n=mu.shape[1])
                mu += shift

            grid = rearrange(mu, "b n (d c) -> b n d c", d=1, c=2)

            grids[layer] = grid

        return grids, coord_inp, (reg_mu1, reg_mu2, reg_mu3)

    def build_layer_gate(self, location_dim, num_layers):
        depth = self.cfg.MODEL.LAYER_GATE.DEPTH
        width = self.cfg.MODEL.LAYER_GATE.WIDTH
        assert depth >= 2
        modules = []
        for i in range(depth - 1):
            modules.append(nn.Linear(location_dim if i == 0 else width, width))
            modules.append(self.act_fn())
        output_dim = num_layers
        modules.append(nn.Linear(width, output_dim))
        modules.append(nn.Softmax(dim=-1))
        return nn.Sequential(*modules)

    def build_neuron_projector(self, location_dim, output_dim=None, final_act=nn.Tanh):
        depth = self.cfg.MODEL.NEURON_PROJECTOR.DEPTH
        width = self.cfg.MODEL.NEURON_PROJECTOR.WIDTH
        assert depth >= 2
        modules = []
        for i in range(depth - 1):
            modules.append(nn.Linear(location_dim if i == 0 else width, width))
            modules.append(self.act_fn())
        output_dim = 2 if output_dim is None else output_dim
        modules.append(nn.Linear(width, output_dim))
        modules.append(final_act())
        return nn.Sequential(*modules)

    def build_eye_shifter(self):
        return nn.Sequential(nn.Linear(2, 8), nn.SiLU(), nn.Linear(8, 2), nn.Tanh())

    @property
    def device(self):
        return next(self.parameters()).device


class TopyNeck(nn.Module):
    def __init__(
        self,
        cfg: AutoConfig,
        in_c_dict: Dict[str, int],
        num_voxel_dict: Dict[str, int],
        neuron_coords_dict: Dict[str, Tensor],
        act_fn=nn.SiLU,
    ):
        super().__init__()
        self.cfg = cfg
        self.in_c_dict = in_c_dict  # {'layer1': 256}
        self.layer_list = list(self.in_c_dict.keys())
        self.act_fn = act_fn
        self.num_voxel_dict = num_voxel_dict  # {'subject1': 1000}
        self.neuron_coords_dict = neuron_coords_dict  # {'subject1': [1000, 3]}
        for k in self.neuron_coords_dict.keys():
            self.neuron_coords_dict[k].requires_grad = False
        self.num_neuron_latent = self.cfg.MODEL.NEURON_PROJECTOR.NUM_NEURON_LATENT
        assert self.num_neuron_latent == 1
        self.subject_list = list(self.num_voxel_dict.keys())

        self.planes = self.cfg.MODEL.NECK.CONV_HEAD.WIDTH

        self.neuron_projectors = nn.ModuleDict()
        self.layer_gates = nn.ModuleDict()  # empty for backward compatibility
        self.mean_method = self.cfg.MODEL.LAYER_GATE.MEAN

        self.voxel_outs = nn.ModuleDict()

        for subject in self.subject_list:
            self.add_subject(subject, self.neuron_coords_dict[subject], overwrite=True)

        self.previous_layer_requires_grad = False

    def add_subject(
        self,
        subject,
        neuron_coords,
        overwrite=False,
        use_linear=True,
        nonlinear_depth=3,
        nonlinear_planes=32,
    ):
        if subject in self.subject_list and not overwrite:
            return
        if subject not in self.subject_list:
            self.subject_list.append(subject)

        neuron_coords.requires_grad = False
        num_voxels = neuron_coords.shape[0]
        num_layers = len(self.layer_list)
        self.num_voxel_dict[subject] = num_voxels
        self.neuron_coords_dict[subject] = neuron_coords

        self.neuron_projectors[subject] = NeuronProjector(
            self.cfg, self.layer_list, neuron_coords
        )

        if use_linear:
            self.voxel_outs[subject] = VoxelOutBlock(
                # self.planes * num_layers,
                self.planes,
                self.num_voxel_dict[subject],
                depth=1,
            )
            # self.voxel_outs[subject] = LinearBlock(
            #     self.planes,
            #     self.num_voxel_dict[subject],
            # )
        else:
            self.voxel_outs[subject] = VoxelOutBlock(
                # self.planes * num_layers,
                self.planes,
                self.num_voxel_dict[subject],
                depth=nonlinear_depth,
                planes=nonlinear_planes,
            )

    def _forward_i(
        self,
        x,
        x_shift,
        indices,
        subject_id,
        session_id,
        eye_coords,
        voxel_indices=None,
        chuck_size=8000,
    ):
        # x: {layer1: [b, c, h, w]}, x_indices for x, rest is indexed
        eye_coords = eye_coords[indices] if eye_coords is not None else None

        b = len(indices)
        d = self.num_neuron_latent

        def _grid_y(voxel_indices):
            grids, gate, reg_mu = self.neuron_projectors[subject_id](
                b, eye_coords, voxel_indices
            )

            out_ys = None
            # out_ys = []
            for i, (k, v) in enumerate(x.items()):
                w = gate[:, i]  # n
                w = rearrange(w, "n -> 1 1 n 1")
                grid = grids[k]  # b, n, d, 2
                out_y = F.grid_sample(
                    v[indices],
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )  # b, c, n, d
                # out_ys.append(out_y)
                if self.mean_method == "mean":
                    if (
                        not self.cfg.MODEL.LAYER_GATE.SKIP
                        and self.cfg.OPTIMIZER.GATE_REGULARIZER < 100
                    ):
                        out_y = out_y * w
                    if out_ys is None:
                        out_ys = out_y
                    else:
                        out_ys += out_y
                elif self.mean_method == "geometric_mean":
                    raise NotImplementedError("don't use geometric mean")
                    out_y = out_y**w
                    if out_ys is None:
                        out_ys = out_y
                    else:
                        out_ys *= out_y
                else:
                    raise NotImplementedError
            # out_ys = torch.cat(out_ys, dim=1)
            out_ys = out_ys * (1 / len(x))
            return out_ys, gate, reg_mu

        def divide_chunks(l, n):
            chunks = []
            for i in range(0, len(l), n):
                chunks.append(l[i : i + n])
            return chunks

        def forward_one_chuck(voxel_indices, grad_flag):
            if grad_flag:
                y, gate_weights, reg_mu = _grid_y(voxel_indices)
            else:
                with torch.no_grad():
                    y, gate_weights, reg_mu = _grid_y(voxel_indices)
            y = rearrange(y, "b c n d -> b n (c d)")
            out = self.voxel_outs[subject_id](y, voxel_indices)
            return out, gate_weights, reg_mu

        if voxel_indices == ... or voxel_indices is None:
            voxel_indices = torch.arange(
                self.num_voxel_dict[subject_id], device=x[list(x.keys())[0]].device
            )

        voxel_index_chunks = divide_chunks(voxel_indices, chuck_size)

        grad_flag = self.training and (
            next(
                self.neuron_projectors[subject_id].projectors.parameters()
            ).requires_grad
            or next(
                self.neuron_projectors[subject_id].layer_gate.parameters()
            ).requires_grad
            or next(self.voxel_outs[subject_id].parameters()).requires_grad
            or self.previous_layer_requires_grad
        )
        if not grad_flag:
            outs = []
            for vi in voxel_index_chunks:
                out, gate_weights, reg_mu = forward_one_chuck(vi, grad_flag)
                outs.append(out)
            out = (
                torch.cat(outs, dim=1)
                if len(outs) > 0
                else torch.tensor([0 for _ in range(b)])
            )
            reg_gate = torch.tensor(0.0)
            reg_mu = (torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0))
        else:
            outs = []
            gate_weights = []
            reg_mus = []
            for vi in voxel_index_chunks:
                out, gate_weight, reg_mu = forward_one_chuck(vi, grad_flag)
                outs.append(out)
                gate_weights.append(gate_weight)
                reg_mus.append(reg_mu)
            out = (
                torch.cat(outs, dim=1)
                if len(outs) > 0
                else torch.tensor([0 for _ in range(b)])
            )
            gate_weights = torch.cat(gate_weights, dim=0)

            def entropy(x):
                return (x * x.log()).sum(dim=1).mean()

            reg_gate = entropy(gate_weights)
            # reg_gate = torch.tensor(0.0)
            reg_mu1 = torch.stack([x[0] for x in reg_mus], dim=0).mean()
            reg_mu2 = torch.stack([x[1] for x in reg_mus], dim=0).mean()
            reg_mu3 = torch.stack([x[2] for x in reg_mus], dim=0).mean()
            reg_mu = (reg_mu1, reg_mu2, reg_mu3)

        reg_p_mu_shift = [0.0] * b

        return out, reg_gate, reg_mu, reg_p_mu_shift

    def forward(
        self,
        x: Dict[str, Tensor],  # shape (B, C, H, W)
        subject_ids: List[str],  # shape (B,)
        session_ids: List[str] = None,  # shape (B,)
        eye_coords: List[Tensor] = None,  # shape (B, 2)
        voxel_indices_dict: Dict[str, Tensor] = None,  # [N]
        x_shift=None,
    ) -> List[Tensor]:
        # for transformer, we need to rearrange the shape
        for k, v in x.items():
            if v.shape[-1] != v.shape[-2]:
                x[k] = rearrange(v, "b h w c -> b c h w")
        # x: {'layer1': [B, 256, H, W]}
        if isinstance(subject_ids, list):
            subject_ids = np.array(subject_ids)
        if isinstance(session_ids, list):
            session_ids = np.array(session_ids)

        out = [None for _ in range(len(subject_ids))]
        reg = [0.0 for _ in range(len(subject_ids))]
        unique_subject_ids = np.unique(subject_ids)
        for i_sub in unique_subject_ids:
            indices1 = subject_ids == i_sub
            indices1 = np.where(indices1)[0]
            unique_session_ids = np.unique(session_ids[indices1])
            for i_sess in unique_session_ids:
                indices2 = session_ids[indices1] == i_sess
                indices2 = np.where(indices2)[0]
                indices = indices1[indices2]
                i_out, i_reg_gate, i_reg_mu, reg_p_mu_shift = self._forward_i(
                    x,
                    x_shift,
                    indices,
                    i_sub,
                    i_sess,
                    eye_coords,
                    voxel_indices=voxel_indices_dict[i_sub]
                    if voxel_indices_dict is not None
                    else None,
                )
                for i, idx in enumerate(indices):
                    out[idx] = i_out[i]
                    i_reg = (
                        i_reg_gate * self.cfg.OPTIMIZER.GATE_REGULARIZER
                        if self.cfg.OPTIMIZER.GATE_REGULARIZER < 100
                        else 0.0
                        + i_reg_mu[0] * self.cfg.OPTIMIZER.MU_REGULARIZER_PDIST
                        + i_reg_mu[1] * self.cfg.OPTIMIZER.MU_REGULARIZER_PCENTER
                        + i_reg_mu[2] * self.cfg.OPTIMIZER.MU_REGULARIZER_MCENTER
                        # + reg_x_shift_smooth[idx]
                        # * self.cfg.OPTIMIZER.X_SHIFT_SMOOTH_REGULARIZER
                        + reg_p_mu_shift[i] * self.cfg.OPTIMIZER.P_MU_SHIFT_REGULARIZER
                    )
                    reg[idx] = i_reg
        return out, reg, x_shift

    @property
    def device(self):
        return next(self.parameters()).device
