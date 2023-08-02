import argparse
import logging

import os
from copy import copy, deepcopy
from typing import Any, Dict, List, Optional, Tuple
from matplotlib import pyplot as plt

import numpy as np
import pytorch_lightning as pl
import torch

import torch.nn.functional as F

# +
from einops import rearrange, repeat

from torch import Tensor, nn

from backbone import build_backbone
from common_utils import count_nan
from config import AutoConfig
from config_utils import convert_to_dict, load_from_yaml, save_to_yaml
from datamodule import NSDDatamodule
from loss import build_loss
from neck import build_neck

from torchmetrics import (
    MetricCollection,
    R2Score,
    MeanSquaredError,
    MeanAbsoluteError,
    PearsonCorrCoef,
)

from abc import ABC, abstractmethod

from models import DevMemVoxelWiseEncodingModel, DevVoxelWiseEncodingModel, MemVoxelWiseEncodingModel

logger = logging.getLogger(__name__)


class EMAMetric(nn.Module):
    def __init__(self, beta=0.9, bias_correction=False):
        super().__init__()
        self.beta = beta
        self.running_v = None
        self.prev_running_v = None
        self.prev_v = None
        self.running_grad = None
        self.bias_correction = bias_correction
        self.t = 0

    def update(self, v):
        if self.running_v is None:
            self.running_v = torch.zeros_like(v)
            self.prev_running_v = torch.zeros_like(v)
            self.prev_v = torch.zeros_like(v)
            self.running_grad = torch.zeros_like(v)
            self.prev_grad = torch.zeros_like(v)

        g = v - self.prev_v
        self.prev_v = copy(v)
        self.running_grad = (1 - self.beta) * g + self.beta * self.running_grad

        self.prev_running_v = copy(self.running_v)
        self.running_v = (1 - self.beta) * v + self.beta * self.running_v

        self.t += 1
        if self.bias_correction:
            self.running_v /= 1 - self.beta**self.t
            self.running_grad /= 1 - self.beta**self.t
        return self.running_v

    def get_gradient(self):
        if self.prev_running_v is None:
            self.prev_running_v = torch.zeros_like(self.running_v)
        return self.running_v - self.prev_running_v

    @staticmethod
    def normalize(x):
        return (x - x.mean()) / x.std()

    def get_status(self):
        return {
            "running_v": self.running_v,
            "running_grad": self.running_grad,
            "running_v_grad": self.get_gradient(),
            "vgrad": self.running_v * self.get_gradient(),
            "nvgrad": self.normalize(self.running_v) * self.get_gradient(),
        }


class PlVEModel(pl.LightningModule):
    def __init__(
        self,
        cfg: AutoConfig,
        roi_dict: Dict[str, Dict[str, Tensor]],
        coord_dict: Dict[str, Tensor],
    ):
        super().__init__()
        self.cfg = cfg
        self._cfg_hparams = convert_to_dict(self.cfg.clone())
        self.save_hyperparameters(self._cfg_hparams)
        self.roi_dict = roi_dict
        self.coord_dict = coord_dict
        self.n_voxel_dict = {s: len(v) for s, v in self.coord_dict.items()}
        self.subject_list = list(self.n_voxel_dict.keys())

        # make coords nn.Parameter so they can automatically be moved to device
        self.coord_dict = nn.ParameterDict(
            {s: nn.Parameter(v) for s, v in self.coord_dict.items()}
        )
        self.coord_dict.requires_grad_(False)

        if self.cfg.EXPERIMENTAL.USE_DEV_MODEL:
            self.model = DevMemVoxelWiseEncodingModel(self.cfg, self.n_voxel_dict)
        else:
            self.model = MemVoxelWiseEncodingModel(self.cfg, self.n_voxel_dict)

        self.loss = build_loss(self.cfg)

        self.metrics = (
            nn.ModuleDict()
        )  # {'TRAIN': {"NSD_01": {"early": MetricCollection}, 'VAL': {}, 'TEST': {}}
        self.init_metrics()

        self.ema_score = nn.ModuleDict()  # {"TRAIN": {"NSD_01": v}, "VAL": {}}
        self.init_emas()

        self.voxel_weight = {}
        for s in self.subject_list:
            self.voxel_weight[s] = 1.0

        # for FinetuneEachVoxelCallback
        self.voxel_score = {}

        # for prediction_step
        self.predict_vi_dict = None

    def init_metrics(self):
        self.metrics = (
            nn.ModuleDict()
        )  # {'TRAIN': {"NSD_01": {"early": MetricCollection}, 'VAL': {}, 'TEST': {}}
        for stage in ["TRAIN", "VAL", "TEST"]:
            self.metrics.update({stage: nn.ModuleDict()})
            for s in self.subject_list:
                self.metrics[stage].update({s: nn.ModuleDict()})
                for roi in self.roi_dict[s].keys():
                    num_voxels = self.n_voxel_dict[s]
                    if roi == "all":
                        num_voxels = num_voxels
                    else:
                        num_voxels = self.roi_dict[s][roi].shape[0]
                    if (
                        stage == "TRAIN"
                        and num_voxels > self.cfg.MODEL.MAX_TRAIN_VOXELS
                    ):
                        num_voxels = self.cfg.MODEL.MAX_TRAIN_VOXELS
                    m = MetricCollection(
                        [
                            # MeanSquaredError(),
                            MeanAbsoluteError(),
                            PearsonCorrCoef(num_outputs=num_voxels),
                            # R2Score(num_outputs=num_voxels),
                        ]
                    )
                    self.metrics[stage][s].update(
                        {roi: m.clone(prefix=f"{stage}/", postfix=f"/{s}/{roi}")}
                    )
                if stage == 'TRAIN':
                    self.metrics[stage][s].update(
                        {'dark': m.clone(prefix=f"{stage}/", postfix=f"/{s}/dark")}
                    )

    def init_emas(self):
        self.ema_score = nn.ModuleDict()  # {"TRAIN": {"NSD_01": v}, "VAL": {}}
        for stage in ["TRAIN", "VAL"]:
            self.ema_score.update({stage: nn.ModuleDict()})
            for s in self.subject_list:
                self.ema_score[stage].update(
                    {
                        s: EMAMetric(
                            beta=self.cfg.LOSS.SYNC.EMA_BETA,
                            bias_correction=self.cfg.LOSS.SYNC.EMA_BIAS_CORRECTION,
                        )
                    }
                )

    def on_fit_start(self) -> None:
        yaml_path = os.path.join(self.logger.log_dir, "config.yaml")
        save_to_yaml(self.cfg, yaml_path)

    def _from_batch(self, batch):
        y = batch[3]
        dark = batch[4]
        subject_name = batch[-2]
        return y, dark, subject_name

    def forward_batch(self, batch, voxel_indices_dict=None):
        (
            img,
            prev_img,
            prev_feats,
            y,
            dark,
            bhv,
            prev_bhvs,
            ssid,
            subject_name,
            data_idx,
        ) = batch
        bsz = img.shape[0]

        def assign(outs, out, idx):
            for i in range(len(idx)):
                o = out[i]
                # o[torch.isnan(o)] = 0.0  # ignore nan
                outs[idx[i]] = o

        outs = [None for _ in range(bsz)]
        reg_layers = []
        unique_subjects = np.unique(subject_name)
        for s in unique_subjects:
            idx = np.where(subject_name == s)[0]

            _vi = voxel_indices_dict[s] if voxel_indices_dict is not None else None

            tup = self.model.forward(
                img[idx],
                s,
                self.coord_dict[s],
                bhv=bhv[idx],
                prev_img=prev_img[idx] if prev_img is not None else None,
                prev_feats=prev_feats[idx] if prev_feats is not None else None,
                prev_bhvs=prev_bhvs[idx],
                voxel_indices=_vi,
                chunk_size=self.cfg.MODEL.CHUNK_SIZE,
            )
            if self.training:
                out, reg_layer = tup
                assign(outs, out, idx)
                reg_layers.append(reg_layer)
            else:
                out = tup
                assign(outs, out, idx)

        if self.training:
            reg_layer = torch.stack(reg_layers, dim=0).sum()
            return outs, reg_layer
        else:
            return outs

    def training_step(self, batch, batch_idx):
        stage = "TRAIN"

        voxel_indices_dict = {}  # {subject_id: [N]} reduce memory usage
        for s in self.subject_list:
            n = self.n_voxel_dict[s]
            voxel_indices = ...
            if n > self.cfg.MODEL.MAX_TRAIN_VOXELS:
                voxel_indices = torch.randperm(n)[: self.cfg.MODEL.MAX_TRAIN_VOXELS]
            voxel_indices_dict[s] = voxel_indices

        outs, reg_layer = self.forward_batch(batch, voxel_indices_dict)
        ys, darks, subject_names = self._from_batch(batch)

        n_voxels = []
        for i, (s, y) in enumerate(zip(subject_names, ys)):
            vi = voxel_indices_dict[s]
            n_voxels.append(vi.shape[0] if vi != ... else y.shape[0])
        total_voxels = sum(n_voxels)

        batch_loss = []
        for i, (s, o, y, d) in enumerate(zip(subject_names, outs, ys, darks)):
            vi = voxel_indices_dict[s]
            y = y[vi].unsqueeze(0)
            o = o.unsqueeze(0)
            if self.cfg.LOSS.DARK.USE:
                if torch.all(d == 0.0):
                    logging.error(f"dark knowledge is all 0 for {s}, check if file exists")
                    d = y
                d = d[vi].unsqueeze(0)
                gt_voxel_loss = self.loss(o, y).squeeze(0)  # [N]
                dark_voxel_loss = self.loss(o, d).squeeze(0)  # [N]
                voxel_loss = gt_voxel_loss * self.gt_weight + dark_voxel_loss * self.darkness_weight
            else:
                voxel_loss = self.loss(o, y).squeeze(0)  # [N]
            # replace nan with 0
            if torch.isnan(voxel_loss).any():
                count, percentage = count_nan(voxel_loss)
                logging.warning(f"loss is nan, replacing {count} ({percentage:.2%})")
                voxel_loss[torch.isnan(voxel_loss)] = 0.0
                y_count, y_percentage = count_nan(y)
                o_count, o_percentage = count_nan(o)
                logging.warning(
                    f"y: {y_count} ({y_percentage:.2%}), o: {o_count} ({o_percentage:.2%})"
                )

            # reweight by ema_score
            w_v = self.voxel_weight[s]
            w_v = 1.0 if isinstance(w_v, float) else w_v[vi]
            voxel_loss = voxel_loss * w_v

            # every voxel has the same weight, across subjects
            voxel_loss = voxel_loss.mean() * n_voxels[i] / total_voxels

            batch_loss.append(voxel_loss)

            self.metrics[stage][s]["all"].update(o.float().detach(), y.float().detach())
            if self.cfg.LOSS.DARK.USE:
                self.metrics[stage][s]["dark"].update(o.float().detach(), d.float().detach())                

        loss = torch.stack(batch_loss).sum()
        reg_layer *= self.cfg.REGULARIZER.LAYER
        loss += reg_layer
        return loss

    def _shared_eval_step(
        self, batch, batch_idx, stage, is_log=True
    ) -> Tuple[List[Tensor], Tensor, List[Tensor]]:
        outs = self.forward_batch(batch)
        ys, _, subject_names = self._from_batch(batch)

        for i, (s, o, y) in enumerate(zip(subject_names, outs, ys)):
            y = y.unsqueeze(0)
            o = o.unsqueeze(0)
            for roi in self.roi_dict[s].keys():
                if roi == "all":
                    self.metrics[stage][s][roi].update(o.float(), y.float())
                else:
                    roi_idx = self.roi_dict[s][roi]
                    self.metrics[stage][s][roi].update(
                        o[:, roi_idx].float(),
                        y[:, roi_idx].float(),
                    )

    def validation_step(self, batch, batch_idx):
        self._shared_eval_step(batch, batch_idx, "VAL")

    def test_step(self, batch, batch_idx):
        self._shared_eval_step(batch, batch_idx, "TEST")

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        outs = self.forward_batch(batch)
        ys, _, subject_names = self._from_batch(batch)
        return outs
    
    @property
    def darkness_weight(self):
        max_epoch = self.cfg.LOSS.DARK.MAX_EPOCH
        epoch = self.current_epoch
        rate = 1.0 - (epoch / max_epoch)
        rate = max(rate, 0.0)
        return rate

    @property
    def gt_weight(self):
        max_epoch = self.cfg.LOSS.DARK.MAX_EPOCH
        epoch = self.current_epoch
        rate = epoch / max_epoch
        rate = min(rate, 1.0)
        return rate

    @torch.no_grad()
    def update_voxel_weight_by_ema(self):
        if self.cfg.LOSS.SYNC.EXP_SCALE == 0:
            return

        stage = self.cfg.LOSS.SYNC.STAGE
        v_dict = {}
        all_v = []
        for s in self.subject_list:
            ema: EMAMetric = self.ema_score[stage][s]
            v = ema.get_status()[self.cfg.LOSS.SYNC.EMA_KEY]
            if self.cfg.LOSS.SYNC.UPDATE_RULE == "exp":
                v = torch.exp(
                    v * self.cfg.LOSS.SYNC.EXP_SCALE + self.cfg.LOSS.SYNC.EXP_SHIFT
                )
            elif self.cfg.LOSS.SYNC.UPDATE_RULE == "square":
                v = v**2
            elif self.cfg.LOSS.SYNC.UPDATE_RULE == "raw":
                v = v
            elif self.cfg.LOSS.SYNC.UPDATE_RULE == "log":  # recommended
                v = torch.log(v + self.cfg.LOSS.SYNC.LOG_SHIFT)
            elif self.cfg.LOSS.SYNC.UPDATE_RULE == "norm":
                std = v.std()
                mean = v.mean()
                v = (v - mean) / std
                v = torch.clamp(v, -3, 3)
                # grad = (grad - grad.min()) / (grad.max() - grad.min())
            elif self.cfg.LOSS.SYNC.UPDATE_RULE == "none":
                return  # do nothing
            else:
                raise NotImplementedError
            v_dict[s] = v
            all_v.append(v)
        all_v = torch.cat(all_v, dim=0)
        vmax, vmin = all_v.max(), all_v.min()

        for s in self.subject_list:
            v = v_dict[s]
            v = (v - vmin) / (vmax - vmin)
            self.voxel_weight[s] = v

    @torch.no_grad()
    def _shared_epoch_end(self, outputs, stage):
        # if self.global_step == 0:
        #     if not hasattr(self, "zero_flag"):
        #         return

        voxel_metric_dict = {}
        nsd_all_v = []
        all_v = []
        for subject_id in self.subject_list:
            log_vi = ...
            if hasattr(self, "dark_gt_vis") and self.dark_gt_vis:
                if subject_id not in self.dark_gt_vis:
                    logging.warning(
                        f"Epoch end: subject_id {subject_id} not in dark_gt_vis"
                    )
                else:
                    log_vi = self.dark_gt_vis[subject_id]

            voxel_metric_dict[subject_id] = {}  # for saving
            for roi in list(self.roi_dict[subject_id].keys()) + ['dark']:
                if stage == "TRAIN" and roi not in ["all", "dark"]:
                    # skip roi for training
                    continue
                if stage != "TRAIN" and roi == "dark":
                    continue
                metric_dict = self.metrics[stage][subject_id][roi].compute()
                for k, v in metric_dict.items():
                    if torch.isnan(v).any():
                        logging.warning(f"Epoch end: NaN in {k} for {subject_id}")
                        v[torch.isnan(v)] = 0
                    metric_dict[k] = v

                mean_d = {}
                for k, v in metric_dict.items():
                    if roi == "all":
                        voxel_metric_dict[subject_id][k] = v.detach().cpu().numpy()
                        if stage != "TEST":
                            if "PearsonCorrCoef" in k:
                                if self.cfg.LOSS.SYNC.USE:
                                    self.ema_score[stage][subject_id].update(v)
                                if stage == "VAL":  # for FinetuneEachVoxelCallback
                                    self.voxel_score[subject_id] = (
                                        v.detach().cpu().numpy()
                                    )
                    if roi == "all" and stage == "TEST" and "PearsonCorrCoef" in k:
                        self.voxel_score[subject_id] = v.detach().cpu().numpy()
                    mean_d[k] = torch.mean(v)

                self.log_dict(mean_d, sync_dist=self.cfg.TRAINER.DDP)

                # for early stopping
                if roi == "all":
                    vs = metric_dict[f"{stage}/PearsonCorrCoef/{subject_id}/{roi}"]
                    if stage == "TRAIN":
                        all_v.append(vs)
                    else:
                        all_v.append(vs[log_vi])

        # for early stopping
        all_v_mean = torch.mean(torch.cat(all_v))
        self.log(f"{stage}/PearsonCorrCoef/mean", all_v_mean, sync_dist=self.cfg.TRAINER.DDP)

        # save to disk
        log_dir = self.logger.log_dir
        epoch = self.current_epoch
        step = self.global_step
        save_dir = os.path.join(log_dir, f"voxel_metric")
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"stage={stage}.step={step:012d}.pkl")
        np.save(path, voxel_metric_dict, allow_pickle=True)

        for subject_id in self.subject_list:
            for roi in self.roi_dict[subject_id].keys():
                self.metrics[stage][subject_id][roi].reset()

        # update voxel_weight by ema
        if (
            self.cfg.LOSS.SYNC.USE
            and stage == self.cfg.LOSS.SYNC.STAGE
            and self.global_step > 0
            and epoch >= self.cfg.LOSS.SYNC.SKIP_EPOCHS
        ):
            self.update_voxel_weight_by_ema()

        return all_v_mean

    def training_epoch_end(self, outputs):
        stage = "TRAIN"
        self._shared_epoch_end(outputs, stage)

    def validation_epoch_end(self, outputs):
        stage = "VAL"
        s = self._shared_epoch_end(outputs, stage)

    def test_epoch_end(self, outputs):
        stage = "TEST"
        s = self._shared_epoch_end(outputs, stage)
        if s is not None:
            self.log("hp_metric", s)

    def configure_optimizers(self):
        from optimizers import build_optimizer

        base_lr = self.cfg.OPTIMIZER.LR
        base_wd = self.cfg.OPTIMIZER.WEIGHT_DECAY

        no_decay = [
            "bias",
            "BatchNorm3D.weight",
            "BatchNorm1D.weight",
            "BatchNorm2D.weight",
            "LayerNorm.weight",
        ]

        optimizer_grouped_parameters = []
        for n, p in self.named_parameters():
            if p.requires_grad == False:
                continue
            if not any(nd in n for nd in no_decay):
                optimizer_grouped_parameters.append(
                    {"params": p, "lr": base_lr, "weight_decay": base_wd}
                )
            else:
                optimizer_grouped_parameters.append(
                    {"params": p, "lr": base_lr, "weight_decay": 0}
                )

        return build_optimizer(self.cfg, optimizer_grouped_parameters)

    def lr_scheduler_step(self, scheduler, *args, **kwargs):
        scheduler.step(
            epoch=self.current_epoch
        )  # timm's scheduler need the epoch value
        
    
    # for plotting
    def get_retinamapper_layerselector_output(self):
        voxel_indices = ...
        mu_dict = {}
        w_dict = {}
        for s in self.subject_list:
            coords = self.coord_dict[s]
            mu = self.model.retina_mapper[s](coords, voxel_indices) # [N, 2]
            w = self.model.layer_selector[s](coords, voxel_indices) # [N, 4]
            mu_dict[s] = mu
            w_dict[s] = w
        return mu_dict, w_dict

if __name__ == "__main__":
    from config_utils import get_cfg_defaults
    
    cfg = get_cfg_defaults()
    cfg.DATAMODULE.BATCH_SIZE = 8
    cfg.TRAINER.ACCUMULATE_GRAD_BATCHES = 1
    cfg.OPTIMIZER.LR = 1e-3
    cfg.OPTIMIZER.NAME = "AdamW"
    cfg.DATASET.ROIS = ["all"]
    cfg.DATASET.FMRI_SPACE = 'fsaverage'
    cfg.DATASET.SUBJECT_LIST = ["subj01"]
    cfg.TRAINER.CALLBACKS.EARLY_STOP.PATIENCE = 10
    cfg.MODEL.CONV_HEAD.SIMPLE = True
    cfg.MODEL.CONV_HEAD.WIDTH = 256
    cfg.MODEL.CONV_HEAD.MAX_DIM = 1024

    cfg.MODEL.MAX_TRAIN_VOXELS = 25600
    cfg.DATASET.N_PREV_FRAMES = 32

    cfg.TRAINER.PRECISION = 16
    
    cfg.RESULTS_DIR = 'tb_logs'
    cfg.DATASET.ROOT = '/data/ALG23'
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, default=None)
    args = parser.parse_args()
    if args.cfg_path is not None:
        # overwrite above cfg
        cfg = load_from_yaml(args.cfg_path)
    
    from train_utils import simple_train
    
    simple_train(cfg)