# %%
from functools import partial
import logging
import os
import sys
from copy import copy
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
from torch import nn
import torch
from torchvision import transforms
from einops import rearrange, reduce
from torch.utils.data import ConcatDataset, DataLoader

from config import AutoConfig

from datasets import NSDDataset
from point_pe import point_position_encoding


class NSDDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        cfg: AutoConfig,
    ):
        super().__init__()
        self.cfg = cfg
        self.dss = [{}, {}, {}, {}]  # train, val1, val2, predict
        self.stage_list = ["train", "val1", "val2", "predict"]
        self.ds_dict = {}  # pass by reference
        for i, stage in enumerate(self.stage_list):
            self.ds_dict[stage] = self.dss[i]

        self.subject_list = self.cfg.DATASET.SUBJECT_LIST
        if self.subject_list == ['all']:
            self.subject_list = [f'subj{i:02d}' for i in range(1, 9)]
            
        self.batch_size = self.cfg.DATAMODULE.BATCH_SIZE

    @property
    def num_voxel_dict(self):
        ret = {}
        ds_dict = self.dss[0]
        for name, ds in ds_dict.items():
            ret[name] = ds.num_voxels
        return ret

    @property
    def roi_dict(self):
        ret = {}
        ds_dict = self.dss[0]
        for subject_name, ds in ds_dict.items():
            ret[subject_name] = ds.roi_dict
        return ret

    @property
    def neuron_coords_dict(self):
        ret = {}
        ds_dict = self.dss[0]
        for subject_name, ds in ds_dict.items():
            ret[subject_name] = ds.neuron_coords
        return ret

    @property
    def collate_fn(self):
        return list(self.dss[0].values())[0].collate_fn

    def train_dataloader(self, subject=None, shuffle=True):
        idx = 0
        if subject is None:
            ds = ConcatDataset(list(self.dss[idx].values()))
        else:
            ds = self.dss[idx][subject]
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            num_workers=self.cfg.DATAMODULE.NUM_WORKERS,
            pin_memory=self.cfg.DATAMODULE.PIN_MEMORY,
        )

    def val_dataloader(self, subject=None, shuffle=True):
        if shuffle == True:
            shuffle = False
            if self.cfg.EXPERIMENTAL.SHUFFLE_VAL:
                shuffle = True
        idx = 1
        if subject is None:
            ds = ConcatDataset(list(self.dss[idx].values()))
        else:
            ds = self.dss[idx][subject]
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            num_workers=self.cfg.DATAMODULE.NUM_WORKERS,
            pin_memory=self.cfg.DATAMODULE.PIN_MEMORY,
        )

    def test_dataloader(self, subject=None, shuffle=False):
        idx = 2
        if subject is None:
            ds = ConcatDataset(list(self.dss[idx].values()))
        else:
            ds = self.dss[idx][subject]
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.cfg.DATAMODULE.NUM_WORKERS,
            pin_memory=self.cfg.DATAMODULE.PIN_MEMORY,
        )

    def predict_dataloader(self, subject=None, shuffle=False):
        idx = 3
        if self.dss[idx] == {}:
            return None
        if subject is None:
            ds = ConcatDataset(list(self.dss[idx].values()))
        else:
            ds = self.dss[idx][subject]
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.cfg.DATAMODULE.NUM_WORKERS,
            pin_memory=self.cfg.DATAMODULE.PIN_MEMORY,
        )

    def teardown(self, stage: Optional[str] = None):
        # Used to clean-up when the run is finished
        # self.dss = {}, {}, {}, {}  # train, test, val, predict
        pass

    def setup(self, stage: Optional[str] = None, overwrite: bool = False):

        stages = self.stage_list

        for subject_name in self.subject_list:
            for stage in stages:
                idx = self.stage_list.index(stage)

                ds = NSDDataset(
                    root=self.cfg.DATASET.ROOT,
                    subject_name=subject_name,
                    split=stage,
                    image_resolution=self.cfg.DATASET.IMAGE_RESOLUTION,
                    fmri_space=self.cfg.DATASET.FMRI_SPACE,
                    rois=self.cfg.DATASET.ROIS,
                    dark_postfix=self.cfg.DATASET.DARK_POSTFIX,
                    load_prev_frames=self.cfg.EXPERIMENTAL.USE_PREV_FRAME,
                    filter_by_session=self.cfg.DATASET.FILTER_BY_SESSION,
                    n_prev_frames=self.cfg.DATASET.N_PREV_FRAMES,
                    cfg=self.cfg,
                )

                self.dss[idx].update({subject_name: ds})

        return

    def __repr__(self):
        s = "DataModule: \n"
        for stage in self.stage_list:
            idx = self.stage_list.index(stage)
            num_datas = sum([len(self.dss[idx][sub]) for sub in self.dss[idx]])
            s += f"  {stage}: {num_datas:,} datas, {len(self.dss[idx])} subjects\n"
        return s



# %%
if __name__ == "__main__":
    from config_utils import get_cfg_defaults

    cfg = get_cfg_defaults()
    dm = NSDDatamodule(cfg)
    dm.setup()
