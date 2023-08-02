import argparse
import copy
import fnmatch
from functools import partial
import glob
import operator
import os
import sys
from typing import Dict

import numpy as np
import pytorch_lightning as pl
import torch

import shutil

from tqdm import tqdm

from config import AutoConfig
from config_utils import load_from_yaml
from datamodule import NSDDatamodule
from plmodels import PlVEModel

from read_utils import (
    read_config,
    read_short_config,
    read_score_df,
    list_runs_from_exp_names,
)


def build_dmt(run_dir):
    cfg: AutoConfig = read_config(run_dir)
    cfg.TRAINER.LIMIT_VAL_BATCHES = 1.0
    cfg.EXPERIMENTAL.SHUFFLE_VAL = False
    dm = NSDDatamodule(cfg)
    dm.setup()

    plmodel = PlVEModel(cfg, dm.roi_dict, dm.neuron_coords_dict)

    trainer = pl.Trainer(
        accelerator="cuda",
        devices=[0],
        precision=16,
        enable_progress_bar=False,
    )

    return dm, plmodel, trainer


@torch.no_grad()
def get_outs(model, trainer, dataloader):
    outs = trainer.predict(model, dataloader)
    outs = torch.stack(sum(outs, []))
    # outs = outs.cpu().numpy().astype(np.float16)
    outs = outs.cpu().half()
    return outs
