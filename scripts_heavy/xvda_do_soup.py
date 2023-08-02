import copy

import pandas as pd
import torch
from cluster_utils import my_nfs_cluster_job, trial_dirname_creator

import argparse
import os
import sys
from random import seed, shuffle

import numpy as np
import ray
from ray import tune
from config import AutoConfig

from config_utils import dict_to_list, get_cfg_defaults, load_from_yaml

from read_utils import (
    find_runs_from_exp_dir,
    read_config,
    read_short_config,
    read_score_df,
    list_runs_from_exp_names,
)
from dark_onemodel import build_dmt, get_outs


def get_parser():
    parser = argparse.ArgumentParser(description="Ray Tune")
    parser.add_argument("--ckpt_dir", type=str, default="/data/dckpt/", help="ckpt dir")
    return parser


args = get_parser().parse_args()

runs = os.listdir(args.ckpt_dir)
for run in runs:
    _d = os.path.join(args.ckpt_dir, run)
    ckpts = os.listdir(_d)
    ckpts = [os.path.join(_d, ckpt) for ckpt in ckpts]
    ckpts = [ckpt for ckpt in ckpts if ckpt.endswith(".ckpt")]
    
    soup_state_dict = None
    n_ingredients = 0
    for ckpt in ckpts:
        print(f"loading {ckpt}")
        state_dict = torch.load(ckpt, map_location='cpu')["state_dict"]
        if soup_state_dict is None:
            soup_state_dict = copy.deepcopy(state_dict)
        else:
            soup_state_dict = {k: v + soup_state_dict[k] for k, v in state_dict.items()}
        n_ingredients += 1
    soup_state_dict = {k: v / n_ingredients for k, v in soup_state_dict.items()}
    
    torch.save(soup_state_dict, os.path.join(_d, "soup.pth"))