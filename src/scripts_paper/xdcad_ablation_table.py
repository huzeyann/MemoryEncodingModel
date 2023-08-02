import argparse
from datamodule import NSDDatamodule
from plmodels import PlVEModel
from read_utils import (
    read_config,
    read_short_config,
    read_score_df,
    list_runs_from_exp_names,
    find_runs_from_exp_dir,
    read_test_voxel_score,
)

import os
import re
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import cortex

from PIL import Image

import copy

import cortex
from matplotlib import pyplot as plt

plt.style.use("dark_background")
from config import AutoConfig


def get_parser():
    parser = argparse.ArgumentParser(description="Ray Tune")
    parser.add_argument(
        "--exp_dir", type=str, default="/nfscc/alg23/xdcac/b3", help="exp dir"
    )
    parser.add_argument(
        "--beta", type=str, default="b3", help="beta"
    )
    parser.add_argument(
        "--save_dir", type=str, default="/nfscc/fig/alg23/xdcad/", help="save dir"
    )
    parser.add_argument("--overwrite", action="store_true", help="overwrite")
    return parser


args = get_parser().parse_args()

ROIS = (
    []
    + ["Primary_Visual", "Visual", "Posterior", "Somatomotor", "Auditory", "Anterior"]
    + ["all"]
)

# BIG_ROIS = ["all", "Visual", "Somatomotor", "Auditory", "Posterior", "Anterior"]

# VISUAL_ROIS = ["Primary_Visual", "Visual", "Posterior", "Somatomotor", "Auditory", "Anterior"]


def job(run):
    cfg: AutoConfig = read_config(run)
    tune_dict = read_short_config(run)
    subject = cfg.DATASET.SUBJECT_LIST[0]
    # t = cfg.EXPERIMENTAL.T_IMAGE
    # all_t = cfg.EXPERIMENTAL.USE_PREV_FRAME
    # rand = cfg.EXPERIMENTAL.SHUFFLE_IMAGES
    row = tune_dict["row"]
    vs = read_test_voxel_score(run)
    vs = vs[subject][f"TEST/PearsonCorrCoef/{subject}/all"]
    dm = NSDDatamodule(cfg)
    dm.setup()
    ds = dm.dss[0][subject]
    roi_dict = ds.roi_dict
    v_list = []
    for roi in ROIS:
        v = vs[roi_dict[roi]].mean()
        v_list.append(v)
    data = (subject, row, run, *v_list)
    return data
    # datas.append(data)

beta = args.beta
df_path = f'/tmp/xdcad_{beta}.pkl'
# if os.path.exists(df_path):
#     df = torch.load(df_path)
# else:
exp_dir = args.exp_dir.replace('b3', beta)
runs = find_runs_from_exp_dir(exp_dir)
print(len(runs))

import multiprocessing as mp

with mp.Pool(16) as pool:
    datas = pool.map(job, runs)

df = pd.DataFrame(
    datas, columns=["subject", "row", "run", *ROIS]
).sort_values(["subject", "row"])

torch.save(df, df_path)
    
    
def print_csv(df):
    print(df.to_csv(index=False, float_format="%.3f"))

hide_col = ['subject', 'run']
df = df.drop(columns=hide_col)

# mean over same row
mean_df = df.groupby(['row']).mean().reset_index()
std_df = df.groupby(['row']).std().reset_index()

print_csv(mean_df)
print_csv(std_df)

