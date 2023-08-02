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
        "--exp_dir", type=str, default="/nfscc/alg23/xdaa/b3", help="exp dir"
    )
    parser.add_argument(
        "--beta", type=str, default="b3", help="beta"
    )
    parser.add_argument(
        "--save_dir", type=str, default="/nfscc/fig/alg23/xdad/", help="save dir"
    )
    parser.add_argument("--overwrite", action="store_true", help="overwrite")
    return parser


args = get_parser().parse_args()

ROIS = (
    []
    + ["Primary_Visual", "Visual", "Somatomotor", "Auditory", "Posterior", "Anterior"]
    + [
        "ErC",
        "area35",
        "area36",
        "PhC",
        "Sub",
        "CA1",
        "CA2",
        "CA3",
        "DG",
        "HT",
    ]
    + ["all"]
)

BIG_ROIS = ["all", "Visual", "Somatomotor", "Auditory", "Posterior", "Anterior"]
H_ROIS = [
    "ErC",
    "area35",
    "area36",
    "PhC",
    "Sub",
    "CA1",
    "CA2",
    "CA3",
    "DG",
    "HT",
]


def job(run):
    cfg: AutoConfig = read_config(run)
    subject = cfg.DATASET.SUBJECT_LIST[0]
    t = cfg.EXPERIMENTAL.T_IMAGE
    all_t = cfg.EXPERIMENTAL.USE_PREV_FRAME
    rand = cfg.EXPERIMENTAL.SHUFFLE_IMAGES
    vs = read_test_voxel_score(run)
    vs = vs[subject][f"TEST/PearsonCorrCoef/{subject}/all"]
    dm = NSDDatamodule(cfg)
    dm.setup()
    ds = dm.dss[0][subject]
    roi_dict = ds.roi_dict
    v_list = []
    for roi in ROIS:
        v = vs[roi_dict[roi]]
        v_list.append(v)
    data = (subject, t, all_t, rand, run, v_list)
    return data
    # datas.append(data)

beta = args.beta
df_path = f'/tmp/xdac_{beta}.pkl'
if os.path.exists(df_path):
    df = torch.load(df_path)
else:
    exp_dir = args.exp_dir.replace('b3', beta)
    runs = find_runs_from_exp_dir(exp_dir)
    print(len(runs))
    
    import multiprocessing as mp

    with mp.Pool(16) as pool:
        datas = pool.map(job, runs)

    df = pd.DataFrame(
        datas, columns=["subject", "t", "all_t", "rand", "run", "vs"]
    ).sort_values(["subject", "t", "all_t", "rand"])

    torch.save(df, df_path)
    
order = [
    # (0, True, False),
]
order += [(t, False, False) for t in range(0, -32, -1)]
order += [(0, True, True)]


row1 = (0, True, False)
row2 = (0, False, False)
row3 = (-6, False, False)
row4 = (-28, False, False)
row5 = (0, True, True)
rows = [row1, row2, row3, row4, row5]
row_names = ["T=-32:0", "T=0", "T=-6", "T=-28", "T=rand"]

def make_table(rois):
    
    datas = []
    for row in rows:
        t, all_t, rand = row
        roi_datas = []
        for roi in rois:
            subject_vs = []
            for subject in df.subject.unique():
                df_subj = df[(df.subject == subject) & (df.t == t) & (df.all_t == all_t) & (df.rand == rand)]
                v = df_subj.vs.values[0][ROIS.index(roi)]
                subject_vs.append(v)
            subject_vs = np.concatenate(subject_vs)
            v = subject_vs.mean()
            roi_datas.append(f"{v:.3f}")
        datas.append(roi_datas)
    
    df_table = pd.DataFrame(datas, columns=rois)
    # add row names
    df_table.insert(0, "T", row_names)
    # add beta name
    df_table.insert(0, "beta", beta)
    
    def print_csv(df):
        print(df.to_csv(index=False, float_format="%.3f"))
    
    print_csv(df_table)
    
    os.makedirs(args.save_dir, exist_ok=True)
    df_table.to_csv(os.path.join(args.save_dir, f"xdad_{beta}.csv"), index=False, float_format="%.3f")

make_table(BIG_ROIS)
make_table(H_ROIS)