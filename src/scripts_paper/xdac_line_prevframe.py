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
        "--save_dir", type=str, default="/nfscc/fig/alg23/xdac/", help="save dir"
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

BIG_ROIS = ["Primary_Visual", "Visual", "Somatomotor", "Auditory", "Posterior", "Anterior"]
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

# xticks = ["-32:0"]
xticks = []
xticks += [f"{t}" for t in range(0, -32, -1)]
xticks += ["rand"]
xticks = ["T=0", "", "", "", "-4", "", "", "", "-8", "", "", "", "-12", "", "", "", "-16", "", "", "", "-20", "", "", "", "-24", "", "", "", "-28", "", "", "", "rand"]

xs = np.arange(len(xticks))

a_colors = plt.cm.Dark2(range(8))
b_colors = plt.cm.tab10(range(10))

def make_plot(rois, save_name, colors):
    # 2x4 plot for each subject
    SUBJECTS = [f"subj{i+1:02d}" for i in range(8)]
    fig, axes = plt.subplots(2, 4, figsize=(30, 7))
    for i, subject in enumerate(SUBJECTS):
        handles = []
        labels = []
        ax = axes[i // 4, i % 4]
        df_subj = df[df.subject == subject]
        for roi in rois:
            values = []
            for t, all_t, rand in order:
                v = df_subj[
                    (df_subj.t == t) & (df_subj.all_t == all_t) & (df_subj.rand == rand)
                ].vs.values[0][ROIS.index(roi)]
                values.append(v.mean())
            ax.plot(xs, values, label=roi, alpha=0.8, color=colors[rois.index(roi)])
            linewidth = 2 if roi not in ['Visual', 'HT'] else 5
            line, = ax.plot(xs, values, label=roi, alpha=0.8, color=colors[rois.index(roi)], linewidth=linewidth)
            handles.append(line)
            labels.append(roi)
        # ax.set_title(subject, fontsize=16)
        # ax.text(0.5, 1.05, subject, fontsize=16, transform=ax.transAxes, horizontalalignment='center')
        # add text at top right corner
        text = f"Subject#{i+1}"
        ax.text(0.95, 0.95, text, fontsize=20, transform=ax.transAxes, horizontalalignment='right', verticalalignment='top')
        ax.set_xticks(range(len(xticks)))
        ax.set_xticklabels(xticks, rotation=0, fontsize=20)
        ax.set_ylim(-0.025, 0.125)
        ax.set_yticks([-0.02, 0, 0.05, 0.1])
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_yticklabels(["-0.02", "0", "0.05", "0.1"], fontsize=20)
        if i % 4 == 0:
            ax.set_ylabel("Pearson's R", fontsize=20)
        
        for roi in rois:
            values = []
            for t, all_t, rand in order:
                v = df_subj[
                    (df_subj.t == t) & (df_subj.all_t == all_t) & (df_subj.rand == rand)
                ].vs.values[0][ROIS.index(roi)]
                values.append(v.mean())
            ax.scatter(xs, values, label=roi, alpha=0.8, s=10, color=colors[rois.index(roi)])
            
        ax.grid(linestyle="--", linewidth=1, alpha=0.5)
    # fig.legend(handles, labels, loc="lower center", ncol=len(rois), bbox_to_anchor=(0.0, -0.1))
    fig.legend(handles, labels, loc="lower center", ncol=len(rois), bbox_to_anchor=(0.5, -0.075), fontsize=20)
    plt.tight_layout()
    os.makedirs(args.save_dir, exist_ok=True)
    plt.savefig(os.path.join(args.save_dir, f"{save_name}.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(args.save_dir, f"{save_name}.png"), bbox_inches="tight")
    plt.close()


make_plot(BIG_ROIS, f"{beta}_cortex_rois", a_colors)
make_plot(H_ROIS, f"{beta}_hippocampus_rois", b_colors)

plt.style.use("default")

make_plot(BIG_ROIS, f"{beta}_cortex_rois_light", a_colors)
make_plot(H_ROIS, f"{beta}_hippocampus_rois_light", b_colors)
