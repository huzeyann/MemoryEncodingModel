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
    read_val_voxel_score,
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
        "--exp_dir", type=str, default="/nfscc/alg23/xbaa/", help="exp dir"
    )
    parser.add_argument(
        "--save_dir", type=str, default="/nfscc/fig/alg23/xbab/", help="save dir"
    )
    parser.add_argument("--overwrite", action="store_true", help="overwrite")
    return parser


args = get_parser().parse_args()


def job(run):
    cfg: AutoConfig = read_config(run)
    subject = cfg.DATASET.SUBJECT_LIST[0]
    t = cfg.EXPERIMENTAL.T_IMAGE
    all_t = cfg.EXPERIMENTAL.USE_PREV_FRAME
    rand = cfg.EXPERIMENTAL.SHUFFLE_IMAGES
    space = cfg.DATASET.FMRI_SPACE
    vs = read_val_voxel_score(run)
    vs = vs[subject][f"VAL/PearsonCorrCoef/{subject}/all"]
    mean_vs = vs.mean(axis=0)
    data = (subject, t, all_t, rand, space, mean_vs)
    return data

# df_path = f'/tmp/xbab.pkl'
# if os.path.exists(df_path):
#     df = torch.load(df_path)
# else:
exp_dir = args.exp_dir
runs = find_runs_from_exp_dir(exp_dir)
print(len(runs))

import multiprocessing as mp

with mp.Pool(16) as pool:
    datas = pool.map(job, runs)

df = pd.DataFrame(
    datas, columns=["subject", "t", "all_t", "rand", "space", "vs"]
).sort_values(["subject", "t", "all_t", "rand", "space"])

# torch.save(df, df_path)

SPACES = ['visual_D', 'visual_B']
SUBJECTS = ['CSI1', 'CSI2', 'CSI3']
PLOT_SUBJECT_NAMES = ['subject#1', 'subject#2', 'subject#3']
    
order = [
    # (0, True, False),
]
order += [(t, True, False) for t in range(0, -32, -1)]
order += [(0, True, True)]

# xticks = ["-32:0"]
xticks = []
xticks += [f"{t}" for t in range(0, -32, -1)]
xticks += ["rand"]
xticks = ["T=0", "", "", "", "-4", "", "", "", "-8", "", "", "", "-12", "", "", "", "-16", "", "", "", "-20", "", "", "", "-24", "", "", "", "-28", "", "", "", "rand"]

xs = np.arange(len(xticks))

a_colors = plt.cm.Dark2(range(8))
b_colors = plt.cm.tab10(range(10))

# make 1x2 plot

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
def plot(ax, space):
    _df = df[df.space == space]
    handles = []
    labels = []
    for i, subject in enumerate(SUBJECTS):
        values = []
        for t, all_t, rand in order:
            v = _df[(_df.subject == subject) & (_df.t == t) & (_df.all_t == all_t) & (_df.rand == rand)].vs.values[0]
            values.append(v.mean())
        name = PLOT_SUBJECT_NAMES[i]
        ax.scatter(xs, values, label=name, alpha=0.8, s=10, color=a_colors[i])
        line, = ax.plot(xs, values, label=name, alpha=0.8, color=a_colors[i])
        handles.append(line)
        labels.append(name)
            
    ax.grid(linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xticks(range(len(xticks)))
    ax.set_xticklabels(xticks, rotation=0, fontsize=16)
    ax.set_ylim(-0.025, 0.125)
    ax.set_yticks([-0.02, 0, 0.05, 0.1])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_yticklabels(["-0.02", "0", "0.05", "0.1"], fontsize=16)
    
    return handles, labels

ax = axes[0]
handles, labels = plot(ax, 'visual_D')
text = f"BOLD5000 beta3"
ax.text(0.95, 0.95, text, fontsize=16, transform=ax.transAxes, horizontalalignment='right', verticalalignment='top')
ax.set_ylabel("Pearson's R", fontsize=16)
ax = axes[1]
handles, labels = plot(ax, 'visual_B')
text = f"BOLD5000 beta2"
ax.text(0.95, 0.95, text, fontsize=16, transform=ax.transAxes, horizontalalignment='right', verticalalignment='top')
fig.legend(handles, labels, loc="lower center", ncol=len(SUBJECTS), bbox_to_anchor=(0.5, -0.075), fontsize=16)
plt.tight_layout()
os.makedirs(args.save_dir, exist_ok=True)
save_name = 'bold5000'
plt.savefig(os.path.join(args.save_dir, f"{save_name}.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(args.save_dir, f"{save_name}.png"), bbox_inches="tight")
plt.close()



# make 2x1 plot
plt.style.use("default")

fig, axes = plt.subplots(2, 1, figsize=(8, 8))
axes = axes.flatten()
def plot(ax, space):
    _df = df[df.space == space]
    handles = []
    labels = []
    for i, subject in enumerate(SUBJECTS):
        values = []
        for t, all_t, rand in order:
            v = _df[(_df.subject == subject) & (_df.t == t) & (_df.all_t == all_t) & (_df.rand == rand)].vs.values[0]
            values.append(v.mean())
        name = PLOT_SUBJECT_NAMES[i]
        # ax.scatter(xs, values, label=name, alpha=0.8, s=10, color=a_colors[i])
        line, = ax.plot(xs, values, label=name, alpha=0.8, color=a_colors[i], linewidth=3)
        handles.append(line)
        labels.append(name)
            
    ax.grid(linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xticks(range(len(xticks)))
    ax.set_xticklabels(xticks, rotation=0, fontsize=16)
    ax.set_ylim(-0.025, 0.125)
    ax.set_yticks([-0.02, 0, 0.05, 0.1])
    ax.set_yticklabels(["-0.02", "0", "0.05", "0.1"], fontsize=16)  
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_yticklabels(["-0.02", "0", "0.05", "0.1"], fontsize=16)
    
    return handles, labels

ax = axes[0]
handles, labels = plot(ax, 'visual_D')
text = f"BOLD5000 beta3"
ax.text(0.95, 0.95, text, fontsize=20, transform=ax.transAxes, horizontalalignment='right', verticalalignment='top')
ax.set_ylabel("Pearson's R", fontsize=20)
ax = axes[1]
handles, labels = plot(ax, 'visual_B')
text = f"BOLD5000 beta2"
ax.text(0.95, 0.95, text, fontsize=20, transform=ax.transAxes, horizontalalignment='right', verticalalignment='top')
ax.set_ylabel("Pearson's R", fontsize=20)
fig.legend(handles, labels, loc="lower center", ncol=len(SUBJECTS), bbox_to_anchor=(0.5, -0.075), fontsize=16)
plt.tight_layout()
os.makedirs(args.save_dir, exist_ok=True)
save_name = 'bold5000_light'
plt.savefig(os.path.join(args.save_dir, f"{save_name}.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(args.save_dir, f"{save_name}.png"), bbox_inches="tight")
plt.close()