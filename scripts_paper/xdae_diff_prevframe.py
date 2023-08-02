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

N_FSAVERAGE = 327684


def get_parser():
    parser = argparse.ArgumentParser(description="Ray Tune")
    parser.add_argument(
        "--exp_dir", type=str, default="/nfscc/alg23/xdaa/", help="exp dir"
    )
    
    parser.add_argument(
        "--save_dir", type=str, default="/nfscc/fig/alg23/xdae/", help="save dir"
    )

    return parser


args = get_parser().parse_args()

def make_df(runs):
    datas = []
    for run in runs:
        cfg: AutoConfig = read_config(run)
        subject = cfg.DATASET.SUBJECT_LIST[0]
        t = cfg.EXPERIMENTAL.T_IMAGE
        all_t = cfg.EXPERIMENTAL.USE_PREV_FRAME
        rand = cfg.EXPERIMENTAL.SHUFFLE_IMAGES
        datas.append([subject, t, all_t, rand, run])
    df = pd.DataFrame(datas, columns=["subject", "t", "all_t", "rand", "run"]).sort_values(
        ["subject", "t", "all_t", "rand"]
    )
    return df


def mycolormap(th=0.05, vmax=0.3):
    import matplotlib.colors as mcolors

    # Define the colormap
    cmap = plt.cm.get_cmap('bwr')

    # Set the minimum and maximum values
    vmin = -vmax

    # Create a normalization instance to map the data values to the range [0, 1]
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # # Create a custom colormap with grey color for values between -0.05 and 0.05
    # colors = [cmap(norm(vmin))]
    # colors.extend([(0.5, 0.5, 0.5, 1), (0.5, 0.5, 0.5, 1)])
    # colors.extend([cmap(norm(vmax))])
    
    colors = []
    for i in range(0, 1000):
        v = vmin + (vmax - vmin) * i / 1000
        if v < th:
            colors.append((0.3535, 0.3535, 0.3535, 1))
        else:
            colors.append(cmap(norm(v)))

    # Create the custom colormap
    custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', colors)
        
    # # Plot a colorbar to show the colormap
    # plt.imshow([np.linspace(vmin, vmax, 100)], cmap=custom_cmap, aspect='auto')
    # plt.colorbar()
    # plt.savefig('/nfscc/fig/tmp_c.png')
    # plt.close()
    
    return custom_cmap


def plot_one_run(vs, png_path):
    if os.path.exists(png_path):
        return
    
    vmax = 0.5
    vmin = -vmax

    cmap = mycolormap(th=-1, vmax=vmax)
    vertex_data = cortex.Vertex(vs, "fsaverage", cmap=cmap, vmin=vmin, vmax=vmax)
    cortex.quickflat.make_png(
        png_path,
        vertex_data,
        with_curvature=False,
        with_rois=False,
        with_labels=True,
        with_sulci=True,
        with_colorbar=False,
    )
    plt.close()
    

for beta in ['b2', 'b3']:

    runs = find_runs_from_exp_dir(os.path.join(args.exp_dir, beta))
    df = make_df(runs)
    
    full = (0, True, False)
    t0 = (0, False, False)
    subject = 'subj01'
    
    full_run = df[(df.subject == subject) & (df.t == full[0]) & (df.all_t == full[1]) & (df.rand == full[2])].run.tolist()[0]
    full_vs = read_test_voxel_score(full_run)[subject][f"TEST/PearsonCorrCoef/{subject}/all"][:N_FSAVERAGE]
    t0_run = df[(df.subject == subject) & (df.t == t0[0]) & (df.all_t == t0[1]) & (df.rand == t0[2])].run.tolist()[0]
    t0_vs = read_test_voxel_score(t0_run)[subject][f"TEST/PearsonCorrCoef/{subject}/all"][:N_FSAVERAGE]

    os.makedirs('/tmp/xdae', exist_ok=True)
    plot_one_run(full_vs, f'/tmp/xdae/{beta}_a.png')
    plot_one_run(t0_vs, f'/tmp/xdae/{beta}_b.png')
    plot_one_run(t0_vs - full_vs, f'/tmp/xdae/{beta}_c.png')
    

    import matplotlib.gridspec as gridspec

    fig = plt.figure(tight_layout=True, figsize=(16, 10))
    gs = gridspec.GridSpec(3, 2)

    ax = fig.add_subplot(gs[0, :1])
    ax.imshow(Image.open(f'/tmp/xdae/{beta}_a.png'))
    ax.axis("off")
    ax.set_title("A: T=-32:0", fontsize=24)
    
    ax = fig.add_subplot(gs[0, 1:])
    ax.imshow(Image.open(f'/tmp/xdae/{beta}_b.png'))
    ax.axis("off")
    ax.set_title("B: T=0", fontsize=24)
    
    ax = fig.add_subplot(gs[1:, :])
    ax.imshow(Image.open(f'/tmp/xdae/{beta}_c.png'))
    ax.axis("off")
    ax.set_title("B - A", fontsize=24)
    
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{beta}.png')
    plt.savefig(save_path, dpi=144)
    plt.close()