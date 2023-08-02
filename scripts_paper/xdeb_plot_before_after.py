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

# monkey patch
from cortex.quickflat.utils import _get_fig_and_ax
def my_add_colorbar(fig, cimg, colorbar_ticks=None, colorbar_location=(0.4, 0.07, 0.2, 0.04), 
                 orientation='horizontal'):
    """Add a colorbar to a flatmap plot

    Parameters
    ----------
    fig : matplotlib Figure object
        Figure into which to insert colormap
    cimg : matplotlib.image.AxesImage object
        Image for which to create colorbar. For reference, matplotlib.image.AxesImage 
        is the output of imshow()
    colorbar_ticks : array-like
        values for colorbar ticks
    colorbar_location : array-like
        Four-long list, tuple, or array that specifies location for colorbar axes 
        [left, top, width, height] (?)
    orientation : string
        'vertical' or 'horizontal'
    """
    from matplotlib import rc, rcParams
    rc('font', weight='bold')
    
    colorbar_location=(0.45, 0.07, 0.1, 0.04)
     
    fig, _ = _get_fig_and_ax(fig)
    cbar_ax = fig.add_axes(colorbar_location)
    cbar = fig.colorbar(cimg, cax=cbar_ax, orientation=orientation)
    
    cbar_ax.set_xticks([-1, 0, 1])
    
    cbar_ax.tick_params(axis='both', colors='grey', labelsize=28)
    cbar.outline.set_edgecolor('grey')
    return cbar_ax
cortex.quickflat.composite.add_colorbar = my_add_colorbar

from PIL import Image

import copy

import cortex
from matplotlib import pyplot as plt, ticker

plt.style.use("dark_background")
from config import AutoConfig

N_FSAVERAGE = 327684


def get_parser():
    parser = argparse.ArgumentParser(description="Ray Tune")
    parser.add_argument(
        "--exp_dir", type=str, default="/nfscc/alg23/xdea/b2/", help="exp dir"
    )
    parser.add_argument("--nsd_dir", type=str, default="/nfscc/natural-scenes-dataset")
    parser.add_argument(
        "--beta", type=str, default="b2", help="beta"
    )
    
    parser.add_argument(
        "--save_dir", type=str, default="/nfscc/fig/alg23/xdea/b3", help="save dir"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="overwrite"
    )
    return parser


args = get_parser().parse_args()

beta = args.beta
exp_dir = args.exp_dir.replace('b3', beta)

runs = find_runs_from_exp_dir(exp_dir)
print(len(runs))


datas = []
for run in runs:
    cfg: AutoConfig = read_config(run)
    subject = cfg.DATASET.SUBJECT_LIST[0]
    after = cfg.EXPERIMENTAL.USE_PREV_FRAME
    datas.append([subject, after, run])
df = pd.DataFrame(datas, columns=["subject", "after", "run"]).sort_values(
    ["subject", "after"]
)


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


def plot_v(v, png_path):
    if os.path.exists(png_path) and not args.overwrite:
        return
    
    vmax = 1.0
    vmin = -vmax

    cmap = mycolormap(th=-1, vmax=vmax)
    vertex_data = cortex.Vertex(v, "fsaverage", cmap=cmap, vmin=vmin, vmax=vmax)
    cortex.quickflat.make_png(
        png_path,
        vertex_data,
        with_curvature=False,
        with_rois=False,
        with_labels=False,
        with_sulci=True,
        with_colorbar=True,
    )
    plt.close()

def plot_one_run(run_dir, subject, png_path):
        
    vs = read_test_voxel_score(run_dir)
    vs = vs[subject][f"TEST/PearsonCorrCoef/{subject}/all"]
    vs = vs[:N_FSAVERAGE]
    ret_v = copy.deepcopy(vs)
    # vs[vs<0] = 0

    plot_v(vs, png_path)

    return ret_v


def plot_nc(beta_version, subject_name, nsd_dir, png_path):
    import nibabel as nib
    # load snr
    # /nfscc/natural-scenes-dataset/nsddata_betas/ppdata/subj01/fsaverage/betas_fithrf_GLMdenoise_RR/lh.ncsnr.mgh
    path = os.path.join(
        nsd_dir,
        "nsddata_betas",
        "ppdata",
        subject_name,
        "fsaverage",
        beta_version,
        "lh.ncsnr.mgh",
    )
    lh_snr = nib.load(path).get_fdata().flatten()
    path = os.path.join(
        nsd_dir,
        "nsddata_betas",
        "ppdata",
        subject_name,
        "fsaverage",
        beta_version,
        "rh.ncsnr.mgh",
    )
    rh_snr = nib.load(path).get_fdata().flatten()
    snr = np.concatenate([lh_snr, rh_snr], axis=0)
    nc = snr**2 / (snr**2 + 1 / 3)
    
    plot_v(nc, png_path)
    
subject = 'subj01'

save_dir = args.save_dir.replace('b3', beta)
os.makedirs(save_dir, exist_ok=True)

before_png_path = os.path.join(save_dir, f"{beta}_before.png")
v1 = plot_one_run(df[df["after"] == False]["run"].iloc[0], subject, before_png_path)
after_png_path = os.path.join(save_dir, f"{beta}_after.png")
v2 = plot_one_run(df[df["after"] == True]["run"].iloc[0], subject, after_png_path)

diff_png_path = os.path.join(save_dir, f"{beta}_diff.png")
v_diff = plot_v(v1 - v2, diff_png_path)

if args.beta == "b3":
    beta_version = "betas_fithrf_GLMdenoise_RR"
elif args.beta == "b2":
    beta_version = "betas_fithrf"
else:
    raise ValueError(f"beta version {args.beta} is not supported")

nc_png_path = os.path.join(save_dir, f"{beta}_nc.png")
plot_nc(beta_version, subject, args.nsd_dir, nc_png_path)

from PIL import Image, ImageSequence

# Load the two PNG images
image1 = Image.open(after_png_path)
image2 = Image.open(before_png_path)

# Create an empty list to store the frames
frames = []

# Append the frames with the two images
frames.append(image1)
frames.append(image2)

# Set the duration for each frame (1 second)
frame_duration = 1000  # in milliseconds

# Create a GIF image
gif_image = Image.new("RGB", image1.size)

# Loop through the frames and paste them into the GIF image
gif_path = os.path.join(save_dir, f"{beta}.gif")
for frame in frames:
    gif_image.paste(frame)
    gif_image.save(gif_path, save_all=True, append_images=[frame], duration=frame_duration, loop=0)