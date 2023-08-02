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


from matplotlib import pyplot as plt

plt.style.use("dark_background")
from config import AutoConfig

N_FSAVERAGE = 327684


def get_parser():
    parser = argparse.ArgumentParser(description="Ray Tune")
    parser.add_argument(
        "--exp_dir", type=str, default="/nfscc/alg23/xdba/b3/", help="exp dir"
    )
    parser.add_argument(
        "--save_dir", type=str, default="/nfscc/fig/alg23/xdbc/", help="save dir"
    )
    parser.add_argument("--overwrite", action="store_true", help="overwrite")
    return parser


args = get_parser().parse_args()

runs = find_runs_from_exp_dir(args.exp_dir)
print(len(runs))


"""
button = np.arange(2, 6)
rtt = np.arange(0, 2)
future = np.arange(28, 35)
is_old = np.arange(8, 19)
run = np.arange(19, 20)
trial = np.arange(20, 21)
run_trail = np.arange(19, 21)

"DATASET.SUBJECT_LIST": tune.grid_search([['subj01'], ['subj02'], ['subj03'], ['subj04'], ['subj05'], ['subj06'], ['subj07'], ['subj08']]),
"EXPERIMENTAL.BLANK_IMAGE": tune.grid_search([False]),
"EXPERIMENTAL.USE_BHV": tune.grid_search([True]),
"EXPERIMENTAL.USE_BHV_PASSTHROUGH": tune.grid_search([True]),
"EXPERIMENTAL.BEHV_ONLY": tune.grid_search([False]),
"EXPERIMENTAL.BEHV_SELECTION": tune.grid_search(collection),
"MODEL.COND.IN_DIM": tune.sample_from(lambda spec: int(len(spec.config['EXPERIMENTAL.BEHV_SELECTION']))),

"""
BHV_NAMES = ["all", "button", "rtt", "future", "is_old", "run", "trial", "run_trial"]
PLOT_BHV_NAMES = ["all", "button press", "reaction time", "future answer", "is old", "run id", "trial id", "relative time"]


datas = []
for run in runs:
    cfg: AutoConfig = read_config(run)
    subject = cfg.DATASET.SUBJECT_LIST[0]
    blank = cfg.EXPERIMENTAL.BLANK_IMAGE
    bhv = cfg.EXPERIMENTAL.USE_BHV
    behv_selection = cfg.EXPERIMENTAL.BEHV_SELECTION
    behv_name = ""
    if behv_selection == np.arange(2, 6).tolist():
        behv_name = "button"
    elif behv_selection == np.arange(0, 2).tolist():
        behv_name = "rtt"
    elif behv_selection == np.arange(28, 35).tolist():
        behv_name = "future"
    elif behv_selection == np.arange(8, 19).tolist():
        behv_name = "is_old"
    elif behv_selection == np.arange(19, 20).tolist():
        behv_name = "run"
    elif behv_selection == np.arange(20, 21).tolist():
        behv_name = "trial"
    elif behv_selection == np.arange(19, 21).tolist():
        behv_name = "run_trial"
    else:
        behv_name = 'all'

    datas.append([subject, behv_name, blank, bhv, run])
df = pd.DataFrame(
    datas, columns=["subject", "behv_name", "blank", "bhv", "run"]
).sort_values(by=["subject", "behv_name", "blank", "bhv"])


def mycolormap(th=0.05, vmax=0.3):
    import matplotlib.colors as mcolors

    # Define the colormap
    cmap = plt.cm.get_cmap("bwr")

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
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)

    # # Plot a colorbar to show the colormap
    # plt.imshow([np.linspace(vmin, vmax, 100)], cmap=custom_cmap, aspect='auto')
    # plt.colorbar()
    # plt.savefig('/nfscc/fig/tmp_c.png')
    # plt.close()

    return custom_cmap


def plot_one_v(vs, png_path):
    if os.path.exists(png_path) and not args.overwrite:
        return

    vmax = 1.0
    vmin = -vmax

    cmap = mycolormap(th=-1, vmax=vmax)
    vertex_data = cortex.Vertex(vs, "fsaverage", cmap=cmap, vmin=vmin, vmax=vmax)
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


def get_vs(run_dir, subject):
    vs = read_test_voxel_score(run_dir)
    vs = vs[subject][f"TEST/PearsonCorrCoef/{subject}/all"]
    vs = vs[:N_FSAVERAGE]
    return vs


SUBJECT_NAMES = [
    "subj01",
    "subj02",
    "subj03",
    "subj04",
    "subj05",
    "subj06",
    "subj07",
    "subj08",
]

os.makedirs("/tmp/xdbc/", exist_ok=True)

png_paths = []
for bhv_name in ["all", "button", "rtt", "future", "is_old", "run_trial"]:
    plot_bhv_name = PLOT_BHV_NAMES[BHV_NAMES.index(bhv_name)]
    for subject in SUBJECT_NAMES:
        A = (subject, bhv_name, True, True)
        A = df[(df.subject == A[0]) & (df.behv_name == A[1]) & (df.blank == A[2]) & (df.bhv == A[3])].run.tolist()[0]
        A = get_vs(A, subject)
        B = (subject, bhv_name, False, True)
        B = df[(df.subject == B[0]) & (df.behv_name == B[1]) & (df.blank == B[2]) & (df.bhv == B[3])].run.tolist()[0]
        B = get_vs(B, subject)
        C = (subject, 'all', False, False)
        C = df[(df.subject == C[0]) & (df.behv_name == C[1]) & (df.blank == C[2]) & (df.bhv == C[3])].run.tolist()[0]
        C = get_vs(C, subject)
        
        A_png = f"/tmp/xdbc/{subject}_{plot_bhv_name}_True_True.png"
        plot_one_v(A, A_png)
        B_png = f"/tmp/xdbc/{subject}_{plot_bhv_name}_False_True.png"
        plot_one_v(B, B_png)
        C_png = f"/tmp/xdbc/{subject}_all_False_False.png"
        plot_one_v(C, C_png)
        
        D = B - C - A
        D_png = f"/tmp/xdbc/{subject}_{plot_bhv_name}_imagerelated.png"
        plot_one_v(D, D_png)
        
        png_path = f"/tmp/xdbc/{subject}_{plot_bhv_name}_2x2.png"
        png_paths.append(png_path)
        if os.path.exists(png_path) and not args.overwrite:
            continue
        fig, axs = plt.subplots(2, 2, figsize=(16, 10))
        for ax in axs.flatten():
            ax.axis("off")
            
        axs[0, 0].imshow(Image.open(A_png))
        axs[0, 0].set_title("A: Image OFF, Cond ON", fontsize=24)
        axs[0, 1].imshow(Image.open(B_png))
        axs[0, 1].set_title("B: Image ON , Cond ON", fontsize=24)
        axs[1, 0].imshow(Image.open(C_png))
        axs[1, 0].set_title("C: Image ON , Cond OFF", fontsize=24)
        axs[1, 1].imshow(Image.open(D_png))
        axs[1, 1].set_title("D: B - C - A", fontsize=24)
        
        plt.suptitle(f"{subject}, Cond: '{plot_bhv_name}'", fontsize=24)
        plt.tight_layout()
        
        plt.savefig(png_path, dpi=144)
        plt.close()

def make_video(png_paths, video_path, fps=4):
    import cv2

    img_array = []
    for filename in png_paths:
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

save_dir = args.save_dir
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "behaviors.mp4")
make_video(png_paths, save_path, fps=3)

# copy the first subject01 images to save_dir
subject = SUBJECT_NAMES[0]
for bhv_name in PLOT_BHV_NAMES:
    plot_path = f"/tmp/xdbc/{subject}_{bhv_name}_2x2.png"
    
    os.system(f"cp '{plot_path}' {save_dir}")
    
    bhv_only_png = os.path.join("/tmp/xdbc/", f"{subject}_{bhv_name}_True_True.png")

    os.system(f"cp '{bhv_only_png}' {save_dir}")
