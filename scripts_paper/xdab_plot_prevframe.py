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
from matplotlib import pyplot as plt

plt.style.use("dark_background")
from config import AutoConfig

N_FSAVERAGE = 327684


def get_parser():
    parser = argparse.ArgumentParser(description="Ray Tune")
    parser.add_argument(
        "--exp_dir", type=str, default="/nfscc/alg23/xdaa/b3/", help="exp dir"
    )
    parser.add_argument(
        "--beta", type=str, default="b3", help="beta"
    )
    
    parser.add_argument(
        "--save_dir", type=str, default="/nfscc/fig/alg23/xdab/", help="save dir"
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
    t = cfg.EXPERIMENTAL.T_IMAGE
    all_t = cfg.EXPERIMENTAL.USE_PREV_FRAME
    rand = cfg.EXPERIMENTAL.SHUFFLE_IMAGES
    datas.append([subject, t, all_t, rand, run])
df = pd.DataFrame(datas, columns=["subject", "t", "all_t", "rand", "run"]).sort_values(
    ["subject", "t", "all_t", "rand"]
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


def plot_one_run(run_dir, subject, png_path):
    if os.path.exists(png_path) and not args.overwrite:
        return
    
    vs = read_test_voxel_score(run_dir)
    vs = vs[subject][f"TEST/PearsonCorrCoef/{subject}/all"]
    vs = vs[:N_FSAVERAGE]
    # vs[vs<0] = 0

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

# row = df.iloc[25]
# plot_one_run(row["run"], row["subject"], '/nfscc/fig/tmp.png')

# add png_path column to df
os.makedirs(f'/tmp/xdab/{beta}', exist_ok=True)
png_paths = []
for i, row in df.iterrows():
    png_name = f"{row['subject']}_{row['t']}_{row['all_t']}_{row['rand']}.png"
    png_path = os.path.join(f'/tmp/xdab/{beta}', png_name)
    png_paths.append(png_path)
df["png_path"] = png_paths

def job(row):
    row = row[1]
    plot_one_run(row["run"], row["subject"], row["png_path"])
    
import multiprocessing as mp

with mp.Pool(1) as p:
    p.map(job, df.iterrows())
    
    
SUBJECT_NAMES = ['subj01', 'subj02', 'subj03', 'subj04', 'subj05', 'subj06', 'subj07', 'subj08']

os.makedirs(f'/tmp/xdab/{beta}2x4/', exist_ok=True)
def make_2x4_plot(t, all_t, rand, part=0):
    save_path = f'/tmp/xdab/{beta}2x4/{t}_{all_t}_{rand}_{part}.png'
    if os.path.exists(save_path) and not args.overwrite:
        return
    
    fig, axs = plt.subplots(2, 4, figsize=(16, 10))
    for i, subject in enumerate(SUBJECT_NAMES):
        png_path = os.path.join(f'/tmp/xdab/{beta}', f"{subject}_{t}_{all_t}_{rand}.png")
        img = Image.open(png_path)
        if part == 0:
            # left crop
            img = img.crop((0, 0, img.width // 2, img.height))
        elif part == 1:
            # right crop
            img = img.crop((img.width // 2, 0, img.width, img.height))
        axs[i // 4, i % 4].imshow(img)
        axs[i // 4, i % 4].axis('off')
        axs[i // 4, i % 4].set_title(subject)
    if rand == True:
        t = 'random'
    elif all_t == True:
        t = '-32:0'
    else:
        t = t
    plt.suptitle(f'T = {t}', fontsize=30)
    plt.tight_layout()
    plt.savefig(save_path, dpi=144)
    plt.close()

def job2(tup, part=0):
    t, all_t, rand = tup
    make_2x4_plot(t, all_t, rand, part=part)

order = [
    (0, True, False),    
]
order += [
    (t, False, False) for t in range(0, -32, -1)
]
order += [
    (0, True, True)
]

from functools import partial
with mp.Pool(16) as p:
    p.map(partial(job2, part=0), order)
    p.map(partial(job2, part=1), order)
    
def make_video(png_paths, video_path, fps=4):
    import cv2
    img_array = []
    for filename in png_paths:
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)
    
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

left_png_paths = [f'/tmp/xdab/{beta}2x4/{t}_{all_t}_{rand}_0.png' for t, all_t, rand in order]
right_png_paths = [f'/tmp/xdab/{beta}2x4/{t}_{all_t}_{rand}_1.png' for t, all_t, rand in order]
# # repeat left frame 3 times
# left_png_paths = [left_png_paths[i // 3] for i in range(len(left_png_paths) * 3)]

all_frames = left_png_paths + right_png_paths

save_dir = args.save_dir
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f'{beta}_previous_frames_a.mp4')
make_video(all_frames, save_path, fps=3)



os.makedirs(f'/tmp/xdab/{beta}1x1/', exist_ok=True)
def make_1x1_plot(subject, t, all_t, rand):
    png_path = os.path.join(f'/tmp/xdab/{beta}', f"{subject}_{t}_{all_t}_{rand}.png")
    save_path = f'/tmp/xdab/{beta}1x1/{subject}_{t}_{all_t}_{rand}.png'
    if os.path.exists(save_path) and not args.overwrite:
        return
    
    fig, axs = plt.subplots(1, 1, figsize=(16, 10))
    img = Image.open(png_path)
    axs.imshow(img)
    axs.axis('off')
    if rand == True:
        t = 'random'
    elif all_t == True:
        t = '-32:0'
    else:
        t = t
    plt.title(f'T = {t}', fontsize=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=72)
    plt.close()

def job3(tup):
    subject, t, all_t, rand = tup
    make_1x1_plot(subject, t, all_t, rand)

with mp.Pool(16) as p:
    tup_list = []
    for subject in SUBJECT_NAMES:
        tup_list += [(subject, t, all_t, rand) for t, all_t, rand in order]
    p.map(job3, tup_list)


all_pngs = []
for subject in SUBJECT_NAMES:
    for t, all_t, rand in order:
        png_path = os.path.join(f'/tmp/xdab/{beta}1x1/', f"{subject}_{t}_{all_t}_{rand}.png")
        all_pngs.append(png_path)

save_dir = args.save_dir
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f'{beta}_previous_frames_b.mp4')
if os.path.exists(save_path) and not args.overwrite:
    pass
else:
    make_video(all_pngs, save_path, fps=3)


# copy the first 2 images to save_dir
subject = SUBJECT_NAMES[0]
for t, all_t, rand in order[:2]:
    png_path = os.path.join(f'/tmp/xdab/{beta}/', f"{subject}_{t}_{all_t}_{rand}.png")
    save_path = os.path.join(save_dir, f'{beta}_pf_{t}_{all_t}_{rand}.png')
    os.system(f'cp {png_path} {save_path}')
    
    
    
order = [
    (t, False, False) for t in range(0, -32, -1)
]

# make 4x8 plot
plt.style.use("default")

fig, axes = plt.subplots(8, 4, figsize=(12, 12))
subject = SUBJECT_NAMES[0]

for ax, (t, all_t, rand) in zip(axes.flatten(), order):
    png_path = os.path.join(f'/tmp/xdab/{beta}/', f"{subject}_{t}_{all_t}_{rand}.png")
    img = Image.open(png_path)
    ax.imshow(img)
    ax.axis('off')
    if rand == True:
        t = 'random'
    elif all_t == True:
        t = '-32:0'
    else:
        t = t
    # ax.set_title(f'T = {t}', fontsize=20)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, f'{beta}_4x8_light.pdf'), bbox_inches="tight")
plt.close()


plt.style.use("dark_background")

# make b3 b2 video
os.makedirs(f'/tmp/xdab/1x2/', exist_ok=True)
def make_1x2_plot(subject, t, all_t, rand):
    b2_path = os.path.join(f'/tmp/xdab/b2', f"{subject}_{t}_{all_t}_{rand}.png")    
    b3_path = os.path.join(f'/tmp/xdab/b3', f"{subject}_{t}_{all_t}_{rand}.png")
    save_path = f'/tmp/xdab/1x2/{subject}_{t}_{all_t}_{rand}.png'
    if os.path.exists(save_path) and not args.overwrite:
        return
    
    fig, axs = plt.subplots(1, 2, figsize=(30, 10))
    
    for ax, path in zip(axs, [b2_path, b3_path]):
        img = Image.open(path)
        ax.imshow(img)
        ax.axis('off')
    
    # axs[0].set_title('beta2', fontsize=20)
    # axs[1].set_title('beta3', fontsize=20)
    
    if rand == True:
        t = 'random'
    elif all_t == True:
        t = '-32:0'
    else:
        t = int(t)
        t = f'{t:02d}'
    
    _isubj = int(subject[-2:])
    plt.tight_layout()
    fig.text(0.5, 0.3, f'T = {t}', ha='center', fontsize=20, fontweight='bold')
    fig.text(0.5, 0.75, f'subject#{_isubj}', ha='center', fontsize=20, fontweight='bold')
    fig.text(0.4, 0.2, 'beta2', ha='center', fontsize=20)
    fig.text(0.6, 0.2, 'beta3', ha='center', fontsize=20)

    plt.savefig(save_path, dpi=144, bbox_inches="tight")
    plt.close()

def job4(tup):
    subject, t, all_t, rand = tup
    make_1x2_plot(subject, t, all_t, rand)
    
with mp.Pool(16) as p:
    tup_list = []
    for subject in SUBJECT_NAMES:
        tup_list += [(subject, t, all_t, rand) for t, all_t, rand in order]
    p.map(job4, tup_list)

all_pngs = []
for subject in SUBJECT_NAMES:
    for t, all_t, rand in order:
        png_path = os.path.join(f'/tmp/xdab/1x2/', f"{subject}_{t}_{all_t}_{rand}.png")
        all_pngs.append(png_path)

save_dir = args.save_dir
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f'previous_frames_c.mp4')
if os.path.exists(save_path) and not args.overwrite:
    pass
else:
    make_video(all_pngs, save_path, fps=3)