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
        "--exp_dir", type=str, default="/nfscc/alg23/xdba/b3/", help="exp dir"
    )
    parser.add_argument(
        "--save_dir", type=str, default="/nfscc/fig/alg23/xdbb/", help="save dir"
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
PLOT_BHV_NAMES = ["all", "button_press", "reaction_time", "future_answer", "is_old", "run_id", "trial_id", "run_trial_id"]

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


def plot_one_run(run_dir, subject, png_path):
    if os.path.exists(png_path) and not args.overwrite:
        return

    vs = read_test_voxel_score(run_dir)
    vs = vs[subject][f"TEST/PearsonCorrCoef/{subject}/all"]
    vs = vs[:N_FSAVERAGE]
    # vs[vs<0] = 0

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


# row = df.iloc[25]
# plot_one_run(row["run"], row["subject"], '/nfscc/fig/tmp.png')

# add png_path column to df
os.makedirs("/tmp/xdbb/", exist_ok=True)
png_paths = []
for i, row in df.iterrows():
    png_name = f"{row['subject']}_{row['behv_name']}_{row['blank']}_{row['bhv']}.png"
    png_path = os.path.join("/tmp/xdbb/", png_name)
    png_paths.append(png_path)
df["png_path"] = png_paths


def job(row):
    row = row[1]
    plot_one_run(row["run"], row["subject"], row["png_path"])


import multiprocessing as mp

with mp.Pool(20) as p:
    p.map(job, df.iterrows())


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

# 2 row is left and right
# 3 col is bhv_only, bhv+image, image_only

os.makedirs("/tmp/xdbb/2x3/", exist_ok=True)


def make_2x3_plot(subject, bhv_name):
    # save_path = f"/tmp/xdbb/2x3/{t}_{all_t}_{rand}_{part}.png"
    save_path = f"/tmp/xdbb/2x3/{subject}_{bhv_name}.png"
    if os.path.exists(save_path) and not args.overwrite:
        return

    fig, axs = plt.subplots(2, 3, figsize=(16, 10))
    for ax in axs.flatten():
        ax.axis("off")
    
    bhv_only_png = os.path.join("/tmp/xdbb/", f"{subject}_{bhv_name}_True_True.png")
    bhvimage_png = os.path.join("/tmp/xdbb/", f"{subject}_{bhv_name}_False_True.png")
    img_only_png = os.path.join("/tmp/xdbb/", f"{subject}_all_False_False.png")
    
    def crop(img_path, part=0):
        img = Image.open(img_path)
        if part == 0:
            img = img.crop((0, 0, img.width // 2, img.height))
        else:
            img = img.crop((img.width // 2, 0, img.width, img.height))
        return img
    
    bhv_name = PLOT_BHV_NAMES[BHV_NAMES.index(bhv_name)]
    
    axs[0, 0].imshow(crop(bhv_only_png, part=0))
    axs[0, 0].set_title(f"AdaLN: ON [{bhv_name}]\n Image: OFF")
    axs[1, 0].imshow(crop(bhv_only_png, part=1))
    axs[0, 1].imshow(crop(bhvimage_png, part=0))
    axs[0, 1].set_title(f"AdaLN: ON [{bhv_name}]\n Image: ON")
    axs[1, 1].imshow(crop(bhvimage_png, part=1))
    axs[0, 2].imshow(crop(img_only_png, part=0))
    axs[0, 2].set_title(f"AdaLN: OFF\n Image: ON")
    axs[1, 2].imshow(crop(img_only_png, part=1))
    
    plt.suptitle(f"{subject}", fontsize=30)
    plt.tight_layout()
    plt.savefig(save_path, dpi=144)
    plt.close()
    

def job2(tup):
    make_2x3_plot(*tup)


from itertools import product

tup_list = list(product(SUBJECT_NAMES, BHV_NAMES))

from functools import partial

with mp.Pool(20) as p:
    p.map(job2, tup_list)


os.makedirs("/tmp/xdbb/single/", exist_ok=True)
def make_single_plot(subject, bhv_name):
    save_path = f"/tmp/xdbb/single/{subject}_{bhv_name}"
    if os.path.exists(save_path) and not args.overwrite:
        return

    bhv_only_png = os.path.join("/tmp/xdbb/", f"{subject}_{bhv_name}_True_True.png")
    bhvimage_png = os.path.join("/tmp/xdbb/", f"{subject}_{bhv_name}_False_True.png")
    img_only_png = os.path.join("/tmp/xdbb/", f"{subject}_all_False_False.png")

    bhv_name = PLOT_BHV_NAMES[BHV_NAMES.index(bhv_name)]
    
    for i, (name, png) in enumerate(zip([f"AdaLN: ON [{bhv_name}]\nImage: OFF", f"AdaLN: ON [{bhv_name}]\nImage: ON", "AdaLN: OFF\nImage: ON"], [bhv_only_png, bhvimage_png, img_only_png])):
        img = Image.open(png)

        fig = plt.figure(figsize=(16, 10))
        plt.axis("off")
        plt.imshow(img)
        plt.title(f"Subject: {subject}\n{name}", fontsize=30)
        plt.tight_layout()
        
        _save_path = f"{save_path}_{i}.png"
        plt.savefig(_save_path, dpi=144)
        plt.close()

def job3(tup):
    make_single_plot(*tup)

with mp.Pool(20) as p:
    p.map(job3, tup_list)

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


all_frames = []


for bhv_name in BHV_NAMES:
    for subject in SUBJECT_NAMES:
        png_path = f"/tmp/xdbb/single/{subject}_{bhv_name}_0.png"
        all_frames.append(png_path)
        png_path = f"/tmp/xdbb/single/{subject}_{bhv_name}_1.png"
        all_frames.append(png_path)
        png_path = f"/tmp/xdbb/single/{subject}_{bhv_name}_2.png"
        all_frames.append(png_path)

for bhv_name in BHV_NAMES:
    for subject in SUBJECT_NAMES:
        png_path = f"/tmp/xdbb/2x3/{subject}_{bhv_name}.png"
        all_frames.append(png_path)

save_dir = args.save_dir
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "behaviors.mp4")
make_video(all_frames, save_path, fps=3)

# copy the first subject01 images to save_dir
subject = SUBJECT_NAMES[0]
for bhv_name in BHV_NAMES:
    bhv_only_png = os.path.join("/tmp/xdbb/", f"{subject}_{bhv_name}_True_True.png")
    bhvimage_png = os.path.join("/tmp/xdbb/", f"{subject}_{bhv_name}_False_True.png")
    img_only_png = os.path.join("/tmp/xdbb/", f"{subject}_all_False_False.png")

    os.system(f"cp {bhv_only_png} {save_dir}")
    os.system(f"cp {bhvimage_png} {save_dir}")
    os.system(f"cp {img_only_png} {save_dir}")