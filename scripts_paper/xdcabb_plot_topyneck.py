# %%
import copy
import os
from config import AutoConfig
from config_utils import load_from_yaml
from datamodule import NSDDatamodule
from plmodels import PlVEModel
import torch
from dark_onemodel import build_dmt, get_outs
import pytorch_lightning as pl
import numpy as np
import argparse

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
    
    cbar_ax.set_xticks([0, 1])
    
    cbar_ax.tick_params(axis='both', colors='grey', labelsize=28)
    cbar.outline.set_edgecolor('grey')
    return cbar_ax
cortex.quickflat.composite.add_colorbar = my_add_colorbar


from matplotlib import pyplot as plt
from matplotlib import cm, ticker

# plt.style.use("dark_background")
plt.style.use("default")

N_FSAVERAGE = 327684

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_path", type=str, default="/nfscc/alg23/xdcab/dev/topyneck.pth"
)
parser.add_argument("--overwrite", action="store_true", help="overwrite")
args = parser.parse_args()

cfg = load_from_yaml("/workspace/configs/dev.yaml")
cfg.DATASET.SUBJECT_LIST = [
    "subj01",
    "subj02",
    "subj03",
    "subj04",
    "subj05",
    "subj06",
    "subj07",
    "subj08",
]
cfg.DATASET.ROIS = ["all"]
cfg.DATASET.FMRI_SPACE = "fship"

dm = NSDDatamodule(cfg)
dm.setup()
plmodel = PlVEModel(cfg, dm.roi_dict, dm.neuron_coords_dict)
sd = torch.load(args.model_path, map_location=torch.device("cpu"))
plmodel.load_state_dict(sd, strict=False)
plmodel.eval()

subject_list = dm.cfg.DATASET.SUBJECT_LIST

mu_dict, w_dict = plmodel.get_retinamapper_layerselector_output()


def arr_creat(upperleft, upperright, lowerleft, lowerright):
    arr = np.linspace(
        np.linspace(lowerleft, lowerright, arrwidth),
        np.linspace(upperleft, upperright, arrwidth),
        arrheight,
        dtype=int,
    )
    return arr[:, :, None]


arrwidth = 256
arrheight = 256

r = arr_creat(0, 255, 0, 255)
g = arr_creat(0, 0, 255, 0)
b = arr_creat(255, 255, 0, 0)

color_gradient_img = np.concatenate([r, g, b], axis=2)
import scipy.ndimage

# img = scipy.ndimage.rotate(img, 90, reshape=False)
color_bar_img = color_gradient_img


def plot_retinamapper(subject, color_gradient_img):
    color_gradient_img = (
        torch.tensor(color_gradient_img).permute(2, 0, 1).float().unsqueeze(0)
    )
    from einops import rearrange, repeat

    mu = mu_dict[subject][:N_FSAVERAGE]
    mu = repeat(mu, "n c -> b n d c", b=1, d=1)
    # %%
    from torch.nn.functional import interpolate, grid_sample

    c = grid_sample(color_gradient_img, mu, align_corners=True)
    c = c.squeeze(0).squeeze(-1).t()
    c = c.numpy().astype(int)
    # %%
    vertex = cortex.VertexRGB(
        c[:, 0],
        c[:, 1],
        c[:, 2],
        "fsaverage",
    )

    png_path = f"/nfscc/fig/alg23/xdcabb/{subject}.png"
    if os.path.exists(png_path) and not args.overwrite:
        return
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    cortex.quickflat.make_png(
        png_path,
        vertex,
        with_curvature=True,
        with_rois=False,
        with_labels=False,
        with_sulci=True,
        with_colorbar=False,
    )
    plt.close()


for subject in subject_list:
    png_path = f"/nfscc/fig/alg23/xdcabb/{subject}.png"
    if os.path.exists(png_path) and not args.overwrite:
        continue
    plot_retinamapper(subject, color_gradient_img)


def plot_retina_grid(ax):
    ax.imshow(color_bar_img)
    ax.grid(axis="both", linestyle="--", alpha=0.5)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(16))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(16))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    for tick in ax.xaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)
    for tick in ax.yaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)


def plot_one_subject(subject):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), width_ratios=[3, 1])
    plot_retina_grid(axs[1])
    png_path = f"/nfscc/fig/alg23/xdcabb/{subject}.png"
    ax = axs[0]
    ax.imshow(plt.imread(png_path))
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"/nfscc/fig/alg23/xdcabb/{subject}_retinamapper.pdf", bbox_inches="tight")
    plt.close()


plot_one_subject("subj01")


def plot_layerselector(subject):
    w = w_dict[subject][:N_FSAVERAGE]
    w = w.numpy()
    for i in range(4):
        vertex = cortex.Vertex(w[:, i], "fsaverage", vmin=0, vmax=1, cmap="viridis")

        png_path = f"/nfscc/fig/alg23/xdcabb/{subject}_ls_{i}.png"
        if os.path.exists(png_path) and not args.overwrite:
            return
        os.makedirs(os.path.dirname(png_path), exist_ok=True)
        cortex.quickflat.make_png(
            png_path,
            vertex,
            with_curvature=True,
            with_rois=False,
            with_labels=False,
            with_sulci=True,
            with_colorbar=True,
        )
        plt.close()


plot_layerselector("subj01")


def plot_2x2(subject):
    fig, axs = plt.subplots(2, 2, figsize=(14, 7))
    for i in range(4):
        png_path = f"/nfscc/fig/alg23/xdcabb/{subject}_ls_{i}.png"
        ax = axs[i // 2, i % 2]
        ax.imshow(plt.imread(png_path))
        ax.axis("off")
        l = i * 3 + 2
        ax.set_title(f"Layer {l} (x{i+1})", fontsize=20)
    plt.tight_layout()
    plt.savefig(f"/nfscc/fig/alg23/xdcabb/{subject}_ls.pdf", bbox_inches="tight")
    plt.close()


plot_2x2("subj01")
