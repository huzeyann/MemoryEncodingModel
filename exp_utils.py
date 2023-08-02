# %%
from collections import OrderedDict
import glob
import json
import sys
import traceback
import re
import logging
from time import sleep
from einops import repeat
import numpy as np

import torch
import os
import pandas as pd
import ray
from ray import tune

import torch.nn.functional as F

import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
from PIL import Image, ImageDraw

import cortex
from matplotlib.pyplot import cm
from config import AutoConfig

from config_utils import flatten_dict, load_from_yaml

from IPython.display import display, HTML, clear_output

from datamodule import NSDDatamodule, build_dm

import glob

plt.style.use("dark_background")
# %%


def load_cfg(run):
    path = glob.glob(run + "/**/hparams.yaml", recursive=True)
    # print(path)
    path = path[0]
    cfg = load_from_yaml(path)
    return cfg

def load_voxel_metric(run, stage="TEST"):
    path = glob.glob(run + f"/**/stage={stage}*.npy", recursive=True)
    path = sorted(path)
    path = path[-1]
    # print(path)
    voxel_metric = np.load(path, allow_pickle=True).item()
    return voxel_metric

# %%
def list_runs_from_exp_names(exp_names, exp_dir="/nfscc/afo/ray_results", only_done=True):
    runs = []
    for exp_name in exp_names:
        i_dir = os.path.join(exp_dir, exp_name)
        runs += os.listdir(i_dir)
        runs = [r for r in runs if os.path.isdir(os.path.join(i_dir, r))]
        runs = [os.path.join(i_dir, r) for r in runs]
        if only_done == True:
            filterer = lambda x: os.path.exists(os.path.join(x, "done"))
            runs = list(filter(filterer, runs))
    runs = sorted(runs)
    return runs