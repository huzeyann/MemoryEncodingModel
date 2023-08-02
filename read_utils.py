# %%
import glob
import json
import sys
import traceback
import re
import logging
from time import sleep
import numpy as np

import torch
import os
import pandas as pd
import ray
from ray import tune

import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
from PIL import Image, ImageDraw

import cortex
from matplotlib.pyplot import cm

from config_utils import flatten_dict, load_from_yaml

from IPython.display import display, HTML, clear_output

plt.style.use("dark_background")

def set_display():
    pd.options.display.float_format = "{:,.4f}".format
    pd.options.display.max_colwidth = 1000
    pd.options.display.max_rows = 1000
    pd.options.display.max_columns = 1000


def pretty_print(df):
    df.style.set_properties(**{"white-space": "pre"})
    return display(HTML(df.to_html().replace("\\n", "<br>")))

def read_config(run):
    cfg_path = glob.glob(os.path.join(run, "**/hparams.yaml"), recursive=True)[0]
    return load_from_yaml(cfg_path)

def read_short_config(run):
    json_path = glob.glob(os.path.join(run, "**/params.json"), recursive=True)[0]
    cfg = json.load(open(json_path, "r"))
    return cfg

def read_score_df(run):
    try:
        csv_path = glob.glob(os.path.join(run, "**/metrics.csv"), recursive=True)[0]
        return pd.read_csv(csv_path)
    except:
        logging.warning(f"Could not find metrics.csv in {run}")
        return None
    
def read_test_voxel_score(run):
    # /nfscc/ray_results/hunt_behavior/full_bhv/t074f6_00000_DATASET.SUBJECT_LIST=subj01MODEL.BACKBONE.ADAPTIVE_LN.SCALE=0.5/lightning_logs/voxel_metric/stage=TEST.step=000000009028.pkl.npy
    vs_path = glob.glob(os.path.join(run, "**/voxel_metric/stage=TEST.step=*.pkl.npy"), recursive=True)[0]
    return np.load(vs_path, allow_pickle=True).item()

def read_val_voxel_score(run):
    vs_path = glob.glob(os.path.join(run, "**/voxel_metric/stage=VAL.step=*.pkl.npy"), recursive=True)
    vs_path = sorted(vs_path)[-1]
    return np.load(vs_path, allow_pickle=True).item()


def list_runs_from_exp_names(exp_names, exp_dir="/nfscc/ray_results/saved"):
    runs = []
    for exp_name in exp_names:
        i_dir = os.path.join(exp_dir, exp_name)
        runs += os.listdir(i_dir)
        runs = [r for r in runs if os.path.isdir(os.path.join(i_dir, r))]
        runs = [os.path.join(i_dir, r) for r in runs]
    runs = sorted(runs)
    return runs


def find_runs_from_exp_dir(exp_dir):
    exp_names = os.listdir(exp_dir)
    runs = list_runs_from_exp_names(exp_names, exp_dir)
    runs = sorted(runs)
    return runs