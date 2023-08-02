import copy

import pandas as pd
import torch
from cluster_utils import my_nfs_cluster_job, trial_dirname_creator

import argparse
import os
import sys
from random import seed, shuffle

import numpy as np
import ray
from ray import tune
from config import AutoConfig

from config_utils import dict_to_list, get_cfg_defaults, load_from_yaml, save_to_yaml

from read_utils import (
    find_runs_from_exp_dir,
    read_config,
    read_short_config,
    read_score_df,
    list_runs_from_exp_names,
)
from dark_onemodel import build_dmt, get_outs


def get_parser():
    parser = argparse.ArgumentParser(description="Ray Tune")
    parser.add_argument("--exp_dir", type=str, default="/nfscc/alg23/xvfe/", help="exp dir")
    parser.add_argument("--save_dir", type=str, default="/nfscc/alg23/xvfeb/", help="save dir")
    return parser


args = get_parser().parse_args()

save_dir = args.save_dir
os.makedirs(save_dir, exist_ok=True)

runs = find_runs_from_exp_dir(args.exp_dir)

print(len(runs))


def do_soup(ckpts):
    soup_state_dict = None
    n_ingredients = 0
    for ckpt in ckpts:
        print(f"loading {ckpt}")
        state_dict = torch.load(ckpt, map_location='cpu')
        if soup_state_dict is None:
            soup_state_dict = copy.deepcopy(state_dict)
        else:
            soup_state_dict = {k: v + soup_state_dict[k] for k, v in state_dict.items()}
        n_ingredients += 1
    soup_state_dict = {k: v / n_ingredients for k, v in soup_state_dict.items()}

    return soup_state_dict

answer = np.concatenate([np.arange(0, 8), np.arange(21, 35)]).tolist()
memory = np.arange(8, 19).tolist()
time = np.arange(19, 21).tolist()

full = np.arange(0, 35).tolist()

no_answer = [i for i in full if i not in answer]
no_memory = [i for i in full if i not in memory]
no_time = [i for i in full if i not in time]
    
    
datas = []
for run in runs:
    done_file = os.path.join(run, "done")
    if not os.path.exists(done_file):
        continue
    
    cfg = read_config(run)
    if cfg.EXPERIMENTAL.USE_PREV_FRAME == False and cfg.EXPERIMENTAL.BEHV_SELECTION == [-1]:
        row = 8
    elif cfg.EXPERIMENTAL.USE_PREV_FRAME == True and cfg.EXPERIMENTAL.BEHV_SELECTION == no_memory:
        row = 9
    elif cfg.EXPERIMENTAL.USE_PREV_FRAME == False and cfg.EXPERIMENTAL.BEHV_SELECTION == no_memory:
        row = 5
    elif cfg.EXPERIMENTAL.BEHV_SELECTION == no_time:
        row = 6
    elif cfg.EXPERIMENTAL.BEHV_SELECTION == no_answer:
        row = 7
    elif cfg.MODEL.COND.IN_DIM == 13:
        row = 3
    elif cfg.MODEL.COND.IN_DIM == 6:
        row = 4
    elif cfg.EXPERIMENTAL.USE_DEV_MODEL == False:
        row = 1
    else:
        row = 2
    
    soup_file = os.path.join(run, "soup.pth")
    lr = cfg.OPTIMIZER.LR
    if lr != 0.0003:
        continue
    
    datas.append([row, lr, soup_file, cfg])
df = pd.DataFrame(datas, columns=['row', 'lr', 'soup_file', 'cfg'])

# for row in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
for row in [8, 9]:
    _row_df = df[df.row == row]
    if len(_row_df) != 1:
        continue
    print("soup on")
    print(_row_df[['row', 'lr', 'soup_file']])
    
    soup_files = _row_df.soup_file.tolist()
    
    _save_dir = os.path.join(save_dir, f"row_{row}")
    os.makedirs(_save_dir, exist_ok=True)
    soup = do_soup(soup_files)
    torch.save(soup, os.path.join(_save_dir, f"soup.pth"))
    cfg: AutoConfig = _row_df.iloc[0].cfg
    save_to_yaml(cfg, os.path.join(_save_dir, f"config.yaml"))