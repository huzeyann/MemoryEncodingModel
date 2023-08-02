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

from config_utils import dict_to_list, get_cfg_defaults, load_from_yaml

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
    parser.add_argument(
        "--dark_name", type=str, default="xvgb", help="dark name"
    )
    parser.add_argument(
        "--save_dir", type=str, default="/nfscc/alg23/xvgb/predict", help="save dir"
    )
    parser.add_argument(
        "--exp_dir", type=str, default="/nfscc/alg23/xvga/", help="exp dir"
    )
    parser.add_argument("--stage", type=str, default="predict", help="stage")
    return parser


args = get_parser().parse_args()


runs = find_runs_from_exp_dir(args.exp_dir)

roi_sch_dict = {
    "all": ["all"],
    "A": ["RSC", "E", "MV", "ML", "MP", "V", "L", "P", "R"],
    "W": [f"w_{i}" for i in range(1, 10)],
}
for _ir in range(1, 11):
    roi_sch_dict[f"R{_ir}"] = [f"r_{_ir}_{i}" for i in range(1, 10)]


def get_sch_name_by_roi(roi_name):
    for sch_name, rois in roi_sch_dict.items():
        if roi_name in rois:
            return sch_name
    raise ValueError(f"roi_name {roi_name} not found in roi_sch_dict")


roi_counter = {}

# all-0-0, ATLAS-REPT-CKPT
datas = []
for run in runs:
    cfg: AutoConfig = read_config(run)
    roi = cfg.DATASET.ROIS[0]
    if roi not in roi_counter:
        roi_counter[roi] = 0
    else:
        roi_counter[roi] += 1
    c = roi_counter[roi]
    sch = get_sch_name_by_roi(roi)
    # if sch == 'A' or sch == 'W':
    #     continue
    sch += f"-{c}"

    run_name = os.path.basename(run)

    # run_ckpt_dir = os.path.join(args.ckpt_dir, run_name)
    # run_ckpts = os.listdir(run_ckpt_dir)
    # run_ckpts = [f for f in run_ckpts if f.endswith(".pth")]  # soup.pth
    # run_ckpts = [os.path.join(run_ckpt_dir, ckpt) for ckpt in run_ckpts]
    # run_ckpts = sorted(run_ckpts)
    run_ckpts = [os.path.join(run, "soup.pth")]

    for _i_ckpt, ckpt in enumerate(run_ckpts):
        sch_ckpt = f"{sch}-{_i_ckpt}"
        datas.append(
            {
                "sch": sch_ckpt,
                "roi": roi,
                "run": run,
                "ckpt": ckpt,
            }
        )

df = pd.DataFrame(datas)
df = df.sort_values(by=["sch"])

schs = df["sch"].unique()

# schs = schs[5::10]

def load_one_ckpt(run_dir, ckpt_path, subject):
    torch.set_float32_matmul_precision("medium")
    dm, plmodel, trainer = build_dmt(run_dir)
    ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
    plmodel.load_state_dict(ckpt)
    plmodel.eval()

    if args.stage == "train":
        dataloader = dm.train_dataloader(subject=subject, shuffle=False)
    elif args.stage == "val":
        dataloader = dm.val_dataloader(subject=subject, shuffle=False)
    elif args.stage == "test":
        dataloader = dm.test_dataloader(subject=subject, shuffle=False)
    elif args.stage == "predict":
        dataloader = dm.predict_dataloader(subject=subject, shuffle=False)
    else:
        raise ValueError(f"stage {args.stage} not supported")
    outs = get_outs(plmodel, trainer, dataloader)
    voxel_indices = dataloader.dataset.voxel_indices
    if voxel_indices == ...:
        voxel_indices = torch.arange(outs.shape[1])
    return outs, voxel_indices, dataloader.dataset


def load_all_rois(sch, subject):
    _sch_df = df[df["sch"] == sch]
    vi_dict = {}
    outs_dict = {}
    for row in _sch_df.itertuples():
        roi = row.roi
        run = row.run
        ckpt = row.ckpt
        print(f"loading {sch} {run} {ckpt}")
        outs, voxel_indices, dataset = load_one_ckpt(run, ckpt, subject)
        vi_dict[roi] = voxel_indices
        outs_dict[roi] = outs.cpu().numpy().astype(np.float16)
    N_voxel = sum([len(vi) for vi in vi_dict.values()])
    N_data = outs.shape[0]
    outs_all = np.zeros((N_data, N_voxel), dtype=np.float16)
    for _i, (roi, vi) in enumerate(vi_dict.items()):
        _outs = outs_dict[roi]
        outs_all[:, vi] = _outs
    return outs_all, dataset

def job(subject):
    ensemble_outs = None
    for sch in schs:
        print(f"loading {sch}")
        outs, dataset = load_all_rois(sch, subject)
        outs = torch.from_numpy(outs).float().cuda()
        _i_sch = np.where(schs == sch)[0][0]
        if ensemble_outs is None:
            ensemble_outs = outs
        else:
            ensemble_outs += outs
    ensemble_outs /= len(schs)
    dataset.save_dark(ensemble_outs, args.dark_name)
    
# job("subj01")
    

@my_nfs_cluster_job
def run_tune(tune_dict, **kwargs):
    subject = tune_dict["subject"]
    job(subject)


tune_dict = {
    "subject": tune.grid_search([f"subj{i:02d}" for i in range(1, 9)]),
}
local_dir = os.path.join(args.save_dir, "ray")
ana = tune.run(
    run_tune,
    config=tune_dict,
    resources_per_trial={"cpu": 1, "gpu": 1},
    verbose=False,
    local_dir=local_dir,
    resume="AUTO+ERRORED",
    name="dark",
    trial_dirname_creator=trial_dirname_creator,
)
