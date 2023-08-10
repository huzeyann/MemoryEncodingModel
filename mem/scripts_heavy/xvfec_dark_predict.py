import copy
import os
from config import AutoConfig
from config_utils import load_from_yaml
from datamodule import NSDDatamodule
from plmodels import PlVEModel
import torch
from dark_onemodel import build_dmt, get_outs
import pytorch_lightning as pl

import argparse

from datasets import NSDDataset

parser = argparse.ArgumentParser()
parser.add_argument("--load_dir", type=str, default="/nfscc/alg23/xvfeb/")
args = parser.parse_args()

def _build_dmt(soup_dir):
    cfg: AutoConfig = load_from_yaml(os.path.join(soup_dir, "config.yaml"))
    cfg.EXPERIMENTAL.SHUFFLE_VAL = False
    dm = NSDDatamodule(cfg)
    dm.setup()

    plmodel = PlVEModel(cfg, dm.roi_dict, dm.neuron_coords_dict)

    trainer = pl.Trainer(
        accelerator="cuda",
        devices=[0],
        precision=16,
        enable_progress_bar=False,
    )

    return dm, plmodel, trainer

run_dirs = os.listdir(args.load_dir)
run_dirs = sorted(run_dirs)
row_names = copy.deepcopy(run_dirs)
run_dirs = [os.path.join(args.load_dir, run_dir) for run_dir in run_dirs]

for run_dir, row in zip(run_dirs, row_names):
    dark_name = f"xvfec_{row}"
    dm, plmodel, trainer = _build_dmt(run_dir)
    soup = torch.load(os.path.join(run_dir, "soup.pth"), map_location=torch.device('cpu'))
    plmodel.load_state_dict(soup)
    plmodel.eval()

    subject_list = dm.cfg.DATASET.SUBJECT_LIST

    for subject in subject_list:
        dataloader = dm.predict_dataloader(subject=subject)
        outs = get_outs(plmodel, trainer, dataloader)

        dataset : NSDDataset = dataloader.dataset
        dataset.save_dark(outs, dark_name)
        
        dataloader = dm.test_dataloader(subject=subject)
        outs = get_outs(plmodel, trainer, dataloader)
        
        dataset : NSDDataset = dataloader.dataset
        dataset.save_dark(outs, dark_name)