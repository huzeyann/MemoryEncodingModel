import argparse
import os
import numpy as np
import torch

import pandas as pd

from metrics import vectorized_correlation
from datamodule import NSDDatamodule
from datasets import NSDDataset

from config_utils import get_cfg_defaults
from config import AutoConfig

SPACE = 'fsaverage'
N = 327684

def get_parser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dark_names", nargs="+", type=str, default=["xvdb"])
    return parser


args = get_parser().parse_args()

datas = []
# for _i_row in range(1, 8):
    # dark_name = f"xvfec_row_{_i_row}"
# for dark_name in ['xvdb']:
# for dark_name in ['xvfee_mem', 'xvfee_baseline']:
# for dark_name in ['xvfef_fmn', 'xvfe_nm1']:
# for dark_name in ['xvfe_nm2', 'xvfef_bbaseline']:
for dark_name in args.dark_names:
    rs = []
    for _i_subject in range(1, 9):
        subject = f"subj{_i_subject:02d}"

        cfg = get_cfg_defaults()
        cfg.DATASET.FMRI_SPACE = SPACE
        cfg.DATASET.ROIS = ["orig"]
        cfg.DATASET.DARK_POSTFIX = dark_name
        cfg.DATASET.SUBJECT_LIST = [subject]

        dm = NSDDatamodule(cfg)
        dm.setup()

        dataloader = dm.test_dataloader(subject=subject)
        dataset: NSDDataset = dataloader.dataset
        
        ys = []
        darks = []
        for batch in dataloader:
            (
                img,
                prev_img,
                prev_feats,
                y,
                dark,
                bhv,
                prev_bhvs,
                ssid,
                subject_name,
                data_idx,
            ) = batch
            
            ys += y
            darks += dark
        
        ys = torch.stack(ys)
        darks = torch.stack(darks)
        
        r = vectorized_correlation(ys.cuda(), darks.cuda())
        
        rs.append(r)
        
        print(f"{dark_name} {subject} {r.mean().item()}")
    
    rs = torch.concatenate(rs)
    
    mean_r = rs.mean()
    mean_r2 = (rs ** 2).mean()
    
    datas.append([dark_name, mean_r.item(), mean_r2.item()])
    
df = pd.DataFrame(datas, columns=['dark_name', 'mean_r', 'mean_r2'])

def print_csv(df):
    print(df.to_csv(index=False, float_format="%.3f"))

print_csv(df)