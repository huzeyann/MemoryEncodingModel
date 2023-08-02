# %%
import argparse
import os
from matplotlib import ticker
import numpy as np
from sympy import Line2D
from config import AutoConfig
from config_utils import get_cfg_defaults
import matplotlib.pyplot as plt
import pandas as pd
import torch

from read_utils import read_config, read_short_config, read_score_df, list_runs_from_exp_names
# %%
def get_parser():
    parser = argparse.ArgumentParser(description="Ray Tune")
    parser.add_argument("--save_dir", type=str, default="/nfscc/alg23/xdcab/dev", help="save dir")
    parser.add_argument("--exp_dir", type=str, default="/nfscc/alg23/xdcaa/topyneck", help="exp dir")
    return parser
args = get_parser().parse_args()
# %%
save_dir = args.save_dir

run_dirs = list_runs_from_exp_names(
    [""], exp_dir=f"{args.exp_dir}"
)
print(len(run_dirs))

# %%
datas = []
for run in run_dirs:
    cfg: AutoConfig = read_config(run)
    subject = cfg.DATASET.SUBJECT_LIST[0]
    reg = cfg.REGULARIZER.LAYER
    val_score = torch.load(os.path.join(run, "soup_val_score.pth"))
    test_score = torch.load(os.path.join(run, "soup_test_score.pth"))
    datas.append([subject, reg, val_score, test_score, run])
df = pd.DataFrame(datas, columns=["subject", "reg", "val_score", 'test_score', 'run'])
# add mean_score column
df['mean_score'] = df[['val_score', 'test_score']].mean(axis=1)
# %%
topyneck_dict = {}
for subject in df.subject.unique():
    # find best reg
    df_subject = df[df.subject == subject]
    best_reg = df_subject[df_subject.mean_score == df_subject.mean_score.max()].reg.values[0]
    print(f"subject: {subject}, best reg: {best_reg}, best score: {df_subject.mean_score.max()}")
    best_run = df_subject[df_subject.mean_score == df_subject.mean_score.max()].run.values[0]
    
    sd = torch.load(os.path.join(best_run, "soup.pth"), map_location=torch.device('cpu'))
    names = ['retina_mapper', 'layer_selector']
    sd = {k: v for k, v in sd.items() if any(name in k for name in names)}
    topyneck_dict.update(sd)
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "topyneck.pth")
torch.save(topyneck_dict, save_path)
print(f'saved to {save_path}')