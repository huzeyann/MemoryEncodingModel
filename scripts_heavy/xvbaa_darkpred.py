import os
import torch
from dark_onemodel import build_dmt, get_outs

import argparse

from datasets import NSDDataset

parser = argparse.ArgumentParser()
parser.add_argument("--run_dir", type=str, default="/nfscc/alg23/xvba/roiall_bsz/tb314c_00000_TRAINER.ACCUMULATE_GRAD_BATCHES=1/")
parser.add_argument("--save_name", type=str, default="xvbaa")
parser.add_argument("--stage", type=str, default="predict")
args = parser.parse_args()

dm, plmodel, trainer = build_dmt(args.run_dir)
soup = torch.load(os.path.join(args.run_dir, "soup.pth"), map_location=torch.device('cpu'))
plmodel.load_state_dict(soup)
plmodel.eval()

subject_list = dm.cfg.DATASET.SUBJECT_LIST

for subject in subject_list:
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
    # dataloader = dm.predict_dataloader(subject=subject)
    outs = get_outs(plmodel, trainer, dataloader)

    dataset : NSDDataset = dataloader.dataset
    dataset.save_dark(outs, args.save_name)