import argparse
import os
import numpy as np

from datamodule import NSDDatamodule
from datasets import NSDDataset

parser = argparse.ArgumentParser(description="")
parser.add_argument("--dark_name", type=str, default="xvbaa")
parser.add_argument("--data_dir", type=str, default="/data/ALG23/")
parser.add_argument("--alg_dir", type=str, default="/nfscc/algonauts2023/")
parser.add_argument("--save_dir", type=str, default="/nfscc/alg23/submission/xvba")
args = parser.parse_args()

from config_utils import get_cfg_defaults
from config import AutoConfig

SPACE = 'fsaverage'
N = 327684

for _i_subject in range(1, 9):
    subject = f"subj{_i_subject:02d}"

    cfg = get_cfg_defaults()
    cfg.DATASET.FMRI_SPACE = SPACE
    cfg.DATASET.ROIS = ["all"]
    cfg.DATASET.DARK_POSTFIX = args.dark_name
    cfg.DATASET.SUBJECT_LIST = [subject]

    dm = NSDDatamodule(cfg)
    dm.setup()

    dataloader = dm.predict_dataloader(subject=subject)
    dataset: NSDDataset = dataloader.dataset

    # /data/ALG23/subj08/image_ids/challenge_set.txt
    # /data/ALG23/subj01/image_ids/predict_set.txt

    predict_images = np.loadtxt(f"{args.data_dir}/{subject}/image_ids/predict_set.txt", dtype=int)
    challenge_images = np.loadtxt(f"{args.data_dir}/{subject}/image_ids/challenge_set.txt", dtype=int)

    # /data/ALG23/subj01/data_mask/fsaverage/voxel_indices.npy
    voxel_indices = np.load(f"{args.data_dir}/{subject}/data_mask/{SPACE}/voxel_indices.npy")

    # /nfscc/algonauts2023/subj01/roi_masks/lh.all-vertices_fsaverage_space.npy
    lh_mask = np.load(f"{args.alg_dir}/{subject}/roi_masks/lh.all-vertices_fsaverage_space.npy")
    rh_mask = np.load(f"{args.alg_dir}/{subject}/roi_masks/rh.all-vertices_fsaverage_space.npy")
    challenge_mask = np.concatenate([lh_mask, rh_mask], axis=0)
    num_lh = lh_mask.sum()
    num_rh = rh_mask.sum()
    num_challenge = challenge_mask.sum()

    # d = np.zeros(N, dtype=np.float32)

    lh, rh = [], []
    for image_id in challenge_images:
        idxs = np.where(predict_images == image_id)[0]
        _lh, _rh = [], []
        for _i in idxs:
            _data = dataset.load_one_dark(_i, args.dark_name)
            _data = _data[:num_challenge]
            _lh.append(_data[:num_lh])
            _rh.append(_data[num_lh:])
        _lh = np.stack(_lh, axis=0).mean(axis=0)
        _rh = np.stack(_rh, axis=0).mean(axis=0)
        lh.append(_lh)
        rh.append(_rh)
    lh = np.stack(lh, axis=0)
    rh = np.stack(rh, axis=0)

    save_dir = f"{args.save_dir}/{subject}"
    os.makedirs(save_dir, exist_ok=True)

    # lh_pred_test.npy
    np.save(f"{save_dir}/lh_pred_test.npy", lh)
    np.save(f"{save_dir}/rh_pred_test.npy", rh)