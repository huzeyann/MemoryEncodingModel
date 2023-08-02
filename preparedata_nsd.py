import shutil
from tqdm import tqdm
import os
import numpy as np
import torch
import pandas as pd

import glob
import logging

import argparse
from pathlib import Path

import nilearn
import nibabel as nib

from PIL import Image

import copy

# get args
parser = argparse.ArgumentParser(description="prepare data for nsd")
parser.add_argument("--nsd_dir", type=str, default="/nfscc/natural-scenes-dataset")
parser.add_argument("--alg_dir", type=str, default="/nfscc/algonauts2023")
parser.add_argument("--output_dir", type=str, default="/data/ALG23")
sh = "fsaverage: nsdgeneral + high nc\n func1mm: nsdgeneral + high nc\n full_fsaverage: full, no filter by nc\n full_func1mm: full, no filter by nc\n hip_fun1mm: hippocampus \n fship: fsaverage+hippocampus, run this after running full_fsaverage and hip_fun1mm\n"
parser.add_argument(
    "--space",
    type=str,
    default="fsaverage",
    choices=[
        "fsaverage",
        "func1mm",
        "full_fsaverage",
        "full_func1mm",
        "hip_func1mm",
        "fship",
    ],
    help=sh,
)
parser.add_argument("--beta", type=str, default="b3", choices=["b3", "b2"])
parser.add_argument("--nc_threshold", type=float, default=-1)  # -1 for auto
parser.add_argument("--val1_ratio", type=float, default=0.04)
parser.add_argument("--val2_ratio", type=float, default=0.02)
parser.add_argument("--image_size", type=int, default=224)
parser.add_argument("--image_overwrite", action="store_true")
parser.add_argument("--jpeg_quality", type=int, default=95)
parser.add_argument("--seed", type=int, default=114514)
parser.add_argument("--n_jobs", type=int, default=0)
parser.add_argument("--skip_fmri", action="store_true")

args = parser.parse_args()

# check args
# assert args.space in ['fsaverage',
#                       'func1mm', 'both'], 'space not in [fsaverage, func1mm, both]'
assert os.path.exists(args.nsd_dir), "nsd_dir not exists"
assert os.path.exists(args.alg_dir), "alg_dir not exists"

os.makedirs(args.output_dir, exist_ok=True)

NUM_SUBJECTS = 8
SUBJECT_NAMES = ["subj%02d" % (i + 1) for i in range(NUM_SUBJECTS)]
NUM_SESSIONS = [40, 40, 32, 30, 40, 32, 40, 30]

#############################################
### load experiment design ###
#############################################

"""
.mat contents:
<masterordering> is 1 x 30000 with the sequence of trials (indices relative to 10k)
<basiccnt> is 3 x 40 where we calculate, for each scan session separately, the number of distinct images in that session that have a number of presentations equal to the row index.
<sharedix> is 1 x 1000 with sorted indices of the shared images (relative to 73k)
<subjectim> is 8 x 10000 with indices of images (relative to 73k). the first 1000 are the common shared 1000 images. it turns out that the indices for these 1000 are in sorted order. this is for simplicity, and there is no significance to the order (since the order in which the 1000 images are shown is randomly determined). the remaining 9000 for each subject are in a randomized non-sorted order.
<stimpattern> is 40 sessions x 12 runs x 75 trials. elements are 0/1 indicating when stimulus trials actually occur. note that the same <stimpattern> is used for all subjects.

Note: subjectim(:,masterordering) is 8 x 30000 indicating the temporal sequence of 73k-ids shown to each subject. This sequence refers only to the stimulus trials (ignoring the blank trials and the rest periods at the beginning and end of each run).
Note: All of these indices (in the nsd_expdesign.mat file) are 1-based indices.
"""


def load_mat(mat_path):
    import scipy.io as sio

    mat = sio.loadmat(mat_path)
    return mat


# /nfscc/natural-scenes-dataset/nsddata/experiments/nsd/nsd_expdesign.mat
mat_path = os.path.join(
    args.nsd_dir, "nsddata", "experiments", "nsd", "nsd_expdesign.mat"
)
expdesign = load_mat(mat_path)

subject_expdesign = {}  # 40 x 12 x 75, 0-based, -1 for blank
subject_image_with_memory = {}  # 40 x 750 x 3, 0-based, -1 for blank
for subject_name in SUBJECT_NAMES:
    subject_id = SUBJECT_NAMES.index(subject_name)

    # load hacky experiment design
    subjectim = expdesign["subjectim"][subject_id, :]
    masterordering = expdesign["masterordering"][0, :]
    masterordering = masterordering - 1  # 0-based
    stimpattern = expdesign["stimpattern"]
    subjectim = subjectim[masterordering]  # 1-based, 30000
    subjectim = subjectim - 1  # 0-based, 30000

    stimpattern = stimpattern.astype(np.int32)
    stimpattern[stimpattern == 0] = -1  # blank trials
    stimpattern[stimpattern == 1] = subjectim  # stimulus trials
    subject_expdesign[subject_name] = stimpattern

    memory_size = 33
    image_with_memory = np.zeros((40, 750, memory_size), dtype=np.int32)
    for session_id in range(40):
        i_img = 0
        for run_id in range(12):
            for trial_id in range(75):
                sp = copy.deepcopy(stimpattern[session_id, run_id])  # 0-based, [75]
                # pad left with -1, size is memory_size + size
                sp = np.pad(sp, (memory_size, 0), mode="constant", constant_values=-1)
                trial_id += memory_size
                if sp[trial_id] != -1:
                    image_with_memory[session_id, i_img] = sp[
                        trial_id - memory_size + 1 : trial_id + 1
                    ]
                    i_img += 1
        assert i_img == 750
    subject_image_with_memory[subject_name] = image_with_memory

    # save
    save_dir = os.path.join(args.output_dir, subject_name)
    os.makedirs(save_dir, exist_ok=True)
    # 40 x 12 x 75, 0-based, -1 for blank
    np.save(os.path.join(save_dir, "expdesign.npy"), stimpattern)
    np.save(
        os.path.join(save_dir, "image_with_memory.npy"), image_with_memory
    )  # 40 x 750 x 3, 0-based, -1 for blank

    #############################
    ### verify stimpattern ###
    #############################

    # load public experiment design
    # /nfscc/natural-scenes-dataset/nsddata_timeseries/ppdata/subj01/func1pt8mm/design/design_session01_run01.tsv
    subject_dir = os.path.join(
        args.nsd_dir,
        "nsddata_timeseries",
        "ppdata",
        subject_name,
        "func1pt8mm",
        "design",
    )
    design_files = glob.glob(os.path.join(subject_dir, "design_session*_run*.tsv"))
    design_files = sorted(design_files)

    # check if stimpattern is the same as the public design
    no_error = True
    zero_counter = {}
    for i, design_file in enumerate(design_files):
        pub_design = np.loadtxt(design_file)
        session_id = int(design_file.split("session")[1][:2]) - 1
        run_id = int(design_file.split("run")[1][:2]) - 1

        if session_id not in zero_counter:
            zero_counter[session_id] = 0
        if np.all(pub_design == 0):
            # Note: there is run13 run14 after session20, I don't know why
            zero_counter[session_id] += 1
            continue

        run_id = run_id - zero_counter[session_id]
        my_design = copy.deepcopy(stimpattern[session_id, run_id, :])  # 1-based, 75
        my_design += 1  # 2-based, 75

        def add_two_zero_after_each_element(arr):
            new_arr = np.zeros((arr.shape[0] * 3), dtype=arr.dtype)
            for i in range(arr.shape[0]):
                new_arr[i * 3] = arr[i]
            return new_arr

        my_design = add_two_zero_after_each_element(my_design)
        # assert np.all(design == my_design), 'design is not the same'
        if not np.all(pub_design == my_design):
            logging.error(
                f"design is not the same for subject {subject_name} session {session_id} run {run_id}"
            )
            logging.error(
                f"design shape: {pub_design.shape}, my_design shape: {my_design.shape}"
            )
            logging.error(
                f"design_mean: {pub_design.mean()}, my_design_mean: {my_design.mean()}"
            )
            logging.error(f"design_file: {design_file}")
            no_error = False

    assert no_error, "design is not the same"


#############################################
### load split and save ###
#############################################

print("loading split...")


for subject_name in SUBJECT_NAMES:
    subject_id = SUBJECT_NAMES.index(subject_name)

    # /nfscc/algonauts2023/subj01/test_split/test_images/test-0001_nsd-00845.png
    test_pngs = glob.glob(
        os.path.join(args.alg_dir, subject_name, "test_split", "test_images", "*.png")
    )
    test_pngs = sorted(test_pngs)
    # 00845 is 0-based
    challenge_test_image_ids = [
        int(os.path.basename(png).split("_")[1].split(".")[0].split("-")[1])
        for png in test_pngs
    ]
    challenge_test_image_ids = np.array(challenge_test_image_ids)

    num_sessions = NUM_SESSIONS[subject_id]
    holdout_session_ids = np.arange(num_sessions - 3, num_sessions)
    training_session_ids = np.arange(num_sessions - 3)

    # this code will cause data leak, because images are repeated 3 times
    # train_idxs, val1_idxs, val2_idxs = [], [], []
    # for session_id in training_session_ids:
    #     session_idxs = np.arange(session_id*750, (session_id+1)*750)
    #     np.random.seed(args.seed)
    #     np.random.shuffle(session_idxs)
    #     train_idxs.append(
    #         session_idxs[:-int((args.val1_ratio + args.val2_ratio)*750)])
    #     val1_idxs.append(
    #         session_idxs[-int((args.val1_ratio + args.val2_ratio)*750):-int(args.val2_ratio*750)])
    #     val2_idxs.append(session_idxs[-int(args.val2_ratio*750):])
    # train_idxs = np.concatenate(train_idxs)
    # val1_idxs = np.concatenate(val1_idxs)
    # val2_idxs = np.concatenate(val2_idxs)
    training_image_ids = subject_image_with_memory[subject_name][training_session_ids]
    training_image_ids = training_image_ids[:, :, -1].reshape(-1)
    unique_training_image_ids = np.unique(training_image_ids)
    np.random.seed(args.seed)
    np.random.shuffle(unique_training_image_ids)
    total_len = len(unique_training_image_ids)
    train_image_ids = unique_training_image_ids[
        : int(total_len * (1 - args.val1_ratio - args.val2_ratio))
    ]
    val1_image_ids = unique_training_image_ids[
        int(total_len * (1 - args.val1_ratio - args.val2_ratio)) : int(
            total_len * (1 - args.val2_ratio)
        )
    ]
    val2_image_ids = unique_training_image_ids[int(total_len * (1 - args.val2_ratio)) :]

    def get_idxs(image_ids):
        idxs = []
        for image_id in image_ids:
            idxs.append(np.where(training_image_ids == image_id)[0])
        return np.concatenate(idxs)

    train_idxs = get_idxs(train_image_ids)
    val1_idxs = get_idxs(val1_image_ids)
    val2_idxs = get_idxs(val2_image_ids)

    predict_idxs = np.arange(
        holdout_session_ids[0] * 750, (holdout_session_ids[-1] + 1) * 750
    )

    def save_list_to_file(list, file):
        with open(file, "w") as f:
            for item in list:
                f.write("%s\n" % item)

    save_dir = os.path.join(args.output_dir, subject_name, "split")
    os.makedirs(save_dir, exist_ok=True)
    save_list_to_file(train_idxs, os.path.join(save_dir, "train.txt"))
    save_list_to_file(val1_idxs, os.path.join(save_dir, "val1.txt"))
    save_list_to_file(val2_idxs, os.path.join(save_dir, "val2.txt"))
    save_list_to_file(predict_idxs, os.path.join(save_dir, "predict.txt"))

    # for challenge
    all_image_ids = subject_image_with_memory[subject_name]
    all_image_ids = all_image_ids.reshape(-1, memory_size)[:, -1]
    all_image_ids = all_image_ids.astype(int)
    predict_set_image_ids = all_image_ids[predict_idxs]
    training_set_image_ids = np.concatenate(
        [all_image_ids[train_idxs], all_image_ids[val1_idxs], all_image_ids[val2_idxs]]
    )

    save_dir = os.path.join(args.output_dir, subject_name, "image_ids")
    os.makedirs(save_dir, exist_ok=True)
    save_list_to_file(
        challenge_test_image_ids, os.path.join(save_dir, "challenge_set.txt")
    )
    save_list_to_file(predict_set_image_ids, os.path.join(save_dir, "predict_set.txt"))

    #############################################
    ### verify ###
    #############################################

    # verify if all the challenge set images are in the predict set
    assert np.all(
        np.isin(challenge_test_image_ids, predict_set_image_ids)
    ), f"challenge set images are not in the predict set for subject {subject_name}"
    # # verify if all the predict set images not in challenge set are in the training set
    # assert np.all(np.isin(predict_set_image_ids, challenge_test_image_ids)) or \
    #     np.all(np.isin(predict_set_image_ids, training_set_image_ids)), \
    #     f"predict set images not in challenge set are not in the training set for subject {subject_name}"

    predict_set_image_ids_unique, predict_set_image_ids_counts = np.unique(
        predict_set_image_ids, return_counts=True
    )

    # counts number of repeats in the predict set for challenge set images
    counts = []
    for image_id in challenge_test_image_ids:
        counts.append(np.sum(predict_set_image_ids == image_id))
    counts = np.array(counts)
    print(
        f"number of repeats in the predict set for challenge set images for subject {subject_name}:"
    )
    print(f"min: {np.min(counts)}, max: {np.max(counts)}, mean: {np.mean(counts)}")

    # counts predicts images in the training set
    counts = []
    for image_id in predict_set_image_ids_unique:
        counts.append(np.sum(training_set_image_ids == image_id))
    counts = np.array(counts)
    print(f"number of predicts images in the training set for subject {subject_name}:")
    print(f"min: {np.min(counts)}, max: {np.max(counts)}, mean: {np.mean(counts)}")

    # counts challenge set images in the training set
    counts = []
    for image_id in challenge_test_image_ids:
        counts.append(np.sum(training_set_image_ids == image_id))
    counts = np.array(counts)
    print(
        f"number of challenge set images in the training set for subject {subject_name}:"
    )
    print(f"min: {np.min(counts)}, max: {np.max(counts)}, mean: {np.mean(counts)}")


# note: session id is idx // 750


#############################################
### load image data ###
#############################################

print("loading image data...")

# /nfscc/algonauts2023/subj01/test_split/test_images/test-0001_nsd-00845.png
all_images = glob.glob(os.path.join(args.alg_dir, "**/*.png"), recursive=True)
# 00845, 0-based
all_image_ids = [
    int(os.path.basename(image).split("_")[1].split(".")[0].split("-")[1])
    for image in all_images
]
# all_image_ids = np.array(all_image_ids, dtype=np.int32)

# save all images as jpeg and keep a dictionary of image id to image path
save_dir = os.path.join(args.output_dir, "images")
os.makedirs(save_dir, exist_ok=True)
image_id_to_path = {}
if args.n_jobs != 0:
    from multiprocessing import Pool
    from functools import partial

    def save_image(i):
        image_id, image = all_image_ids[i], all_images[i]
        save_path = os.path.join(args.output_dir, "images", f"{image_id:05d}.jpeg")
        if os.path.exists(save_path) and not args.image_overwrite:
            return
        image_jpeg = Image.open(image).convert("RGB")
        image_jpeg = image_jpeg.resize((args.image_size, args.image_size))
        image_jpeg.save(save_path, "JPEG", quality=args.jpeg_quality)

    with Pool(args.n_jobs) as p:
        p.map(save_image, range(len(all_images)))

for image_id, image in tqdm(
    zip(all_image_ids, all_images), total=len(all_images), desc="saving images"
):
    save_path = os.path.join(args.output_dir, "images", f"{image_id:05d}.jpeg")
    image_id_to_path[image_id] = save_path
    if os.path.exists(save_path) and not args.image_overwrite:
        continue
    image_jpeg = Image.open(image).convert("RGB")
    image_jpeg = image_jpeg.resize((args.image_size, args.image_size))
    image_jpeg.save(save_path, "JPEG", quality=args.jpeg_quality)


# # save image id to path as json
# import json
# json_path = os.path.join(args.output_dir, 'image_path.json')
# with open(json_path, 'w') as f:
#     json.dump(image_id_to_path, f, indent=4, sort_keys=True)


#############################################
### load behavior data ###
#############################################

print("loading behavior data...")


def z_score(data, axis=0):
    data = data - data.mean(axis=axis, keepdims=True)
    data = data / (data.std(axis=axis, keepdims=True) + 1e-8)
    return data


subject_bhvdata = {}
for subject_name in SUBJECT_NAMES:
    subject_id = SUBJECT_NAMES.index(subject_name)

    # /nfscc/natural-scenes-dataset/nsddata/ppdata/subj01/behav/responses.tsv
    path = os.path.join(
        args.nsd_dir, "nsddata", "ppdata", subject_name, "behav", "responses.tsv"
    )
    responses = pd.read_csv(path, sep="\t")

    """
    tsv file format:
    SUBJECT	SESSION	RUN	TRIAL	73KID	10KID	TIME	ISOLD	ISCORRECT	RT	CHANGEMIND	MEMORYRECENT	MEMORYFIRST	ISOLDCURRENT	ISCORRECTCURRENT	TOTAL1	TOTAL2	BUTTON	MISSINGDATA
    1	1	1	1	46003	626	0.5050821574404835701	0	1	803.52978099836036563	0	NaN	NaN	0	1	1	0	1	0
    """

    missing_data = responses["MISSINGDATA"].values
    missing_data = missing_data.astype(np.int32)
    np.nan_to_num(missing_data, copy=False, nan=0)

    def z(x):
        return z_score(x, axis=0)

    def sz(x):
        x = x.reshape(-1, 750)
        x = z_score(x, axis=1)
        x = x.reshape(-1)
        return x

    rt = responses["RT"].values
    fill_value = np.nanmean(rt)
    np.nan_to_num(rt, copy=False, nan=fill_value)
    cm = responses["CHANGEMIND"].values
    fill_value = np.nanmean(cm)
    np.nan_to_num(cm, copy=False, nan=fill_value)
    m1 = copy.deepcopy(responses["MEMORYRECENT"].values)
    m1[~np.isnan(m1)] = 1  # 1: old, 0: new
    m1[np.isnan(m1)] = 0
    m2 = copy.deepcopy(responses["MEMORYFIRST"].values)
    m2[~np.isnan(m2)] = 1  # 1: old, 0: new
    m2[np.isnan(m2)] = 0
    m3 = responses["ISOLDCURRENT"].values
    m3 = m3.astype(np.int32)
    m3[m3 == 1] = 1  # 1: old, 0: new
    m3[m3 == 0] = 0
    c1 = responses["ISCORRECT"].values
    fill_value = np.nanmean(c1)
    np.nan_to_num(c1, copy=False, nan=fill_value)
    c2 = responses["ISCORRECTCURRENT"].values
    fill_value = np.nanmean(c2)
    np.nan_to_num(c2, copy=False, nan=fill_value)
    bt = responses["BUTTON"].values
    fill_value = np.nanmean(bt)
    np.nan_to_num(bt, copy=False, nan=fill_value)
    n1 = responses["TOTAL1"].values
    fill_value = np.nanmean(n1)
    np.nan_to_num(n1, copy=False, nan=fill_value)
    n2 = responses["TOTAL2"].values
    fill_value = np.nanmean(n2)
    np.nan_to_num(n2, copy=False, nan=fill_value)
    mi = responses["MISSINGDATA"].values
    ts = responses["TIME"].values
    r1 = responses["MEMORYRECENT"].values
    r2 = responses["MEMORYFIRST"].values
    t1 = np.zeros_like(r1)
    t2 = np.zeros_like(r2)
    for __i in range(len(r1)):
        if np.isnan(r1[__i]):
            t1[__i] = 500
        else:
            t1[__i] = ts[__i] - ts[__i - int(r1[__i]) - 1]
        if np.isnan(r2[__i]):
            t2[__i] = 500
        else:
            t2[__i] = ts[__i] - ts[__i - int(r2[__i]) - 1]
    fill_value = 15000
    np.nan_to_num(r1, copy=False, nan=fill_value)
    np.nan_to_num(r2, copy=False, nan=fill_value)
    r1 = np.log(r1 + 1)
    r2 = np.log(r2 + 1)
    t1 = np.log(t1 * 24 * 60 + 1)
    t2 = np.log(t2 * 24 * 60 + 1)

    # TODO: fill session id in predicting data
    se = responses["SESSION"].values
    ru = responses["RUN"].values
    tr = responses["TRIAL"].values
    su = responses["SUBJECT"].values

    bhv_data = np.stack(
        [
            z(rt),
            sz(rt),
            bt,
            cm,
            n1,
            n2,
            c1,
            c2,
            m1,
            m2,
            m3,
            z(t1),
            sz(t1),
            z(t2),
            sz(t2),
            z(r1),
            sz(r1),
            z(r2),
            sz(r2),
            (ru - 6) / 12,
            (tr - 30) / 60,
        ],
        axis=1,
    )
    fill_bhv = np.array(
        [[0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    pi = np.array([0, 1, 2, 11, 12, 15, 16])  # index for future/past response

    # include previous trial and future trial
    pad_bhv_data = np.concatenate([fill_bhv, bhv_data], axis=0)
    p = pi
    c = np.arange(bhv_data.shape[1]) + bhv_data.shape[1]
    f = p + bhv_data.shape[1] * 2
    indices = np.concatenate([c, p, f])
    included_bhv_data = []
    for _i_trail in range(len(bhv_data)):
        id_10k = masterordering[_i_trail]
        _i_same_stim_trial = np.where(masterordering == id_10k)[0]
        # pad previous and future trials with -1
        _i_same_stim_trial = np.concatenate([[-1], _i_same_stim_trial, [-1]])
        _i_repeat = np.where(_i_same_stim_trial == _i_trail)[0][0]
        _idxs = _i_same_stim_trial[_i_repeat - 1 : _i_repeat + 2]
        _idxs[_idxs >= len(bhv_data)] = -1
        _idxs += 1
        _data = pad_bhv_data[_idxs]
        _data = _data.reshape(-1)[indices]
        included_bhv_data.append(_data)

    bhv_data = np.stack(included_bhv_data, axis=0)

    bhv_data = bhv_data.astype(np.float32)

    last_bhv_data = np.concatenate(
        [fill_bhv, fill_bhv[:, pi], fill_bhv[:, pi]], axis=1
    ).astype(np.float32)
    bhv_data = np.concatenate([bhv_data, last_bhv_data.reshape(1, -1)], axis=0)

    print(f"behavior data shape: {bhv_data.shape}")
    print(f"behavior data max: {bhv_data.max()}, min: {bhv_data.min()}")

    # verify
    assert (
        bhv_data.shape[0] == 750 * NUM_SESSIONS[subject_id] + 1
    ), f"behavior data shape is not correct for subject {subject_name}"
    assert (
        np.isnan(bhv_data).sum() == 0
    ), f"behavior data contains nan for subject {subject_name}"

    # save
    path = os.path.join(args.output_dir, subject_name, "behavior_data.npy")
    np.save(path, bhv_data)

    subject_bhvdata[subject_name] = bhv_data

#############################################
### load fsaverage space, filter by nc ###
#############################################

if args.beta == "b3":
    beta_version = "betas_fithrf_GLMdenoise_RR"
elif args.beta == "b2":
    beta_version = "betas_fithrf"
else:
    raise ValueError(f"beta version {args.beta} is not supported")


def add_beta_to_name(save_name, beta_version):
    if beta_version == "b3":
        pass
    elif beta_version == "b2":
        save_name += "_b2"
    else:
        raise ValueError(f"beta version {beta_version} is not supported")
    return save_name


def z_score(data, axis=1):
    data = data - data.mean(axis=axis, keepdims=True)
    data = data / (data.std(axis=axis, keepdims=True) + 1e-8)
    return data


if args.space == "fsaverage":
    save_name = "fsaverage"
    save_name = add_beta_to_name(save_name, args.beta)
    print("loading fsaverage space...")

    for subject_name in SUBJECT_NAMES:
        subject_id = SUBJECT_NAMES.index(subject_name)

        # load challenge mask (nsdgeneral + RSC)
        # /nfscc/algonauts2023/subj01/roi_masks/lh.all-vertices_fsaverage_space.npy
        path = os.path.join(
            args.alg_dir,
            subject_name,
            "roi_masks",
            "lh.all-vertices_fsaverage_space.npy",
        )
        lh_mask = np.load(path)
        path = os.path.join(
            args.alg_dir,
            subject_name,
            "roi_masks",
            "rh.all-vertices_fsaverage_space.npy",
        )
        rh_mask = np.load(path)
        challenge_mask = np.concatenate([lh_mask, rh_mask], axis=0)

        # load snr
        # /nfscc/natural-scenes-dataset/nsddata_betas/ppdata/subj01/fsaverage/betas_fithrf_GLMdenoise_RR/lh.ncsnr.mgh
        path = os.path.join(
            args.nsd_dir,
            "nsddata_betas",
            "ppdata",
            subject_name,
            "fsaverage",
            beta_version,
            "lh.ncsnr.mgh",
        )
        lh_snr = nib.load(path).get_fdata().flatten()
        path = os.path.join(
            args.nsd_dir,
            "nsddata_betas",
            "ppdata",
            subject_name,
            "fsaverage",
            beta_version,
            "rh.ncsnr.mgh",
        )
        rh_snr = nib.load(path).get_fdata().flatten()
        snr = np.concatenate([lh_snr, rh_snr], axis=0)
        nc = snr**2 / (snr**2 + 1 / 3)
        # apply mask
        m_snr = snr[challenge_mask == 1]
        m_nc = nc[challenge_mask == 1]

        nc_th = args.nc_threshold
        if nc_th > 0 and nc_th < 1:
            pass
        elif nc_th > 1:
            nc_th = np.percentile(m_nc, nc_th)
        else:
            # find nc_th so added nc mean equals to challenge nc mean
            nc_th = m_nc.mean()
            add_nc_mean = 1.0
            target_nc_mean = m_nc.mean()
            while not np.isclose(add_nc_mean, target_nc_mean, atol=0.001):
                if add_nc_mean > target_nc_mean:
                    nc_th *= 0.5
                else:
                    nc_th *= 1.5
                nc_mask = nc > nc_th
                full_mask = nc_mask | challenge_mask
                new_mask = full_mask & (~challenge_mask)
                add_nc = nc[new_mask == 1]
                add_nc_mean = add_nc.mean()
            print(f"auto nc_th: {nc_th}")

        # apply nc threshold
        nc_mask = nc > nc_th
        full_mask = nc_mask | challenge_mask
        # print(f"subject: {subject_name}, nc_voxels: {nc_voxels.sum()}")
        new_mask = full_mask & (~challenge_mask)
        print(
            f"subject: {subject_name}, challenge_mask: {challenge_mask.sum()}, full_voxels: {full_mask.sum()}, new_voxels: {new_mask.sum()}"
        )
        print(f"NC mean: {m_nc.mean()}, NC std: {m_nc.std()}")
        add_nc = nc[new_mask == 1]
        for i in range(1, 10):
            print(f"{i}%: {np.percentile(m_nc, i * 10):.3f}", end=", ")
        print()
        print(f"add NC mean: {add_nc.mean()}, add NC std: {add_nc.std()}")

        # save data voxel indices
        # voxel indices is indices of voxels in the `full_data`
        challenge_voxel_indices = np.where(challenge_mask == 1)[0]
        new_voxel_indices = np.where(new_mask == 1)[0]
        full_voxel_indices = np.concatenate(
            [challenge_voxel_indices, new_voxel_indices], axis=0
        )
        save_dir = os.path.join(args.output_dir, subject_name, "data_mask", save_name)
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "mask.npy"), full_mask)
        np.save(os.path.join(save_dir, "voxel_indices.npy"), full_voxel_indices)

        # save roi voxel indices
        # roi voxel indices is indices of voxels in the `full_voxel_indices`
        # /nfscc/algonauts2023/subj01/roi_masks/lh.streams_challenge_space.npy
        path = os.path.join(
            args.alg_dir, subject_name, "roi_masks", "lh.streams_challenge_space.npy"
        )
        lh_streams_challenge_space = np.load(path)
        path = os.path.join(
            args.alg_dir, subject_name, "roi_masks", "rh.streams_challenge_space.npy"
        )
        rh_streams_challenge_space = np.load(path)
        streams_challenge_space = np.concatenate(
            [lh_streams_challenge_space, rh_streams_challenge_space], axis=0
        )
        v = streams_challenge_space
        names = ["RSC", "E", "MV", "ML", "MP", "V", "L", "P"]
        roi_vi = []
        for _iv, name in enumerate(names):
            _voxel_indices = np.where(v == _iv)[0]
            save_dir = os.path.join(args.output_dir, subject_name, "roi", save_name)
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, f"{name}.npy"), _voxel_indices)
            roi_vi.append(_voxel_indices)
        roi_voxel_indices = np.concatenate(roi_vi, axis=0)
        orig = np.arange(len(challenge_voxel_indices))
        added = np.arange(len(challenge_voxel_indices), len(full_voxel_indices))
        rest = np.array(
            list(set(np.arange(len(full_voxel_indices))) - set(roi_voxel_indices))
        )
        assert np.all(rest == added), f"{subject_name} rest != added"
        np.save(os.path.join(save_dir, "orig.npy"), orig)
        np.save(os.path.join(save_dir, "added.npy"), added)
        np.save(os.path.join(save_dir, "R.npy"), rest)

        # 10x9 random roi
        all_vi = np.arange(len(full_voxel_indices))
        np.random.seed(45510)
        for _i_roi in range(10):
            np.random.shuffle(all_vi)
            # divide into 9 even sized chunks
            chunks = np.array_split(all_vi, 9)
            for _i_chunk in range(9):
                np.save(
                    os.path.join(save_dir, f"r_{_i_roi+1}_{_i_chunk+1}.npy"),
                    chunks[_i_chunk],
                )

        # save voxel coordinates
        import nilearn
        from nilearn import datasets, surface

        fsaverage = nilearn.datasets.fetch_surf_fsaverage("fsaverage7")
        lh_coords, lh_faces = nilearn.surface.load_surf_mesh(fsaverage["sphere_left"])
        rh_coords, rh_faces = nilearn.surface.load_surf_mesh(fsaverage["sphere_right"])
        lh_xmin, lh_xmax = np.min(lh_coords[:, 0]), np.max(lh_coords[:, 0])
        lh_xmax = lh_xmin + (lh_xmax - lh_xmin) * 1.5
        rh_xmin, rh_xmax = np.min(rh_coords[:, 0]), np.max(rh_coords[:, 0])
        if rh_xmin < lh_xmax:
            rh_coords[:, 0] += lh_xmax - rh_xmin
        coords = np.concatenate((lh_coords, rh_coords), axis=0)
        coords = coords[full_voxel_indices]
        print(f"subject: {subject_name}, coords: {coords.shape}")
        save_dir = os.path.join(args.output_dir, subject_name, "coords", save_name)
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "coords.npy"), coords)

        #############################################
        ### load fmri data ###
        #############################################

        if args.skip_fmri:
            continue

        print(f"loading fmri data for subject: {subject_name}...")

        # /data/natural-scenes-dataset/nsddata_betas/ppdata/subj01/fsaverage/betas_fithrf_GLMdenoise_RR/lh.betas_session01.mgh
        for session_id in tqdm(
            range(1, NUM_SESSIONS[subject_id] + 1 - 3), desc="session"
        ):
            path = os.path.join(
                args.nsd_dir,
                "nsddata_betas",
                "ppdata",
                subject_name,
                "fsaverage",
                beta_version,
                f"lh.betas_session{session_id:02d}.mgh",
            )
            lh_data = nib.load(path).get_fdata()
            path = os.path.join(
                args.nsd_dir,
                "nsddata_betas",
                "ppdata",
                subject_name,
                "fsaverage",
                beta_version,
                f"rh.betas_session{session_id:02d}.mgh",
            )
            rh_data = nib.load(path).get_fdata()
            data = np.concatenate((lh_data, rh_data), axis=0)
            assert data.shape == (
                327684,
                1,
                1,
                750,
            ), f"{subject_name} {session_id} {data.shape}"
            data = data.squeeze()
            data = data[full_voxel_indices]
            # fill nan with 0
            data = np.nan_to_num(data)
            data = data.astype(np.float32)
            data = z_score(data, axis=1)
            data = data.astype(np.float16)

            # assert no nan
            assert np.isnan(data).sum() == 0

            save_dir = os.path.join(args.output_dir, subject_name, "fmri", save_name)
            os.makedirs(save_dir, exist_ok=True)

            for i in range(750):
                save_i = (session_id - 1) * 750 + i
                np.save(os.path.join(save_dir, f"{save_i:06d}.npy"), data[:, i])


#############################################
### load fsaverage space, all vertices ###
#############################################


"""
HCPMMP1:

The 22 Cortices ("Regions"):
The first five regions cover early and intermediate visual cortex:
1) Primary_Visual
2) Early_Visual
3) Dorsal_Stream_Visual
4) Ventral_Stream_Visual
5) MT+_Complex_and_Neighboring_Visual_Areas

The next four regions cover the  sensorimotor areas:
6) Somatosensory_and_Motor
7) Paracentral_Lobular_and_Mid_Cingulate
8) Premotor
9) Posterior_Opercular

Next are three auditory regions:
10) Early_Auditory
11) Auditory_Association
12) Insular_and_Frontal_Opercular
Two regions covering the rest of the temporal cortex:
13) Medial_Temporal
14) Lateral_Temporal

Four regions covering the rest of the posterior cortex:
15) Temporo-Parieto-Occipital_Junction
16) Superior_Parietal
17) Inferior_Parietal
18) Posterior_Cingulate

The final four regions cover the rest of anterior cortex:
19) Anterior_Cingulate_and_Medial_Prefrontal
20) Orbital_and_Polar_Frontal
21) Inferior_Frontal
22) Dorsolateral_Prefrontal

180 ROIs belongs to 22 Cortices ("Regions"):
1 5 3 2 2 2 4 6 6 8 8 8 3 18 18 3 3 4 3 5 5 4 5 10 15 22 18 15 16 18 18 18 18 18 7 7 7 7 7 7 16 7 7 16 16 16 16 16 6 6 6 8 7 8 19 19 19 19 19 19 19 19 20 22 22 19 22 22 20 22 21 21 21 21 8 21 21 21 21 22 22 22 22 22 19 20 20 20 20 20 20 16 8 22 22 9 9 9 9 10 10 10 12 11 12 12 12 12 12 9 12 12 17 16 13 13 13 18 13 11 10 11 13 13 11 11 11 14 14 14 14 13 14 14 5 15 15 15 18 17 17 17 17 17 17 17 17 17 3 4 4 13 5 5 5 5 4 18 18 4 19 19 19 12 12 12 20 21 14 10 10 11 11 14 12 19 19
"""

if args.space == "full_fsaverage":
    save_name = "full_fsaverage"
    save_name = add_beta_to_name(save_name, args.beta)
    print("loading full fsaverage space...")

    for subject_name in SUBJECT_NAMES:
        subject_id = SUBJECT_NAMES.index(subject_name)

        def __load(name):
            lh_path = os.path.join(
                args.nsd_dir,
                "nsddata",
                "freesurfer",
                "fsaverage",
                "label",
                f"lh.{name}.mgz",
            )
            lh = nib.load(lh_path).get_fdata()
            rh_path = os.path.join(
                args.nsd_dir,
                "nsddata",
                "freesurfer",
                "fsaverage",
                "label",
                f"rh.{name}.mgz",
            )
            rh = nib.load(rh_path).get_fdata()
            data = np.concatenate((lh, rh), axis=0)
            return data

        # load nsdgeneral
        # /data/natural-scenes-dataset/nsddata/freesurfer/fsaverage/label/lh.nsdgeneral.mgz
        nsdgeneral = __load("nsdgeneral")
        nsdgeneral = np.where(nsdgeneral == 1)[0]

        # load hcpmmp1
        # /data/natural-scenes-dataset/nsddata/freesurfer/fsaverage/label/lh.HCP_MMP1.mgz
        hcpmmp1 = __load("HCP_MMP1")
        hcpmmp22_names = [
            "Primary_Visual",
            "Early_Visual",
            "Dorsal_Stream_Visual",
            "Ventral_Stream_Visual",
            "MT+_Complex_and_Neighboring_Visual_Areas",
            "Somatosensory_and_Motor",
            "Paracentral_Lobular_and_Mid_Cingulate",
            "Premotor",
            "Posterior_Opercular",
            "Early_Auditory",
            "Auditory_Association",
            "Insular_and_Frontal_Opercular",
            "Medial_Temporal",
            "Lateral_Temporal",
            "Temporo-Parieto-Occipital_Junction",
            "Superior_Parietal",
            "Inferior_Parietal",
            "Posterior_Cingulate",
            "Anterior_Cingulate_and_Medial_Prefrontal",
            "Orbital_and_Polar_Frontal",
            "Inferior_Frontal",
            "Dorsolateral_Prefrontal",
        ]
        hcpmmp180_belong = "1 5 3 2 2 2 4 6 6 8 8 8 3 18 18 3 3 4 3 5 5 4 5 10 15 22 18 15 16 18 18 18 18 18 7 7 7 7 7 7 16 7 7 16 16 16 16 16 6 6 6 8 7 8 19 19 19 19 19 19 19 19 20 22 22 19 22 22 20 22 21 21 21 21 8 21 21 21 21 22 22 22 22 22 19 20 20 20 20 20 20 16 8 22 22 9 9 9 9 10 10 10 12 11 12 12 12 12 12 9 12 12 17 16 13 13 13 18 13 11 10 11 13 13 11 11 11 14 14 14 14 13 14 14 5 15 15 15 18 17 17 17 17 17 17 17 17 17 3 4 4 13 5 5 5 5 4 18 18 4 19 19 19 12 12 12 20 21 14 10 10 11 11 14 12 19 19".split(
            " "
        )
        hcpmmp180_belong = [int(x) - 1 for x in hcpmmp180_belong]
        hcpmmp5_names = ["Visual", "Somatomotor", "Auditory", "Posterior", "Anterior"]
        hcpmmp22_belong = [
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            2,
            3,
            3,
            3,
            3,
            4,
            4,
            4,
            4,
        ]

        save_dir = os.path.join(args.output_dir, subject_name, "roi", save_name)
        os.makedirs(save_dir, exist_ok=True)

        np.save(os.path.join(save_dir, "nsdgeneral.npy"), nsdgeneral)
        
        for _i_22 in range(22):
            _i_180s = np.where(np.array(hcpmmp180_belong) == _i_22)[0]
            _i_180s += 1
            vis = []
            for _i_180 in _i_180s:
                vis += np.where(hcpmmp1 == _i_180)[0].tolist()
                vis += np.where(hcpmmp1 == _i_180 + 180)[0].tolist()
            vis = np.unique(vis)
            vis = np.sort(vis)
            np.save(os.path.join(save_dir, f"{hcpmmp22_names[_i_22]}.npy"), vis)
            
        for _i_5 in range(5):
            _i_22s = np.where(np.array(hcpmmp22_belong) == _i_5)[0]
            vis = []
            for _i_22 in _i_22s:
                vis += np.load(os.path.join(save_dir, f"{hcpmmp22_names[_i_22]}.npy")).tolist()
            vis = np.unique(vis)
            vis = np.sort(vis)
            np.save(os.path.join(save_dir, f"{hcpmmp5_names[_i_5]}.npy"), vis)

        # save voxel coordinates
        import nilearn
        from nilearn import datasets, surface

        fsaverage = nilearn.datasets.fetch_surf_fsaverage("fsaverage7")
        lh_coords, lh_faces = nilearn.surface.load_surf_mesh(fsaverage["sphere_left"])
        rh_coords, rh_faces = nilearn.surface.load_surf_mesh(fsaverage["sphere_right"])
        lh_xmin, lh_xmax = np.min(lh_coords[:, 0]), np.max(lh_coords[:, 0])
        lh_xmax = lh_xmin + (lh_xmax - lh_xmin) * 1.5
        rh_xmin, rh_xmax = np.min(rh_coords[:, 0]), np.max(rh_coords[:, 0])
        if rh_xmin < lh_xmax:
            rh_coords[:, 0] += lh_xmax - rh_xmin
        coords = np.concatenate((lh_coords, rh_coords), axis=0)
        print(f"subject: {subject_name}, coords: {coords.shape}")
        save_dir = os.path.join(args.output_dir, subject_name, "coords", save_name)
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "coords.npy"), coords)

        #############################################
        ### load fmri data ###
        #############################################

        if args.skip_fmri:
            continue

        print(f"loading fmri data for subject: {subject_name}...")

        # /data/natural-scenes-dataset/nsddata_betas/ppdata/subj01/fsaverage/betas_fithrf_GLMdenoise_RR/lh.betas_session01.mgh
        for session_id in tqdm(
            range(1, NUM_SESSIONS[subject_id] + 1 - 3), desc="session"
        ):
            path = os.path.join(
                args.nsd_dir,
                "nsddata_betas",
                "ppdata",
                subject_name,
                "fsaverage",
                beta_version,
                f"lh.betas_session{session_id:02d}.mgh",
            )
            lh_data = nib.load(path).get_fdata()
            path = os.path.join(
                args.nsd_dir,
                "nsddata_betas",
                "ppdata",
                subject_name,
                "fsaverage",
                beta_version,
                f"rh.betas_session{session_id:02d}.mgh",
            )
            rh_data = nib.load(path).get_fdata()
            data = np.concatenate((lh_data, rh_data), axis=0)
            assert data.shape == (
                327684,
                1,
                1,
                750,
            ), f"{subject_name} {session_id} {data.shape}"
            data = data.squeeze()
            # fill nan with 0
            data = np.nan_to_num(data)
            data = data.astype(np.float32)
            data = z_score(data, axis=1)
            data = data.astype(np.float16)

            # assert no nan
            assert np.isnan(data).sum() == 0

            save_dir = os.path.join(args.output_dir, subject_name, "fmri", save_name)
            os.makedirs(save_dir, exist_ok=True)

            for i in range(750):
                save_i = (session_id - 1) * 750 + i
                np.save(os.path.join(save_dir, f"{save_i:06d}.npy"), data[:, i])


#############################################
### load func1mm space, filter by nc  ###
#############################################

if args.space == "func1mm":
    save_name = "func1mm"
    save_name = add_beta_to_name(save_name, args.beta)
    print("loading func1mm space...")

    for subject_name in SUBJECT_NAMES:
        subject_id = SUBJECT_NAMES.index(subject_name)

        # load snr
        # /data/huze/natural-scenes-dataset/nsddata_betas/ppdata/subj01/func1mm/betas_fithrf_GLMdenoise_RR/ncsnr.nii.gz
        path = os.path.join(
            args.nsd_dir,
            "nsddata_betas",
            "ppdata",
            subject_name,
            "func1mm",
            beta_version,
            "ncsnr.nii.gz",
        )
        snr = nib.load(path).get_fdata()
        nc = snr**2 / (snr**2 + 1 / 3)

        # load nsdgeneral mask
        # /data/huze/natural-scenes-dataset/nsddata/ppdata/subj01/func1mm/roi/nsdgeneral.nii.gz
        path = os.path.join(
            args.nsd_dir,
            "nsddata",
            "ppdata",
            subject_name,
            "func1mm",
            "roi",
            "nsdgeneral.nii.gz",
        )
        nsdgeneral = nib.load(path).get_fdata()
        nsdgeneral_mask = nsdgeneral > 0

        # load RSC PPA OPA mask
        # /data/huze/natural-scenes-dataset/nsddata/ppdata/subj01/func1mm/roi/floc-places.nii.gz
        path = os.path.join(
            args.nsd_dir,
            "nsddata",
            "ppdata",
            subject_name,
            "func1mm",
            "roi",
            "floc-places.nii.gz",
        )
        floc_places = nib.load(path).get_fdata()
        floc_places_mask = floc_places > 0

        challenge_mask = nsdgeneral_mask | floc_places_mask

        nc_th = args.nc_threshold
        if nc_th > 0 and nc_th < 1:
            pass
        elif nc_th > 1:
            nc_th = np.percentile(m_nc, nc_th)
        else:
            # find nc_th so added nc mean equals to challenge nc mean
            nc_th = m_nc.mean()
            add_nc_mean = 1.0
            target_nc_mean = m_nc.mean()
            while not np.isclose(add_nc_mean, target_nc_mean, atol=0.001):
                if add_nc_mean > target_nc_mean:
                    nc_th *= 0.5
                else:
                    nc_th *= 1.5
                nc_mask = nc > nc_th
                full_mask = nc_mask | challenge_mask
                new_mask = full_mask & (~challenge_mask)
                add_nc = nc[new_mask == 1]
                add_nc_mean = add_nc.mean()
            print(f"auto nc_th: {nc_th}")

        nc_mask = nc > nc_th

        full_mask = challenge_mask | nc_mask
        new_mask = full_mask & (~challenge_mask)

        # print(f"subject: {subject_name}, nc_mask: {nc_mask.sum()}")
        print(
            f"subject: {subject_name}, challenge_mask: {challenge_mask.sum()}, full_mask: {full_mask.sum()}, new_mask: {new_mask.sum()}"
        )
        print(
            f"NC mean: {nc[challenge_mask].mean()}, NC std: {nc[challenge_mask].std()}"
        )
        for i in range(1, 10):
            print(f"{i}%: {np.percentile(nc[challenge_mask], i * 10):.3f}", end=", ")
        print()

        # save roi voxel indices
        save_dir = os.path.join(args.output_dir, subject_name, "roi", save_name)
        os.makedirs(save_dir, exist_ok=True)
        nsd_general_masked_mask = nsdgeneral_mask.reshape(-1)[
            full_mask.reshape(-1) == 1
        ]
        nsd_general_vi = np.where(nsd_general_masked_mask == 1)[0]
        np.save(os.path.join(save_dir, "nsdgeneral.npy"), nsd_general_vi)

        # save voxel indices
        # voxel indices is indices of voxels in the `full_data`
        save_dir = os.path.join(args.output_dir, subject_name, "data_mask", save_name)
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "mask.npy"), full_mask)
        full_voxel_indices = np.where(full_mask.reshape(-1) == 1)[0]
        np.save(os.path.join(save_dir, "voxel_indices.npy"), full_voxel_indices)

        # save voxel coordinates
        coords = full_mask.nonzero()
        coords = np.stack(coords, axis=1)
        print(f"subject: {subject_name}, coords: {coords.shape}")
        save_dir = os.path.join(args.output_dir, subject_name, "coords", save_name)
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "coords.npy"), coords)

        #############################################
        ### load fmri data ###
        #############################################

        if args.skip_fmri:
            continue

        print(f"loading fmri data for subject: {subject_name}...")

        session_ids = np.arange(1, NUM_SESSIONS[subject_id] + 1 - 3)
        # /data/natural-scenes-dataset/nsddata_betas/ppdata/subj01/func1mm/betas_fithrf_GLMdenoise_RR/betas_session01.nii.gz
        for session_id in tqdm(session_ids, desc="session"):
            path = os.path.join(
                args.nsd_dir,
                "nsddata_betas",
                "ppdata",
                subject_name,
                "func1mm",
                beta_version,
                f"betas_session{session_id:02d}.nii.gz",
            )
            data = nib.load(path)
            data = data.get_fdata()  # this is single-threaded and slow
            assert len(data.shape) == 4
            assert data.shape[3] == 750

            data = data.reshape(-1, 750)
            data = data[full_mask.reshape(-1) == 1]

            # fill nan with 0
            data = np.nan_to_num(data)
            data = data.astype(np.float32)
            data /= 300
            # debug divide 300
            # vmax = np.percentile(data, 95)
            # vmin = np.percentile(data, 5)
            # print(f"subject: {subject_name}, session: {session_id}, vmax: {vmax}, vmin: {vmin}")
            data = z_score(data, axis=1)
            data = data.astype(np.float16)

            # assert no nan
            assert np.isnan(data).sum() == 0

            save_dir = os.path.join(args.output_dir, subject_name, "fmri", save_name)
            os.makedirs(save_dir, exist_ok=True)

            for i in range(750):
                save_i = (session_id - 1) * 750 + i
                np.save(os.path.join(save_dir, f"{save_i:06d}.npy"), data[:, i])


#############################################
### load func1mm space, full  ###
#############################################

if args.space == "full_func1mm":
    save_name = "full_func1mm"
    save_name = add_beta_to_name(save_name, args.beta)
    print("loading func1mm space...")

    for subject_name in SUBJECT_NAMES:
        subject_id = SUBJECT_NAMES.index(subject_name)

        # load nsdgeneral mask
        # /data/huze/natural-scenes-dataset/nsddata/ppdata/subj01/func1mm/roi/nsdgeneral.nii.gz
        path = os.path.join(
            args.nsd_dir,
            "nsddata",
            "ppdata",
            subject_name,
            "func1mm",
            "roi",
            "nsdgeneral.nii.gz",
        )
        nsdgeneral = nib.load(path).get_fdata()
        nsdgeneral_mask = nsdgeneral > 0

        # print(f"subject: {subject_name}, nc_mask: {nc_mask.sum()}")
        print(
            f"subject: {subject_name}, challenge_mask: {challenge_mask.sum()}, full_mask: {full_mask.sum()}, new_mask: {new_mask.sum()}"
        )

        # save roi voxel indices
        save_dir = os.path.join(args.output_dir, subject_name, "roi", save_name)
        os.makedirs(save_dir, exist_ok=True)
        nsd_general_masked_mask = nsdgeneral_mask.reshape(-1)[
            full_mask.reshape(-1) == 1
        ]
        nsd_general_vi = np.where(nsd_general_masked_mask == 1)[0]
        np.save(os.path.join(save_dir, "nsdgeneral.npy"), nsd_general_vi)

        # save voxel indices
        # voxel indices is indices of voxels in the `full_data`
        save_dir = os.path.join(args.output_dir, subject_name, "data_mask", save_name)
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "mask.npy"), full_mask)
        full_voxel_indices = np.where(full_mask.reshape(-1) == 1)[0]
        np.save(os.path.join(save_dir, "voxel_indices.npy"), full_voxel_indices)

        # save voxel coordinates
        coords = full_mask.nonzero()
        coords = np.stack(coords, axis=1)
        print(f"subject: {subject_name}, coords: {coords.shape}")
        save_dir = os.path.join(args.output_dir, subject_name, "coords", save_name)
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "coords.npy"), coords)

        #############################################
        ### load fmri data ###
        #############################################

        if args.skip_fmri:
            continue

        print(f"loading fmri data for subject: {subject_name}...")

        session_ids = np.arange(1, NUM_SESSIONS[subject_id] + 1 - 3)
        # /data/natural-scenes-dataset/nsddata_betas/ppdata/subj01/func1mm/betas_fithrf_GLMdenoise_RR/betas_session01.nii.gz
        for session_id in tqdm(session_ids, desc="session"):
            path = os.path.join(
                args.nsd_dir,
                "nsddata_betas",
                "ppdata",
                subject_name,
                "func1mm",
                beta_version,
                f"betas_session{session_id:02d}.nii.gz",
            )
            data = nib.load(path)
            data = data.get_fdata()  # this is single-threaded and slow
            assert len(data.shape) == 4
            assert data.shape[3] == 750

            data = data.reshape(-1, 750)
            data = data[full_mask.reshape(-1) == 1]

            # fill nan with 0
            data = np.nan_to_num(data)
            data = data.astype(np.float32)
            data /= 300
            # debug divide 300
            # vmax = np.percentile(data, 95)
            # vmin = np.percentile(data, 5)
            # print(f"subject: {subject_name}, session: {session_id}, vmax: {vmax}, vmin: {vmin}")
            data = z_score(data, axis=1)
            data = data.astype(np.float16)

            # assert no nan
            assert np.isnan(data).sum() == 0

            save_dir = os.path.join(args.output_dir, subject_name, "fmri", save_name)
            os.makedirs(save_dir, exist_ok=True)

            for i in range(750):
                save_i = (session_id - 1) * 750 + i
                np.save(os.path.join(save_dir, f"{save_i:06d}.npy"), data[:, i])

#############################################
### load func1mm space, hippocampus mask ###
#############################################

if args.space == "hip_func1mm":
    save_name = "hip_func1mm"
    save_name = add_beta_to_name(save_name, args.beta)
    print("loading func1mm thalamus and MTL...")

    for subject_name in SUBJECT_NAMES:
        subject_id = SUBJECT_NAMES.index(subject_name)

        # /data/natural-scenes-dataset/nsddata/ppdata/subj01/func1mm/roi/thalamus.nii.gz
        path = os.path.join(
            args.nsd_dir,
            "nsddata",
            "ppdata",
            subject_name,
            "func1mm",
            "roi",
            "thalamus.nii.gz",
        )
        thalamus = nib.load(path).get_fdata()
        thamalmus_names = ["LGN", "SC", "Vpu", "DLpu", "DMpu"]
        thalamus_mask = thalamus > 0
        # /data/natural-scenes-dataset/nsddata/ppdata/subj01/func1mm/roi/MTL.nii.gz
        path = os.path.join(
            args.nsd_dir,
            "nsddata",
            "ppdata",
            subject_name,
            "func1mm",
            "roi",
            "MTL.nii.gz",
        )
        mtl = nib.load(path).get_fdata()
        mtl_names = [
            "ErC",
            "area35",
            "area36",
            "PhC",
            "Sub",
            "CA1",
            "CA2",
            "CA3",
            "DG",
            "HT",
        ]
        mtl_mask = mtl > 0

        mask = thalamus_mask | mtl_mask

        print(f"subject: {subject_name}, mask: {mask.sum()}")

        full_mask = mask

        # save roi voxel indices
        save_dir = os.path.join(args.output_dir, subject_name, "roi", save_name)
        os.makedirs(save_dir, exist_ok=True)
        mtl[mtl != 0] += int(thalamus.max())
        ar = mtl + thalamus
        masked_all_roi = ar.reshape(-1)[full_mask.reshape(-1) == 1]
        names = thamalmus_names + mtl_names
        for _i_roi, name in enumerate(names):
            _i_roi += 1
            vi = np.where(masked_all_roi == _i_roi)[0]
            np.save(os.path.join(save_dir, f"{name}.npy"), vi)

        # merge
        hhhb = ["Sub", "CA1", "CA2", "CA3", "DG"]
        vis = []
        for name in hhhb:
            vi = np.load(os.path.join(save_dir, f"{name}.npy"))
            vis.append(vi)
        vi = np.concatenate(vis)
        np.save(os.path.join(save_dir, f"HHHB.npy"), vi)

        ht = np.load(os.path.join(save_dir, f"HT.npy"))
        vi = np.concatenate([vi, ht])
        np.save(os.path.join(save_dir, f"H.npy"), vi)

        PrC = ["area35", "area36"]
        vis = []
        for name in PrC:
            vi = np.load(os.path.join(save_dir, f"{name}.npy"))
            vis.append(vi)
        vi = np.concatenate(vis)
        np.save(os.path.join(save_dir, f"PrC.npy"), vi)

        pulvinar = ["Vpu", "DLpu", "DMpu"]
        vis = []
        for name in pulvinar:
            vi = np.load(os.path.join(save_dir, f"{name}.npy"))
            vis.append(vi)
        vi = np.concatenate(vis)
        np.save(os.path.join(save_dir, f"pulvinar.npy"), vi)

        # save voxel indices
        # voxel indices is indices of voxels in the `full_data`
        save_dir = os.path.join(args.output_dir, subject_name, "data_mask", save_name)
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "mask.npy"), full_mask)
        full_voxel_indices = np.where(full_mask.reshape(-1) == 1)[0]
        np.save(os.path.join(save_dir, "voxel_indices.npy"), full_voxel_indices)

        # save voxel coordinates
        coords = full_mask.nonzero()
        coords = np.stack(coords, axis=1)
        print(f"subject: {subject_name}, coords: {coords.shape}")
        save_dir = os.path.join(args.output_dir, subject_name, "coords", save_name)
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "coords.npy"), coords)

        #############################################
        ### load fmri data ###
        #############################################

        if args.skip_fmri:
            continue

        print(f"loading fmri data for subject: {subject_name}...")

        session_ids = np.arange(1, NUM_SESSIONS[subject_id] + 1 - 3)
        # /data/natural-scenes-dataset/nsddata_betas/ppdata/subj01/func1mm/betas_fithrf_GLMdenoise_RR/betas_session01.nii.gz
        for session_id in tqdm(session_ids, desc="session"):
            path = os.path.join(
                args.nsd_dir,
                "nsddata_betas",
                "ppdata",
                subject_name,
                "func1mm",
                beta_version,
                f"betas_session{session_id:02d}.nii.gz",
            )
            data = nib.load(path)
            data = data.get_fdata()  # this is single-threaded and slow
            assert len(data.shape) == 4
            assert data.shape[3] == 750

            data = data.reshape(-1, 750)
            data = data[full_mask.reshape(-1) == 1]

            # fill nan with 0
            data = np.nan_to_num(data)
            data = data.astype(np.float32)
            data /= 300
            # debug divide 300
            # vmax = np.percentile(data, 95)
            # vmin = np.percentile(data, 5)
            # print(f"subject: {subject_name}, session: {session_id}, vmax: {vmax}, vmin: {vmin}")
            data = z_score(data, axis=1)
            data = data.astype(np.float16)

            # assert no nan
            assert np.isnan(data).sum() == 0

            save_dir = os.path.join(args.output_dir, subject_name, "fmri", save_name)
            os.makedirs(save_dir, exist_ok=True)

            for i in range(750):
                save_i = (session_id - 1) * 750 + i
                np.save(os.path.join(save_dir, f"{save_i:06d}.npy"), data[:, i])


if args.space == "fship":
    save_name = "fship"
    save_name = add_beta_to_name(save_name, args.beta)
    print("merging fsaverage and func1mm thalamus and MTL...")

    for subject_name in SUBJECT_NAMES:
        subject_id = SUBJECT_NAMES.index(subject_name)

        # save voxel indices
        save_dir = os.path.join(args.output_dir, subject_name, "data_mask", save_name)
        os.makedirs(save_dir, exist_ok=True)
        n_full_fsaverage = 327684
        n_hip_func1mm = np.load(
            os.path.join(
                args.output_dir, subject_name, "data_mask", "hip_func1mm", "mask.npy"
            )
        ).sum()
        print(
            f"subject: {subject_name}, n_full_fsaverage: {n_full_fsaverage}, n_hip_func1mm: {n_hip_func1mm}"
        )
        tup = [n_full_fsaverage, n_hip_func1mm]
        np.savetxt(os.path.join(save_dir, f"{tup[0]}-{tup[1]}.txt"), tup, fmt="%d")

        # save roi voxel indices
        save_dir = os.path.join(args.output_dir, subject_name, "roi", save_name)
        os.makedirs(save_dir, exist_ok=True)
        fs_load_dir = os.path.join(
            args.output_dir, subject_name, "roi", "full_fsaverage"
        )
        roi_files = os.listdir(fs_load_dir)
        # copy full_fsaverage roi files
        for roi_file in roi_files:
            shutil.copyfile(
                os.path.join(fs_load_dir, roi_file), os.path.join(save_dir, roi_file)
            )
        hip_load_dir = os.path.join(args.output_dir, subject_name, "roi", "hip_func1mm")
        roi_files = os.listdir(hip_load_dir)
        # load and plus hip_func1mm roi files
        for roi_file in roi_files:
            hip_roi = np.load(os.path.join(hip_load_dir, roi_file))
            hip_roi += n_full_fsaverage
            np.save(os.path.join(save_dir, roi_file), hip_roi)

        # save voxel coordinates
        fs_coords = np.load(
            os.path.join(
                args.output_dir, subject_name, "coords", "full_fsaverage", "coords.npy"
            )
        )
        hip_coords = np.load(
            os.path.join(
                args.output_dir, subject_name, "coords", "hip_func1mm", "coords.npy"
            )
        )
        fs_x_max = fs_coords[:, 0].max()
        hip_x_min = hip_coords[:, 0].min()
        hip_coords[:, 0] += int(fs_x_max) - hip_x_min + 1
        save_dir = os.path.join(args.output_dir, subject_name, "coords", save_name)
        os.makedirs(save_dir, exist_ok=True)
        np.save(
            os.path.join(save_dir, "coords.npy"),
            np.concatenate([fs_coords, hip_coords], axis=0),
        )

        #############################################
        ### load fmri data ###
        #############################################

        if args.skip_fmri:
            continue

        fs_load_dir = os.path.join(
            args.output_dir,
            subject_name,
            "fmri",
            add_beta_to_name("full_fsaverage", args.beta),
        )
        hip_load_dir = os.path.join(
            args.output_dir,
            subject_name,
            "fmri",
            add_beta_to_name("hip_func1mm", args.beta),
        )
        files = os.listdir(fs_load_dir)
        save_dir = os.path.join(args.output_dir, subject_name, "fmri", save_name)
        os.makedirs(save_dir, exist_ok=True)
        # load and concatenate full_fsaverage with hip_func1mm
        for file in tqdm(files):
            fs_data = np.load(os.path.join(fs_load_dir, file))
            hip_data = np.load(os.path.join(hip_load_dir, file))
            data = np.concatenate([fs_data, hip_data], axis=0)
            data = data.astype(np.float16)
            np.save(os.path.join(save_dir, file), data)
