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
parser = argparse.ArgumentParser(description="prepare data for bold5000")
parser.add_argument("--main_dir", type=str, default="/data/BOLD5000")
parser.add_argument("--image_dir", type=str, default="/data/BOLD5000/images")
parser.add_argument(
    "--roi_mask_dir",
    type=str,
    default="/data/BOLD5000/ds001499-download/derivatives/spm", # see openneuron
)
parser.add_argument("--output_dir", type=str, default="/data/mybold5000")
parser.add_argument("--beta", type=str, default="B", choices=["A", "B", "C", "D"])
parser.add_argument("--val1_ratio", type=float, default=0.1)
parser.add_argument("--val2_ratio", type=float, default=0.1)
parser.add_argument("--image_size", type=int, default=224)
parser.add_argument("--image_overwrite", action="store_true")
parser.add_argument("--jpeg_quality", type=int, default=95)
parser.add_argument("--seed", type=int, default=114514)
parser.add_argument("--n_jobs", type=int, default=0)
parser.add_argument("--skip_fmri", action="store_true")

args = parser.parse_args()

NUM_SUBJECTS = 3
SUBJECT_NAMES = ["CSI%01d" % (i + 1) for i in range(NUM_SUBJECTS)]

NUM_SESSIONS = 15
IMAGES_PER_RUN = 37

#############################################
### resave all images and give them index ###
#############################################

all_imgnames = []
for subject in SUBJECT_NAMES:
    imgnames = os.path.join(args.main_dir, f"{subject}_imgnames.txt")
    imgnames = np.loadtxt(imgnames, dtype=str)
    all_imgnames.append(imgnames)
all_imgnames = np.concatenate(all_imgnames)
all_imgnames = np.unique(all_imgnames)

def scan_subdir_for_image(subdir):
    """Scan a directory for images, return a list of image paths"""
    img_list = []
    
    def is_img(x):
        return (
            x.endswith(".jpg")
            or x.endswith(".png")
            or x.endswith(".jpeg")
            or x.endswith(".JPG")
            or x.endswith(".PNG")
            or x.endswith(".JPEG")
            or x.endswith(".tif")
            or x.endswith(".tiff")
            or x.endswith(".TIF")
            or x.endswith(".TIFF")
        )

    for root, dirs, files in os.walk(subdir):
        for file in files:
            if is_img(file):
                img_list.append(os.path.join(root, file))
    return img_list

img_fullpath_list = scan_subdir_for_image(os.path.join(args.image_dir))
img_basename_list = [os.path.basename(x) for x in img_fullpath_list]
argsort = np.argsort(img_basename_list)
img_fullpath_list = np.array(img_fullpath_list)[argsort]
img_basename_list = np.array(img_basename_list)[argsort]
img_ids = np.arange(len(img_fullpath_list))

# save all images as jpeg
save_dir = os.path.join(args.output_dir, "images")
os.makedirs(save_dir, exist_ok=True)
for _i_img in tqdm(range(len(img_fullpath_list))):
    image_id, image = img_ids[_i_img], img_fullpath_list[_i_img]
    save_path = os.path.join(args.output_dir, "images", f"{image_id:05d}.jpeg")
    if os.path.exists(save_path) and not args.image_overwrite:
        continue
    image_jpeg = Image.open(image).convert("RGB")
    image_jpeg = image_jpeg.resize((args.image_size, args.image_size))
    image_jpeg.save(save_path, "JPEG", quality=args.jpeg_quality)

#############################################
### load experiment design ###
#############################################

img_basename_list = img_basename_list
subject_image_with_memory = {}

for subject in SUBJECT_NAMES:
    imgnames = os.path.join(args.main_dir, f'{subject}_imgnames.txt')
    imgnames = np.loadtxt(imgnames, dtype=str)

    img_ids = []
    for imgname in imgnames:
        img_id = np.where(img_basename_list == imgname)[0][0]
        img_ids.append(img_id)
    img_ids = np.array(img_ids)

    # divide into len=37 chunks, each chunk is a run
    img_ids = img_ids.reshape(-1, IMAGES_PER_RUN)
    # pad forward with -1, -1 for blank images
    memory_size = 33
    pad = np.ones((img_ids.shape[0], memory_size), dtype=int) * -1
    padded_img_ids = np.concatenate([pad, img_ids], axis=1)

    image_with_memory = np.zeros((img_ids.shape[0], IMAGES_PER_RUN, memory_size), dtype=int)
    for _i_run in range(img_ids.shape[0]):
        for _i_img in range(img_ids.shape[1]):
            sp = copy.deepcopy(padded_img_ids[_i_run, _i_img + 1:_i_img + memory_size + 1])
            image_with_memory[_i_run, _i_img] = sp
    subject_image_with_memory[subject] = image_with_memory

    # save
    save_dir = os.path.join(args.output_dir, subject)
    os.makedirs(save_dir, exist_ok=True)
    np.save(
        os.path.join(save_dir, "image_with_memory.npy"), image_with_memory
    )  # n_run x 37 x 33, 0-based, -1 for blank


#############################################
### make split and save ###
#############################################

for subject in SUBJECT_NAMES:
    training_image_ids = image_with_memory[:, :, 0].flatten()
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

    def save_list_to_file(list, file):
        with open(file, "w") as f:
            for item in list:
                f.write("%s\n" % item)

    save_dir = os.path.join(args.output_dir, subject, "split")
    os.makedirs(save_dir, exist_ok=True)
    save_list_to_file(train_idxs, os.path.join(save_dir, "train.txt"))
    save_list_to_file(val1_idxs, os.path.join(save_dir, "val1.txt"))
    save_list_to_file(val2_idxs, os.path.join(save_dir, "val2.txt"))
    save_list_to_file(val2_idxs, os.path.join(save_dir, "predict.txt"))


#############################################
### load behavior data (dummy zeros) ###
#############################################

for subject in SUBJECT_NAMES:
    n_data = len(subject_image_with_memory[subject].flatten())
    response = np.zeros((n_data, 35), dtype=np.float32)
    # save
    path = os.path.join(args.output_dir, subject, "behavior_data.npy")
    np.save(path, response)
    


#############################################
### load fMRI, only visual brain ###
#############################################

for subject in SUBJECT_NAMES:
    # load brainmask
    brainmask_dir = os.path.join(args.roi_mask_dir, f"sub-{subject}")
    brainmask_files = []
    for root, dirs, files in os.walk(brainmask_dir):
        for file in files:
            if file.startswith("sub-") and file.endswith(".nii.gz"):
                brainmask_files.append(os.path.join(root, file))

    all_brainmasks = []
    for i, f in enumerate(brainmask_files):
        data = nib.load(f)
        data = data.get_fdata()
        all_brainmasks.append(data)
        
    all_brainmasks = np.stack(all_brainmasks, axis=-1)

    mask = all_brainmasks.sum(-1) > 0

    x, y, z = np.nonzero(mask)
    coords = np.stack([x, y, z], axis=-1)
            
    from einops import rearrange
    mask = rearrange(mask, 'x y z -> (x y z)')
    mask_indices = np.where(mask)[0]

    space = f'visual_{args.beta}'

    # save roi voxel_indices
    roi_dir = os.path.join(args.output_dir, subject, "roi", space)
    os.makedirs(roi_dir, exist_ok=True)
    pass

    # save voxel coordinates
    save_dir = os.path.join(args.output_dir, subject, "coords", space)
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "coords.npy"), coords)

    # find corresponding session
    beta = args.beta
    if beta == 'A':
        b = "TYPEA-ASSUMEHRF"
    if beta == 'B':
        b = "TYPEB-FITHRF"
    if beta == 'C':
        b = 'TYPEC-FITHRF-GLMDENOISE'
    if beta == 'D':
        b = 'TYPED-FITHRF-GLMDENOISE-RR'
        
    pfx = f"{subject}_GLMbetas-{b}_ses-"
    fmri_files = []
    for root, dirs, files in os.walk(args.main_dir):
        for file in files:
            if file.startswith(pfx) and file.endswith(".nii.gz"):
                fmri_files.append(os.path.join(root, file))
    fmri_files.sort()

    def z_score(data, axis=1):
        data = data - data.mean(axis=axis, keepdims=True)
        data = data / (data.std(axis=axis, keepdims=True) + 1e-8)
        return data

    total = 0
    for i, f in tqdm(enumerate(fmri_files), total=len(fmri_files)):
        fmri = nib.load(f)
        fmri = fmri.get_fdata()
        fmri = rearrange(fmri, 'x y z t -> (x y z) t')
        t = fmri.shape[-1]
        masked_fmri = fmri[mask_indices, :]
        masked_fmri = z_score(masked_fmri)

        save_dir = os.path.join(args.output_dir, subject, "fmri", space)
        os.makedirs(save_dir, exist_ok=True)
        for _i in range(t):
            save_path = os.path.join(args.output_dir, subject, "fmri", space, f"{total:06d}.npy")
            np.save(save_path, masked_fmri[:, _i])
            total += 1