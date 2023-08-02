# %%
import os

from abc import ABC, abstractmethod
import random

import numpy as np
import torch
from torch import Tensor
from typing import Any, Tuple, List, Dict, Union
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from PIL import Image
from torchvision import transforms

import logging

from config import AutoConfig


class NSDDataset(Dataset):
    def __init__(
        self,
        root="/data/ALG23",
        subject_name="subj01",
        split="train",  # train, val1, val2, predict
        image_resolution=(224, 224),
        fmri_space="fsaverage",
        rois: List[str] = ["all"],  # list of rois to use
        load_prev_frames: bool = True,
        dark_postfix: str = None,
        filter_by_session: str = "all",
        n_prev_frames=24,
        cfg: AutoConfig = None,
    ):
        super().__init__()
        self.root = root
        self.subject_name = subject_name
        self.split = split
        self.image_resolution = image_resolution
        self.fmri_space = fmri_space
        self.rois = rois
        self.load_prev_frames = load_prev_frames
        self.dark_postfix = dark_postfix
        self.filter_by_session = filter_by_session
        self.n_prev_frames = n_prev_frames
        self.cfg = cfg

        self.image_with_memory = None
        # (40 x 750) x 32, 73k image id, -1 for blank, 0-based
        self.y_idxs = np.arange(0, int(40 * 750))
        self.behavior_data = None
        # (sess x 750) x 4, 4 is number of features
        self.neuron_coords = None
        # N x 3, 3 is x, y, z
        self.voxel_indices = ...
        # N, indices of voxels to use
        self.roi_dict = {}
        # dict of roi_name: roi_voxel_indices

        self._load_meta_data()

        if self.cfg.EXPERIMENTAL.ANOTHER_SPLIT:
            self._another_split_data()
        elif self.cfg.EXPERIMENTAL.NO_SPLIT:
            self._no_split_data()
        else:
            self._split_data()  # split data into train/val1/val2/predict

        self.transform = transforms.Compose(
            [
                transforms.Resize(self.image_resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def _load_meta_data(self):
        subject_dir = os.path.join(self.root, self.subject_name)
        self.image_with_memory = np.load(
            os.path.join(subject_dir, "image_with_memory.npy")
        )
        n_session, n_sti, n_memory = self.image_with_memory.shape
        self.image_with_memory = self.image_with_memory.reshape(-1, n_memory)

        if self.cfg.EXPERIMENTAL.SHUFFLE_IMAGES:
            random.shuffle(self.image_with_memory)

        self.behavior_data = np.load(os.path.join(subject_dir, "behavior_data.npy"))
        self.neuron_coords = np.load(
            os.path.join(subject_dir, "coords", self.fmri_space, "coords.npy")
        )
        self.neuron_coords = torch.from_numpy(self.neuron_coords).float()

        roi_dir = os.path.join(subject_dir, "roi", self.fmri_space)
        if os.path.exists(roi_dir) is False:
            self.roi_dict = {"all": ...}
            self.voxel_indices = ...
        else:
            available_rois = {}
            for roi_file in os.listdir(roi_dir):
                if roi_file.endswith(".npy"):
                    roi_name = roi_file.split(".")[0]
                    available_rois[roi_name] = np.load(os.path.join(roi_dir, roi_file))

            black_list_roi_names = []
            black_list_roi_names += [f'w_{i}' for i in range(1, 10)]
            for j in range(1, 11):
                black_list_roi_names += [f'r_{j}_{i}' for i in range(1, 10)]
            
            if self.rois == ["all"]:
                self.voxel_indices = ...
                for roi_name, roi_data in available_rois.items():
                    # candidate_roi_names = (
                    #     ["RSC", "E", "MV", "ML", "MP", "V", "L", "P"]
                    #     + ["R"]
                    #     + ["added", "orig"]
                    #     + ["LGN", "SC", "Vpu", "DLpu", "DMpu"]
                    #     + [
                    #         "ErC",
                    #         "area35",
                    #         "area36",
                    #         "PhC",
                    #         "Sub",
                    #         "CA1",
                    #         "CA2",
                    #         "CA3",
                    #         "DG",
                    #         "HT",
                    #     ]
                    #     + ["HHHB", "Hip"]
                    #     + ["PrC"]
                    #     + ["H"]
                    #     + ["pulvinar"]
                    #     + [
                    #         "Primary_Visual",
                    #         "Early_Visual",
                    #         "Dorsal_Stream_Visual",
                    #         "Ventral_Stream_Visual",
                    #         "MT+_Complex_and_Neighboring_Visual_Areas",
                    #         "Somatosensory_and_Motor",
                    #         "Paracentral_Lobular_and_Mid_Cingulate",
                    #         "Premotor",
                    #         "Posterior_Opercular",
                    #         "Early_Auditory",
                    #         "Auditory_Association",
                    #         "Insular_and_Frontal_Opercular",
                    #         "Medial_Temporal",
                    #         "Lateral_Temporal",
                    #         "Temporo-Parieto-Occipital_Junction",
                    #         "Superior_Parietal",
                    #         "Inferior_Parietal",
                    #         "Posterior_Cingulate",
                    #         "Anterior_Cingulate_and_Medial_Prefrontal",
                    #         "Orbital_and_Polar_Frontal",
                    #         "Inferior_Frontal",
                    #         "Dorsolateral_Prefrontal",
                    #     ]
                    #     + ["Visual", "Somatomotor", "Auditory", "Posterior", "Anterior"]
                    #     + ["nsdgeneral"]
                    # )
                    # if any(
                    #     [
                    #         candidate_roi_name == roi_name
                    #         for candidate_roi_name in candidate_roi_names
                    #     ]
                    # ):
                    #     self.roi_dict[roi_name] = roi_data
                    if not any(
                        [
                            _name == roi_name
                            for _name in black_list_roi_names
                        ]
                    ):
                        self.roi_dict[roi_name] = roi_data
            elif self.rois == ["orig"]:
                self.voxel_indices = available_rois["orig"]
                for roi_name, roi_data in available_rois.items():
                    candidate_roi_names = [
                        "orig",
                        "RSC",
                        "E",
                        "MV",
                        "ML",
                        "MP",
                        "V",
                        "L",
                        "P",
                    ]
                    if any(
                        [
                            candidate_roi_name in roi_name
                            for candidate_roi_name in candidate_roi_names
                        ]
                    ):
                        self.roi_dict[roi_name] = roi_data
            else:
                self.voxel_indices = []
                count = 0
                for roi_name in self.rois:
                    vi = available_rois[roi_name]
                    self.voxel_indices.append(vi)
                    length = len(vi)
                    self.roi_dict[roi_name] = np.arange(count, count + length)
                    count += length
                self.voxel_indices = np.concatenate(self.voxel_indices, axis=0)
                self.voxel_indices = torch.from_numpy(self.voxel_indices).long()

            self.roi_dict["all"] = ...

        self.neuron_coords = self.neuron_coords[self.voxel_indices]

    def _index_data(self, idxs):
        idxs = np.array(idxs)
        # sorted
        idxs = np.sort(idxs)
        self.image_with_memory = self.image_with_memory[idxs] if len(idxs) > 0 else None
        self.behavior_data = self.behavior_data[idxs] if len(idxs) > 0 else None
        self.y_idxs = self.y_idxs[idxs] if len(idxs) > 0 else None

    def _split_data(self):
        split_idxs_path = os.path.join(
            self.root, self.subject_name, "split", self.split + ".txt"
        )
        split_idxs = np.loadtxt(split_idxs_path, dtype=np.int32)
        if self.filter_by_session != [-1]:
            sess_ids = self.filter_by_session

            def filter_fn(x):
                return (x // 750 + 1) in sess_ids

            split_idxs = list(filter(filter_fn, split_idxs))
        self._index_data(split_idxs)

    def _another_split_data(self):
        idxs = []
        for stage in ["train", "val1", "val2"]:
            split_idxs_path = os.path.join(
                self.root, self.subject_name, "split", stage + ".txt"
            )
            split_idxs = np.loadtxt(split_idxs_path, dtype=np.int32)
            idxs.append(split_idxs)
        idxs = np.concatenate(idxs)
        idxs = np.sort(idxs)
        idxs = idxs.reshape(-1, 750)
        if self.filter_by_session != [-1]:
            sess_ids = self.filter_by_session
            idxs = idxs[sess_ids]
        # sess_idxs = np.arange(0, 750)
        # np.random.shuffle(sess_idxs)
        # train_idxs = idxs[:, sess_idxs[:500]].reshape(-1)
        # val1_idxs = idxs[:, sess_idxs[500:625]].reshape(-1)
        # val2_idxs = idxs[:, sess_idxs[625:]].reshape(-1)

        # train val1 val2 500:125:125
        idxs = idxs.reshape(-1)
        train_idxs = np.concatenate([idxs[::6], idxs[1::6], idxs[3::6], idxs[4::6]])
        val1_idxs = np.concatenate([idxs[2::6]])
        val2_idxs = np.concatenate([idxs[5::6]])

        if self.split == "train":
            self._index_data(train_idxs)
        elif self.split == "val1":
            self._index_data(val1_idxs)
        elif self.split == "val2":
            self._index_data(val2_idxs)

    def _no_split_data(self):
        idxs = []
        for stage in ["train", "val1", "val2"]:
            split_idxs_path = os.path.join(
                self.root, self.subject_name, "split", stage + ".txt"
            )
            split_idxs = np.loadtxt(split_idxs_path, dtype=np.int32)
            idxs.append(split_idxs)
        idxs = np.concatenate(idxs)
        split_idxs = idxs
        if self.filter_by_session != [-1]:
            sess_ids = self.filter_by_session

            def filter_fn(x):
                return (x // 750 + 1) in sess_ids

            split_idxs = list(filter(filter_fn, split_idxs))
        self._index_data(split_idxs)

    def __len__(self):
        return len(self.image_with_memory)

    @property
    def num_voxels(self):
        return len(self.neuron_coords)

    def __getitem__(self, i):
        img, prev_img, prev_feats = self.get_images(i)
        y = self.get_y(i) if self.split != "predict" else None
        dark = self.get_dark(i) if self.split != "predict" else None
        bhv, prev_bhvs = self.get_behavior(i)
        ssid = self.get_sessionid(i)
        subject_name = self.subject_name
        data_idx = i
        return (
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
        )

    @staticmethod
    def collate_fn(batch):
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
        ) = zip(*batch)

        img = torch.stack(img, dim=0)
        prev_img = torch.stack(prev_img, dim=0) if prev_img[0] is not None else None
        prev_feats = (
            torch.stack(prev_feats, dim=0) if prev_feats[0] is not None else None
        )
        y = y  # leave as list
        dark = dark
        bhv = torch.stack(bhv, dim=0)
        prev_bhvs = torch.stack(prev_bhvs, dim=0)
        ssid = np.array(ssid)
        subject_name = np.array(subject_name)
        data_idx = np.array(data_idx)
        return (
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
        )

    def get_images(self, i) -> Tensor:
        if self.load_prev_frames:
            prev_img = self._load_image(self.image_with_memory[i, -2])
            prev_feats = []
            for _i_pv in range(1, self.n_prev_frames):
                _i_pv = -_i_pv - 2
                prev_feats.append(self._load_feat(self.image_with_memory[i, _i_pv]))
            prev_feats = torch.stack(prev_feats, dim=0)
        else:
            prev_img, prev_feats = None, None
        _i_t = self.cfg.EXPERIMENTAL.T_IMAGE - 1  # default to 0
        img = self._load_image(self.image_with_memory[i, _i_t])
        return img, prev_img, prev_feats

    def _load_image(self, idx_73k):
        if idx_73k == -1:  # blank image
            return torch.ones(3, *self.image_resolution, dtype=torch.float32) * 0.1
        if self.cfg.EXPERIMENTAL.BLANK_IMAGE:
            return torch.ones(3, *self.image_resolution, dtype=torch.float32) * 0.1

        path = os.path.join(self.root, "images", f"{idx_73k:05d}.jpeg")
        img = Image.open(path)
        img = img.convert("RGB")
        img = self.transform(img)
        return img

    def _load_feat(self, idx_73k):
        if idx_73k == -1:
            return torch.zeros(1024, dtype=torch.float32)

        path = os.path.join(self.root, "feats", f"{idx_73k:05d}.npy")
        feat = np.load(path).astype(np.float32)
        feat = torch.from_numpy(feat)
        return feat

    def get_y(self, i) -> Tensor:
        i = self.y_idxs[i]
        fmri_path = os.path.join(
            self.root, self.subject_name, "fmri", self.fmri_space, f"{i:06d}.npy"
        )
        data = self._load_y(fmri_path)
        return data

    def get_dark(self, i) -> Tensor:
        if self.dark_postfix is None or len(self.dark_postfix) == 0:
            return None
        i = self.y_idxs[i]
        dark_path = os.path.join(
            self.root,
            self.subject_name,
            "dark",
            self.fmri_space,
            f"{i:06d}.{self.dark_postfix}.npy",
        )
        data = self._load_y(dark_path)
        return data

    def _load_y(self, path):
        if not os.path.exists(path):
            return torch.zeros(self.num_voxels, dtype=torch.float32)
        data = np.load(path).astype(np.float32)
        data = torch.from_numpy(data)
        data = data.flatten()
        data = data[self.voxel_indices]
        return data

    def get_behavior(self, i):
        bhv = self.behavior_data[i]

        prev_bhvs = []
        zero_counter = 0
        for _i_pv in range(self.n_prev_frames):
            if self.image_with_memory[i, -_i_pv - 2] == -1:  # no memory image 1
                # fill with last value
                prev_bhvs.append(self.behavior_data[-1])
                zero_counter += 1
            else:
                prev_bhv = self.behavior_data[i - _i_pv - 1 + zero_counter]
                prev_bhvs.append(prev_bhv)
        prev_bhvs = np.stack(prev_bhvs, axis=0)

        if self.cfg.EXPERIMENTAL.BEHV_SELECTION != [-1]:
            _hbv_idx = self.cfg.EXPERIMENTAL.BEHV_SELECTION
            _hbv_idx = np.array(_hbv_idx)
            bhv = bhv[_hbv_idx]
            prev_bhvs = prev_bhvs[:, _hbv_idx]

        bhv = torch.from_numpy(bhv.astype(np.float32))
        prev_bhvs = torch.from_numpy(prev_bhvs.astype(np.float32))
        return bhv, prev_bhvs

    def get_sessionid(self, i):
        return i // 750 + 1

    def save_dark(self, outs, name):
        outs = outs.cpu().numpy().astype(np.float16)
        os.makedirs(
            os.path.join(self.root, self.subject_name, "dark", self.fmri_space),
            exist_ok=True,
        )
        for i, out in enumerate(outs):
            i = self.y_idxs[i]
            path = os.path.join(
                self.root,
                self.subject_name,
                "dark",
                self.fmri_space,
                f"{i:06d}.{name}.npy",
            )
            np.save(path, out)

    def load_one_dark(self, i, name):
        i = self.y_idxs[i]
        path = os.path.join(
            self.root,
            self.subject_name,
            "dark",
            self.fmri_space,
            f"{i:06d}.{name}.npy",
        )
        return np.load(path).astype(np.float32)


if __name__ == "__main__":
    dataset = NSDDataset()
