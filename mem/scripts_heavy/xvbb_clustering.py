import os
from matplotlib import pyplot as plt
import numpy as np
import torch
from dark_onemodel import build_dmt, get_outs

import argparse

from datasets import NSDDataset

parser = argparse.ArgumentParser()
parser.add_argument(
    "--weight",
    type=str,
    default="/nfscc/alg23/xvba/roiall_bsz/tb314c_00000_TRAINER.ACCUMULATE_GRAD_BATCHES=1/soup.pth",
)
parser.add_argument("--data_dir", type=str, default="/data/ALG23/")
parser.add_argument("--alg_dir", type=str, default="/nfscc/algonauts2023/")
parser.add_argument("--seed", type=int, default=45510)
parser.add_argument("--space", type=str, default="fsaverage")
parser.add_argument("--roi_name", type=str, default="w")
parser.add_argument("--roi_size", type=int, default=9)
parser.add_argument("--save_name", type=str, default="xvbaa")
args = parser.parse_args()

weight = torch.load(args.weight, map_location=torch.device("cpu"))

# model.voxel_outs_weight.subj07.weight
subjects = [
    "subj01",
    "subj02",
    "subj03",
    "subj04",
    "subj05",
    "subj06",
    "subj07",
    "subj08",
]
lengths = []
w = []
for subject in subjects:
    for k, v in weight.items():
        p = f"model.voxel_outs_weight.{subject}.weight"
        if p in k:
            w.append(v)
            lengths.append(v.shape[0])
            break
w = torch.concatenate(w, dim=0)

# seed everything
seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed_all(seed)

def gpu_kmeans_cluster(voxel_outs, n_clusters=100):
    from fast_pytorch_kmeans import KMeans

    kmeans = KMeans(
        n_clusters=n_clusters, verbose=True, mode="cosine", max_iter=1000, tol=1e-6
    )
    labels = kmeans.fit_predict(voxel_outs)
    return kmeans, labels


from torchmetrics.functional import (
    pairwise_cosine_similarity,
    pairwise_euclidean_distance,
)

k = 1000
K = k
km, labels = gpu_kmeans_cluster(w.to("cuda"), n_clusters=k)
c = km.centroids
labels = labels.cpu().numpy()
km_labels = labels

# d = torch.cdist(c, c)
d = pairwise_cosine_similarity(c)
d[torch.isnan(d)] = 0
# d = pairwise_euclidean_distance(c)
d = d.cpu().numpy()

import scipy.cluster.hierarchy as shc

Z = shc.linkage(d, method="ward", optimal_ordering=False)

roi_num_dict = {}
max_dist = 20
hroi_prefix = args.roi_name
target_num_rois = args.roi_size
num_rois = 0
while num_rois != target_num_rois:
    dn_labels = shc.fcluster(Z, max_dist, criterion="distance")
    num_rois = len(np.unique(dn_labels))
    if num_rois > target_num_rois:
        max_dist *= 1.5
    elif num_rois < target_num_rois:
        max_dist *= 0.5
    else:
        pass
    print(f"max_dist: {max_dist} num_rois: {num_rois}")


# continue or exit
vi_dict = {}
kvi_dict = {}
for i in np.unique(dn_labels):
    cluster_voxel_indices = []
    labels = (dn_labels == i).nonzero()[0]
    for l in labels:
        voxel_indices = (km_labels == l).nonzero()[0]
        cluster_voxel_indices.append(voxel_indices)
    cluster_voxel_indices = np.concatenate(cluster_voxel_indices)
    cluster_voxel_indices.sort()
    cluster_voxel_indices = cluster_voxel_indices
    vi_dict[i] = cluster_voxel_indices
    kvi_dict[i] = labels
    print(f"cluster {i}", cluster_voxel_indices.shape)
    
inp = input("Continue? [y/n]")
if inp != "y":
    exit()




import seaborn as sns
g = sns.clustermap(
    # c.cpu().numpy(),
    d,
    figsize=(16, 16),
    method="ward",
    # method="weighted",
    # metric="euclidean",
    # metric="cosine",
    yticklabels=0,
    xticklabels=0,
    row_cluster=True,
    col_cluster=True,
    # cmap="coolwarm",
)
# xtick_labels = [int(tick.get_text()) for tick in g.ax_heatmap.get_xticklabels()]
# g.ax_heatmap.tick_params(axis="y", labelleft=True, labelright=True)
# print(xtick_labels)
# unique, counts = np.unique(km_labels, return_counts=True)
# reordered_counts = counts[xtick_labels]
# print(reordered_counts)

# disable dendrogram
g.ax_row_dendrogram.set_visible(False)
g.ax_col_dendrogram.set_visible(False)
# disable colorbar
g.cax.set_visible(False)

# mat_path = os.path.join(f"/tmp/mat.pdf")
# plt.savefig(mat_path, bbox_inches="tight", pad_inches=0)
mat_path = os.path.join(f"/tmp/mat.png")
plt.savefig(mat_path, bbox_inches="tight", pad_inches=0, dpi=300)
# plt.show()
plt.close()


fig = plt.figure(figsize=(16, 4))
# plt.title("Hierarchical Clustering Dendrogram")
# plt.xlabel("Data points")
# plt.ylabel("Distance")

# plot dendrogram based on clustering results
shc.dendrogram(
    Z,
    labels=dn_labels,
    color_threshold=max_dist,
    # truncate_mode="level",
    # p=1,
    # show_leaf_counts=True,
    # leaf_rotation=90,
    # leaf_font_size=10,
    # show_contracted=False,
    # link_color_func=lambda x: link_cols[x],
    above_threshold_color="k",
    # distance_sort="descending",
    ax=plt.gca(),
)
plt.axhline(max_dist, color="grey", linestyle="--", linewidth=2)
# for i, s in enumerate(labels_str):
#     plt.text(
#         0.8,
#         0.95 - i * 0.04,
#         s,
#         transform=plt.gca().transAxes,
#         va="top",
#         color=cluster_colors[i],
#     )

fig.patch.set_facecolor("white")

dn_labels = shc.fcluster(Z, max_dist, criterion="distance")

# display the dendrogram
# plt.title("Dendrogram")
# plt.ylabel("Distance")
plt.axis("off")
dend_path = "/tmp/dendrogram.png"
plt.savefig(dend_path, bbox_inches="tight", pad_inches=0, dpi=300)
# plt.show()
plt.close()
print(len(np.unique(dn_labels)))
roi_num_dict[hroi_prefix] = len(np.unique(dn_labels))

# %%
# put mat and dendrogram together
import cv2
# fig, axs = plt.subplots(2, 1, figsize=(16, 18))
# axs = axs.flatten()
big_im = np.zeros((2500, 2000, 4), dtype=np.float32)
im = plt.imread(dend_path)
# trim blank space
im = im[im.sum(axis=1).sum(axis=1) != 255 * 3]
print(im.shape)
# resize
new_width = 2000
new_height = int(im.shape[0] * new_width / im.shape[1])
im = cv2.resize(im, (new_width, new_height))
print(im.shape)
big_im[:im.shape[0], :, :] = im
# axs[0].imshow(im)
# axs[0].axis("off")
im = plt.imread(mat_path)
# trim blank space
im = im[im.sum(axis=1).sum(axis=1) != 255 * 3]
# resize
new_width = 2000
new_height = int(im.shape[0] * new_width / im.shape[1])
im = cv2.resize(im, (new_width, new_height))
print(im.shape)
big_im[-im.shape[0]:, :, :] = im
# axs[1].imshow(im)
# axs[1].axis("off")
# remove space between subplots
# plt.subplots_adjust(wspace=0, hspace=0)
fig = plt.figure(figsize=(16, 16))
plt.imshow(big_im.transpose(1, 0, 2))
plt.axis("off")
plt.savefig("/nfscc/fig/veROIdendrogram.jpeg", bbox_inches="tight", pad_inches=0, dpi=96)
# plt.show()
plt.close()



# save
vi_dict = {}
kvi_dict = {}
for i in np.unique(dn_labels):
    cluster_voxel_indices = []
    labels = (dn_labels == i).nonzero()[0]
    for l in labels:
        voxel_indices = (km_labels == l).nonzero()[0]
        cluster_voxel_indices.append(voxel_indices)
    cluster_voxel_indices = np.concatenate(cluster_voxel_indices)
    cluster_voxel_indices.sort()
    cluster_voxel_indices = cluster_voxel_indices
    vi_dict[i] = cluster_voxel_indices
    kvi_dict[i] = labels
    print(f"cluster {i}")
    print(cluster_voxel_indices.shape)
sums = []
start = 0
rois = {}
for subject_id, length in zip(subjects, lengths):
    rois[subject_id] = {}
    end = start + length
    for i_k, vi in vi_dict.items():
        sub_vi = vi[start <= vi]
        sub_vi = sub_vi[sub_vi < end]
        rois[subject_id][i_k] = sub_vi - start
        sums.append(len(sub_vi))
    start += length

mat = np.zeros((len(subjects), len(vi_dict)))
for i in range(len(subjects)):
    for j in range(len(vi_dict)):
        if j + 1 in rois[subjects[i]]:
            mat[i, j] = len(rois[subjects[i]][j + 1])
        else:
            mat[i, j] = 0
fig = plt.figure(figsize=(20, 10))
sns.heatmap(
    mat,
    cmap="Reds",
    xticklabels=list(vi_dict.keys()),
    yticklabels=subjects,
    annot=True,
    fmt=".0f",
    vmax=10000,
)
# plt.show()
plt.savefig("/nfscc/fig/heatmap.png", bbox_inches="tight", pad_inches=0, dpi=300)
plt.close()

# /data/ALG23/subj01/roi/fsaverage/
subject_data_dir = args.data_dir
print(f"saving {hroi_prefix} rois to {subject_data_dir}")
min_length = 0
for subject_id in rois.keys():
    save_dir = os.path.join(subject_data_dir, f"{subject_id}/roi/{args.space}")
    os.makedirs(save_dir, exist_ok=True)
    for i_k in vi_dict.keys():
        path = os.path.join(save_dir, f"{hroi_prefix}_{i_k}.npy")
        if i_k in rois[subject_id]:
            indices = rois[subject_id][i_k]
        else:
            indices = []
            indices = np.array(indices)
        if len(indices) < min_length:
            indices = np.array([])
        np.save(path, indices)