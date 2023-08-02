import copy
from cluster_utils import my_nfs_cluster_job, trial_dirname_creator

import argparse
import os
import sys
from random import seed, shuffle

import numpy as np
import ray
from ray import tune

from config_utils import dict_to_list, get_cfg_defaults, load_from_yaml

from train_utils import max_batch_size, simple_train


def get_parser():
    parser = argparse.ArgumentParser(description="Ray Tune")

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="verbose", default=False
    )

    parser.add_argument(
        "-p", "--progress", action="store_true", help="progress", default=False
    )

    parser.add_argument(
        "--rm", action="store_true", default=False, help="Remove all previous results"
    )

    parser.add_argument(
        "--name", type=str, default="debug", help="Name of the experiment"
    )
    return parser

BACKBONEC = {
'clip_vit_l': (224, [5, 11, 17, 23], [1024, 1024, 1024, 1024], [2048, 2048, 2048, 1024]),
'clip_vit_b': (224, [2, 5, 8, 11], [768, 768, 768, 768], [1536, 1536, 1536, 768]),
'clip_vit_s': (224, [2, 5, 8, 11], [768, 768, 768, 768], [1536, 1536, 1536, 768]),
'dinov2_vit_l': (224, [5, 11, 17, 23], [1024, 1024, 1024, 1024], [2048, 2048, 2048, 1024]),
'dinov2_vit_b': (224, [2, 5, 8, 11], [768, 768, 768, 768], [1536, 1536, 1536, 768]),
'dinov2_vit_s': (224, [2, 5, 8, 11], [384, 384, 384, 384], [768, 768, 768, 384]),
}

@my_nfs_cluster_job
def job(tune_dict, cfg, progress=False, **kwargs):
    if "row" in tune_dict:
        global ROW_LIST
        row = tune_dict["row"]
        tune_dict.pop("row")
        print(ROW_LIST[row])
        tune_dict.update(ROW_LIST[row])

    cfg.merge_from_list(dict_to_list(tune_dict))
    
    reso, layers, dim, dim2 = BACKBONEC[cfg.MODEL.BACKBONE.NAME]
    cfg.DATASET.IMAGE_RESOLUTION = [reso, reso]
    cfg.MODEL.BACKBONE.LAYERS = layers
    cfg.MODEL.BACKBONE.FEATURE_DIMS = dim
    cfg.MODEL.BACKBONE.CLS_DIMS = dim2
    
    cfg = max_batch_size(cfg)

    ret = simple_train(
        cfg=cfg,
        progress=progress,
        **kwargs,
    )


def run_ray(
    name, cfg, tune_config, rm=False, progress=False, verbose=False, num_samples=1
):
    cfg = copy.deepcopy(cfg)
    if rm:
        import shutil

        shutil.rmtree(os.path.join(cfg.RESULTS_DIR, name), ignore_errors=True)

    ana = tune.run(
        tune.with_parameters(job, cfg=cfg, progress=progress),
        local_dir=cfg.RESULTS_DIR,
        config=tune_config,
        resources_per_trial={"cpu": 1, "gpu": 1},
        num_samples=num_samples,
        name=name,
        verbose=verbose,
        resume="AUTO+ERRORED",
        trial_dirname_creator=trial_dirname_creator,
    )


# ROW_LIST = [
#     {"EXPERIMENTAL.USE_PREV_IMAGE": False},
#     {"EXPERIMENTAL.USE_PREV_IMAGE": True},
#     {"EXPERIMENTAL.USE_EVEN_PREV_IMAGE": True},
#     {"EXPERIMENTAL.SHUFFLE_IMAGES": True},
# ]
# -
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # cfg = load_from_yaml("/workspace/configs/dino_t1.yaml")

    cfg = get_cfg_defaults()
    cfg.DATAMODULE.BATCH_SIZE = 32
    cfg.TRAINER.ACCUMULATE_GRAD_BATCHES = 2
    cfg.OPTIMIZER.LR = 3e-3
    cfg.OPTIMIZER.NAME = "AdamW"
    cfg.DATASET.ROIS = ["orig"]
    cfg.DATASET.FMRI_SPACE = 'fsaverage'
    cfg.DATASET.SUBJECT_LIST = ["subj01"]
    cfg.MODEL.BACKBONE.NAME = "dinov2_vit_s"
    cfg.TRAINER.CALLBACKS.EARLY_STOP.PATIENCE = 20
    cfg.MODEL.CONV_HEAD.SIMPLE = True

    cfg.TRAINER.PRECISION = 16

    cfg.TRAINER.LIMIT_TRAIN_BATCHES = 0.1
    cfg.TRAINER.LIMIT_VAL_BATCHES = 1.0

    cfg.RESULTS_DIR = "/nfscc/ray_results/prev_image"    
    
    cfg.EXPERIMENTAL.USE_PREV_FRAME = False
    cfg.EXPERIMENTAL.STRAIGHT_FORWARD = True

    cfg.MODEL.BACKBONE.LORA.SCALE = 0.0
    cfg.MODEL.BACKBONE.ADAPTIVE_LN.SCALE = 0.0
    tune_config = {
        "DATASET.SUBJECT_LIST": tune.grid_search([["subj01"], ["subj02"], ["subj03"], ["subj04"], ["subj05"], ["subj06"], ["subj07"], ["subj08"]]),
        "EXPERIMENTAL.T_IMAGE": tune.grid_search([-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -16, -20, -24, -28]),
    }
    name = "image_t_full"
    run_ray(name, cfg, tune_config, args.rm, args.progress, args.verbose, 1)