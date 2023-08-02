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
    parser.add_argument(
        "--time", type=int, default=-1, help="Time limit of the experiment"
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
    name, cfg, tune_config, rm=False, progress=False, verbose=False, num_samples=1, time_budget_s=None
):
    cfg = copy.deepcopy(cfg)
    if rm:
        import shutil

        shutil.rmtree(os.path.join(cfg.RESULTS_DIR, name), ignore_errors=True)

    try:
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
            time_budget_s=time_budget_s
        )
    except Exception as e:
        print(e)
        # print traceback
        import traceback

        traceback.print_exc()
        
        
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
    t = args.time if args.time > 0 else None

    cfg = get_cfg_defaults()
    cfg.DATAMODULE.BATCH_SIZE = 64
    cfg.DATAMODULE.NUM_WORKERS = 8
    cfg.TRAINER.ACCUMULATE_GRAD_BATCHES = 1
    cfg.OPTIMIZER.LR = 3e-3
    cfg.OPTIMIZER.NAME = "AdamW"
    cfg.DATASET.ROIS = ["all"]
    cfg.DATASET.FMRI_SPACE = 'fship'
    cfg.DATASET.SUBJECT_LIST = ["subj01"]
    cfg.MODEL.BACKBONE.NAME = "dinov2_vit_s"
    cfg.TRAINER.CALLBACKS.EARLY_STOP.PATIENCE = 10
    cfg.MODEL.CONV_HEAD.SIMPLE = True
    cfg.MODEL.CONV_HEAD.WIDTH = 64
    cfg.MODEL.CONV_HEAD.MAX_DIM = 256
    cfg.MODEL.MAX_TRAIN_VOXELS = 1145141919810
    cfg.TRAINER.PRECISION = 16
    cfg.TRAINER.LIMIT_TRAIN_BATCHES = 0.5
    cfg.TRAINER.LIMIT_VAL_BATCHES = 0.25

    cfg.RESULTS_DIR = "/nfscc/ray_results/fship_grid/"
    
    
    cfg.EXPERIMENTAL.USE_PREV_FRAME = False
    cfg.EXPERIMENTAL.STRAIGHT_FORWARD = True
    cfg.EXPERIMENTAL.STRAIGHT_FORWARD_BUT_KEEP_BACKBONE_GRAD = False
    cfg.EXPERIMENTAL.BLANK_IMAGE = False
    cfg.EXPERIMENTAL.ANOTHER_SPLIT = True
    cfg.EXPERIMENTAL.SHUFFLE_VAL = True
    
    cfg.MODEL.BACKBONE.LORA.SCALE = 0.0
    cfg.MODEL.BACKBONE.ADAPTIVE_LN.SCALE = 0.0
    for subject in ["subj01", "subj02", "subj03", "subj04", "subj05", "subj06", "subj07", "subj08"]:
        cfg.DATASET.SUBJECT_LIST = [subject]
        # baseline
        tune_config = {
            "EXPERIMENTAL.BLANK_IMAGE": tune.grid_search([True]),
        }
        name = f"blank_{subject}"
        run_ray(name, cfg, tune_config, args.rm, args.progress, args.verbose, 1, t)
        
        # behv
        # RT: [0]
        # ANS: [1, 5, 6]
        # BUTTON: [7]
        # OLD: [2, 3, 4, 8, 9, 10, 11]
        
        cfg.EXPERIMENTAL.BLANK_IMAGE = True
        cfg.EXPERIMENTAL.STRAIGHT_FORWARD_BUT_KEEP_BACKBONE_GRAD = True
        cfg.MODEL.BACKBONE.LORA.SCALE = 0.2
        cfg.MODEL.BACKBONE.ADAPTIVE_LN.SCALE = 0.5
        cfg.MODEL.COND.DROPOUT = 0.0
        tune_config = {
            "EXPERIMENTAL.BEHV_SELECTION": tune.grid_search([[0], [1, 5, 6], [7], [2, 3, 4, 8, 9, 10, 11], [-1]]),
            "MODEL.COND.IN_DIM": tune.sample_from(lambda spec: int(len(spec.config['EXPERIMENTAL.BEHV_SELECTION']) * 2) if spec.config['EXPERIMENTAL.BEHV_SELECTION'] != [-1] else 24),
        }
        name = f"behv_{subject}"
        run_ray(name, cfg, tune_config, args.rm, args.progress, args.verbose, 1, t)
        
        # prev
        cfg.EXPERIMENTAL.BLANK_IMAGE = False
        cfg.EXPERIMENTAL.STRAIGHT_FORWARD_BUT_KEEP_BACKBONE_GRAD = False
        cfg.MODEL.BACKBONE.LORA.SCALE = 0.0
        cfg.MODEL.BACKBONE.ADAPTIVE_LN.SCALE = 0.0
        tune_config = {
            "EXPERIMENTAL.T_IMAGE": tune.grid_search(list(range(0, -30, -1))),
        }
        name = f"prev_{subject}"
        run_ray(name, cfg, tune_config, args.rm, args.progress, args.verbose, 1, t)