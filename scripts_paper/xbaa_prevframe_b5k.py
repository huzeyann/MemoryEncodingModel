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

@my_nfs_cluster_job
def job(tune_dict, cfg, progress=False, **kwargs):
    if "row" in tune_dict:
        global ROW_LIST
        row = tune_dict["row"]
        tune_dict.pop("row")
        print(ROW_LIST[row])
        tune_dict.update(ROW_LIST[row])

    cfg.merge_from_list(dict_to_list(tune_dict))
        
    cfg = max_batch_size(cfg)

    ret = simple_train(
        cfg=cfg,
        progress=progress,
        rm_soup=True,
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

    cfg = load_from_yaml("/workspace/configs/bold5000.yaml")

    cfg.DATASET.ROIS = ["all"]
    cfg.DATASET.FMRI_SPACE = 'visual_B'

    cfg.RESULTS_DIR = "/nfscc/alg23/xbaa/"
    
    cfg.EXPERIMENTAL.USE_DEV_MODEL = True
    
    cfg.EXPERIMENTAL.USE_PREV_FRAME = False
    cfg.EXPERIMENTAL.BLANK_IMAGE = False
    cfg.EXPERIMENTAL.USE_RETINA_MAPPER = False
    cfg.EXPERIMENTAL.USE_LAYER_SELECTOR = False
    cfg.EXPERIMENTAL.USE_BHV = False
    cfg.EXPERIMENTAL.USE_BHV_PASSTHROUGH = False
    cfg.EXPERIMENTAL.BACKBONE_NOGRAD = True
    cfg.MODEL.BACKBONE.LORA.SCALE = 0.0
    cfg.MODEL.BACKBONE.ADAPTIVE_LN.SCALE = 0.0
    
    # prev + current
    cfg.EXPERIMENTAL.USE_PREV_FRAME = True
    tune_config = {
        "DATASET.FMRI_SPACE": tune.grid_search(['visual_D', 'visual_B']),
        "DATASET.SUBJECT_LIST": tune.grid_search([['CSI1'], ['CSI2'], ['CSI3']]),
        "EXPERIMENTAL.USE_PREV_FRAME": tune.grid_search([True]),
        "EXPERIMENTAL.T_IMAGE": tune.grid_search([0]),
    }
    name = f"prev_current"
    run_ray(name, cfg, tune_config, args.rm, args.progress, args.verbose, 1, t)
    
    # baseline
    tune_config = {
        "DATASET.FMRI_SPACE": tune.grid_search(['visual_D', 'visual_B']),
        "DATASET.SUBJECT_LIST": tune.grid_search([['CSI1'], ['CSI2'], ['CSI3']]),
        "EXPERIMENTAL.SHUFFLE_IMAGES": tune.grid_search([True]),
    }
    name = f"rand"
    run_ray(name, cfg, tune_config, args.rm, args.progress, args.verbose, 1, t)
    
    # prev
    cfg.EXPERIMENTAL.BLANK_IMAGE = False
    tune_config = {
        "DATASET.FMRI_SPACE": tune.grid_search(['visual_D', 'visual_B']),
        "DATASET.SUBJECT_LIST": tune.grid_search([['CSI1'], ['CSI2'], ['CSI3']]),
        "EXPERIMENTAL.T_IMAGE": tune.grid_search(list(range(0, -32, -1))),
    }
    name = f"prev"
    run_ray(name, cfg, tune_config, args.rm, args.progress, args.verbose, 1, t)
    