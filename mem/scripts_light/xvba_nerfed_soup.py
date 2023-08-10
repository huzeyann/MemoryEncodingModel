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

from train_utils import max_batch_size, modular_train, simple_train


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
        "--name", type=str, default="all", help="Name of the experiment"
    )
    parser.add_argument(
        "--topyneck_path", type=str, default="", help="Path to topyneck"
    )
    parser.add_argument(
        "--cfg_path",
        type=str,
        default="/workspace/configs/xvba.yaml",
        help="Path to config",
    )
    parser.add_argument(
        "--results_dir", type=str, default="/nfscc/alg23/xvba/", help="Path to results"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--distill_name", type=str, default="", help="dark knowledge name")
    return parser


@my_nfs_cluster_job
def job(tune_dict, cfg, progress=False, **kwargs):
    topyneck_path = kwargs.pop("topyneck_path")

    cfg.merge_from_list(dict_to_list(tune_dict))

    cfg = max_batch_size(cfg)

    ret = simple_train(  # todo
        cfg=cfg,
        progress=progress,
        topyneck_path=topyneck_path,
        rm_soup=False,
        **kwargs,
    )


def run_ray(
    name,
    cfg,
    tune_config,
    rm=False,
    progress=False,
    verbose=False,
    num_samples=1,
    time_budget_s=None,
    topyneck_path=None,
):
    cfg = copy.deepcopy(cfg)
    if rm:
        import shutil

        shutil.rmtree(os.path.join(cfg.RESULTS_DIR, name), ignore_errors=True)

    try:
        ana = tune.run(
            tune.with_parameters(
                job, cfg=cfg, progress=progress, topyneck_path=topyneck_path
            ),
            local_dir=cfg.RESULTS_DIR,
            config=tune_config,
            resources_per_trial={"cpu": 1, "gpu": 1},
            num_samples=num_samples,
            name=name,
            verbose=verbose,
            resume="AUTO+ERRORED",
            trial_dirname_creator=trial_dirname_creator,
            time_budget_s=time_budget_s,
        )
    except Exception as e:
        print(e)
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    cfg = load_from_yaml(args.cfg_path)
    cfg.RESULTS_DIR = args.results_dir
    cfg.DATAMODULE.BATCH_SIZE = args.batch_size
    
    dark_name = args.distill_name
    if dark_name is not None and len(dark_name) > 0:
        cfg.LOSS.DARK.USE = True
        cfg.DATASET.DARK_POSTFIX = dark_name
        
    tune_config = {
        # "DATASET.SUBJECT_LIST": tune.grid_search([["subj01"]]),
        # "DATASET.ROIS": tune.grid_search([["all"], ["RSC"], ["E"], ["MV"], ["ML"], ["MP"], ["V"], ["L"], ["P"], ["R"]]),
        # "DATASET.ROIS": tune.grid_search([["E"], ["ML"], ["MP"], ["V"]]),
        "DATASET.ROIS": tune.grid_search([["all"]]),
    }
    name = args.name
    run_ray(
        name,
        cfg,
        tune_config,
        args.rm,
        args.progress,
        args.verbose,
        1,
        None,
        args.topyneck_path,
    )
