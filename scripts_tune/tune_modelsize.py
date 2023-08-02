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
    parser.add_argument(
        "--topyneck_path", type=str, default="/nfscc/alg23/xvab/mem/topyneck.pth", help="Path to topyneck"
    )
    return parser


@my_nfs_cluster_job
def job(tune_dict, cfg, progress=False, **kwargs):
    topyneck_path = kwargs.pop('topyneck_path')

    cfg.merge_from_list(dict_to_list(tune_dict))

    cfg = max_batch_size(cfg)

    ret = simple_train(
        cfg=cfg,
        progress=progress,
        rm_soup=True,
        topyneck_path=topyneck_path,
        **kwargs,
    )


def run_ray(
    name, cfg, tune_config, rm=False, progress=False, verbose=False, num_samples=1, time_budget_s=None, topyneck_path=None
):
    cfg = copy.deepcopy(cfg)
    if rm:
        import shutil

        shutil.rmtree(os.path.join(cfg.RESULTS_DIR, name), ignore_errors=True)

    try:
        ana = tune.run(
            tune.with_parameters(job, cfg=cfg, progress=progress, topyneck_path=topyneck_path),
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


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    t = args.time if args.time > 0 else None

    cfg = load_from_yaml("/workspace/configs/xvaa.yaml")
    cfg.RESULTS_DIR = "/nfscc/alg23/width/"
    cfg.DATASET.ROIS = ['orig']
    # cfg.DATAMODULE.BATCH_SIZE = 32
    # cfg.OPTIMIZER.LR = 1e-3

    tune_config = {
        "DATASET.SUBJECT_LIST": tune.grid_search([["subj01"]]),
        "MODEL.WIDTH_RATIO": tune.grid_search([1.0, 1.25, 1.5, 2.0]),
    }
    name = f"ratio"
    run_ray(name, cfg, tune_config, args.rm, args.progress, args.verbose, 5, t, args.topyneck_path)
