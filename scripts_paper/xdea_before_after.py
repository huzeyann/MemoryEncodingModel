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
        topyneck_path="/nfscc/alg23/xdcab/dev/topyneck.pth",
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
            time_budget_s=time_budget_s,
        )
    except Exception as e:
        print(e)
        # print traceback
        import traceback

        traceback.print_exc()


answer = np.concatenate([np.arange(0, 8), np.arange(21, 35)]).tolist()
memory = np.arange(8, 19).tolist()
time = np.arange(19, 21).tolist()

full = np.arange(0, 35).tolist()

no_answer = [i for i in full if i not in answer]
no_memory = [i for i in full if i not in memory]
no_time = [i for i in full if i not in time]

ROW_LIST = [
    {
        "EXPERIMENTAL.USE_PREV_FRAME": True,
        "EXPERIMENTAL.USE_BHV": True,
        "EXPERIMENTAL.USE_BHV_PASSTHROUGH": True,
    },  # after
    {
        "EXPERIMENTAL.USE_PREV_FRAME": False,
        "EXPERIMENTAL.USE_BHV": False,
        "EXPERIMENTAL.USE_BHV_PASSTHROUGH": False,
    },  # before
]
# -
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    t = args.time if args.time > 0 else None

    cfg = load_from_yaml("/workspace/configs/dev_B.yaml")

    cfg.DATASET.SUBJECT_LIST = ["subj01"]
    cfg.DATASET.ROIS = ["all"]
    cfg.DATASET.FMRI_SPACE = "fship_b2"

    cfg.TRAINER.LIMIT_TRAIN_BATCHES = 0.5
    cfg.TRAINER.LIMIT_VAL_BATCHES = 0.5
    cfg.TRAINER.CALLBACKS.EARLY_STOP.PATIENCE = 30

    cfg.RESULTS_DIR = "/nfscc/alg23/xdea/b2"

    cfg.EXPERIMENTAL.USE_DEV_MODEL = True

    tune_config = {
        "row": tune.grid_search(list(range(len(ROW_LIST)))),
    }
    name = f"ba"
    run_ray(name, cfg, tune_config, args.rm, args.progress, args.verbose, 1, t)
