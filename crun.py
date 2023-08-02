from cluster_utils import my_nfs_cluster_job, trial_dirname_creator
import argparse
import logging
import os
import sys
from random import seed, shuffle

import numpy as np
from ray import tune

from datamodule import NSDDatamodule
from config_utils import dict_to_list, get_cfg_defaults, load_from_yaml

from run_utils import *


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
        "--topyneck",
        type=str,
        default="",
        help="topyneck path",
    )

    return parser


def run_train_with_boost(
    tune_dict,
    cfg: AutoConfig = None,
    progress=False,
    bigmodel_path=None,
    **kwargs,
):
    cfg.merge_from_list(dict_to_list(tune_dict))
    cfg = copy.deepcopy(cfg)

    dm: NSDDatamodule = build_dm(cfg)
    dm.setup()

    sub_dir = f""
    cbs, lgs, log_dir, ckpt_dir = get_callbacks_and_loggers(cfg, sub_dir=sub_dir)

    soup_path = os.path.join(log_dir, "soup.pth")
    if os.path.exists(soup_path):
        print("soup exists, skipping")
        return (None, None, None, None, None, log_dir, ckpt_dir, None)

    os.makedirs(log_dir, exist_ok=True)

    model_args = (
        cfg,
        dm.num_voxel_dict,
        dm.roi_dict,
        dm.neuron_coords_dict,
        dm.noise_ceiling_dict,
    )
    torch.save(cfg, os.path.join(log_dir, "cfg.pth"))

    model = VEModel(*model_args)

    if bigmodel_path is not None and os.path.exists(bigmodel_path):
        # boost speed with learned weights
        state_dict = load_state_dict(bigmodel_path)
        subject_list = dm.dss[0].keys()
        filtered_state_dict = {}
        for key, value in state_dict.items():
            found = False
            for subject in subject_list:
                if subject in key:
                    found = True
                    break
            if not found:
                continue
            # # # print(cfg.DATASET.ROIS)
            # # print(dm.dss[0].keys())
            # print(subject)

            vi = dm.dss[0][subject].voxel_index
            if "voxel_outs" in key:
                raise ValueError("voxel_outs should not be in state_dict")
                value = value[vi]
            filtered_state_dict[key] = value

        model.load_state_dict(filtered_state_dict, strict=False)

        freeze(model.neck.neuron_projectors)
    else:
        logging.info("No bigmodel_path, training from scratch")
    # for subject in subject_list:
    # freeze(model.neck.neuron_projectors[subject].projectors)
    # freeze(model.neck.voxel_outs)
    # freeze(model.backbone)

    trainer = pl.Trainer(
        precision=cfg.TRAINER.PRECISION,
        accelerator="cuda",
        gradient_clip_val=cfg.TRAINER.GRADIENT_CLIP_VAL,
        # strategy=DDPStrategy(find_unused_parameters=False),
        devices=cfg.TRAINER.DEVICES,
        max_epochs=cfg.TRAINER.MAX_EPOCHS,
        max_steps=cfg.TRAINER.MAX_STEPS,
        val_check_interval=cfg.TRAINER.VAL_CHECK_INTERVAL,
        accumulate_grad_batches=cfg.TRAINER.ACCUMULATE_GRAD_BATCHES,
        limit_train_batches=cfg.TRAINER.LIMIT_TRAIN_BATCHES,
        limit_val_batches=cfg.TRAINER.LIMIT_VAL_BATCHES,
        log_every_n_steps=cfg.TRAINER.LOG_TRAIN_N_STEPS,
        callbacks=cbs,
        logger=lgs,
        enable_checkpointing=True,
        enable_progress_bar=progress,
    )

    trainer.fit(model, datamodule=dm)

    best_k_models = None
    best_model_path = None

    trainer.checkpoint_callback.to_yaml(os.path.join(log_dir, "checkpoint.yaml"))

    ckpt: ModelCheckpoint = trainer.checkpoint_callback
    best_k_models = ckpt.best_k_models
    best_model_path = ckpt.best_model_path

    return (best_k_models, best_model_path, trainer, dm, model, log_dir, ckpt_dir, cbs)


@my_nfs_cluster_job
def run(
    tune_dict,
    cfg: AutoConfig = None,
    progress=False,
    bigmodel_path=None,
    **kwargs,
):

    if "row" in tune_dict:
        global ROW_LIST
        row = tune_dict["row"]
        tune_dict.pop("row")
        print(ROW_LIST[row])
        tune_dict.update(ROW_LIST[row])

    cfg.merge_from_list(dict_to_list(tune_dict))
    cfg = copy.deepcopy(cfg)

    (
        best_k_models,
        best_model_path,
        trainer,
        dm,
        model,
        log_dir,
        ckpt_dir,
        cbs,
    ) = run_train_with_boost(
        tune_dict,
        cfg=cfg,
        progress=progress,
        bigmodel_path=bigmodel_path,
        **kwargs,
    )

    from run_utils import greedy_soup_sh_voxel

    val_score, test_score = greedy_soup_sh_voxel(
        trainer, dm, model, best_k_models, log_dir, target="heldout"
    )
    if cfg.TRAINER.CALLBACKS.CHECKPOINT.REMOVE:
        shutil.rmtree(ckpt_dir, ignore_errors=True)


# @my_nfs_cluster_job
# def run(
#     tune_dict,
#     cfg: AutoConfig = None,
#     progress=False,
#     bigmodel_path=None,
#     **kwargs,
# ):
#     import time

#     time.sleep(30)


def run_tune(
    name,
    cfg,
    tune_config,
    rm=False,
    progress=False,
    verbose=False,
    num_samples=1,
    bigmodel_path=None,
):
    if rm:
        import shutil

        shutil.rmtree(os.path.join(cfg.RESULTS_DIR, name), ignore_errors=True)

    ana = tune.run(
        tune.with_parameters(
            run,
            cfg=cfg,
            progress=progress,
            bigmodel_path=bigmodel_path,
        ),
        local_dir="/nfscc/afo/ray_results/",
        # storage_path="/nfscc/afo/ray_results/",
        config=tune_config,
        resources_per_trial={"cpu": 1, "gpu": 1},
        num_samples=num_samples,
        name=name,
        verbose=verbose,
        resume="AUTO+ERRORED",
        trial_dirname_creator=trial_dirname_creator,
    )


# ROW_LIST = [
#     {
#         "MODEL.NECK.POOL_HEAD.USE": False,
#         "MODEL.NECK.POOL_HEAD.NAME": "None",
#     },
#     {
#         "MODEL.NECK.POOL_HEAD.USE": True,
#         "MODEL.NECK.POOL_HEAD.NAME": "AvgMaxPoolHead",
#     },
#     {
#         "MODEL.NECK.POOL_HEAD.USE": True,
#         "MODEL.NECK.POOL_HEAD.NAME": "PoolCompHead",
#         "MODEL.NECK.POOL_HEAD.POOL_SIZE": 2,
#     },
#     {
#         "MODEL.NECK.POOL_HEAD.USE": True,
#         "MODEL.NECK.POOL_HEAD.NAME": "ConvCompHead",
#         "MODEL.NECK.POOL_HEAD.POOL_SIZE": 2,
#         "MODEL.NECK.POOL_HEAD.CONV_KERNEL": 4,
#     },
# ]

# -
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    cfg = load_from_yaml("/workspace/configs/dino_t1.yaml")

    from backbone import LAYER_DICT, RESOLUTION_DICT

    # cfg.TRAINER.MAX_EPOCHS = 1
    
    cfg.DATASET.SUBJECT_LIST = ["NSD_01"]
    cfg.DATAMODULE.BATCH_SIZE = 32
    cfg.TRAINER.ACCUMULATE_GRAD_BATCHES = 1

    tune_config = {
        # "MODEL.NECK.POOL_HEAD.POOL_SIZE": tune.grid_search([2, 4]),
        # "row": tune.grid_search(list(range(len(ROW_LIST)))),
        "OPTIMIZER.LR": tune.grid_search([5e-3]),
        "LOSS.SYNC.USE": tune.grid_search([True, False]),
        # "OPTIMIZER.LR": tune.grid_search([1e-3, 9e-3, 8e-3, 7e-3, 6e-3, 5e-3, 4e-3, 3e-3]),
        # "OPTIMIZER.WEIGHT_DECAY": tune.grid_search([1e-3, 3e-4, 1e-4, 3e-5, 1e-5]),
    }

    name = "callforfix2"

    run_tune(
        name,
        cfg,
        tune_config,
        args.rm,
        args.progress,
        args.verbose,
        15,
        args.topyneck,
    )
