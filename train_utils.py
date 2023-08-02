import datetime
import logging
import operator
import os
import shutil
from typing import Dict
import torch
import numpy as np

import pytorch_lightning as pl

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.strategies import DDPStrategy

from config import AutoConfig

from ray import tune

from pytorch_lightning.callbacks import (
    BackboneFinetuning,
    EarlyStopping,
    ModelCheckpoint,
)
from pytorch_lightning import loggers as pl_loggers
from datamodule import NSDDatamodule
from models import DevVoxelWiseEncodingModel

from plmodels import PlVEModel

torch.set_float32_matmul_precision("medium")


def max_batch_size(cfg: AutoConfig):
    if cfg.TRAINER.ACCUMULATE_GRAD_BATCHES == 1:
        return cfg
    gpu_mem = torch.cuda.mem_get_info()[0]
    base = 22000000000
    if gpu_mem > base:
        original_batch_size = (
            cfg.DATAMODULE.BATCH_SIZE * cfg.TRAINER.ACCUMULATE_GRAD_BATCHES
        )
        r = gpu_mem // base
        cfg.DATAMODULE.BATCH_SIZE = int(cfg.DATAMODULE.BATCH_SIZE * r)
        cfg.TRAINER.ACCUMULATE_GRAD_BATCHES = int(
            original_batch_size / cfg.DATAMODULE.BATCH_SIZE
        )
        assert cfg.TRAINER.ACCUMULATE_GRAD_BATCHES >= 1
        logging.warning(f"Batch size is now {cfg.DATAMODULE.BATCH_SIZE}")
        logging.warning(
            f"Accumulate grad batches is now {cfg.TRAINER.ACCUMULATE_GRAD_BATCHES}"
        )
    return cfg


class EmptyCache(Callback):
    def __init__(self):
        super().__init__()

        hooks = [
            "on_validation_epoch_end",
            "on_train_epoch_end",
            "on_test_epoch_end",
            "on_predict_epoch_end",
            "on_validation_epoch_start",
            "on_train_epoch_start",
            "on_test_epoch_start",
            "on_predict_epoch_start",
        ]

        def empty_cache(*args, **kwargs):
            torch.cuda.empty_cache()

        for hook in hooks:
            setattr(self, hook, empty_cache)


class LogCoordsMLPCallback(Callback):
    def __init__(self, cfg: AutoConfig):
        super().__init__()
        self.cfg = cfg

    @torch.no_grad()
    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: PlVEModel
    ) -> None:
        model: DevVoxelWiseEncodingModel = pl_module.model
        coord_dict = pl_module.coord_dict
        voxel_indices = ...
        step = trainer.global_step
        subject_list = pl_module.subject_list

        def log_hist(name, tensor):
            try:
                pl_module.logger.experiment.add_histogram(
                    name, tensor.flatten(), step)
            except Exception as e:
                print(e)

        for subject_name in subject_list:
            # memory_gate = model.memory_gate[subject_name](
            #     coord_dict[subject_name], voxel_indices
            # )
            # for i in range(memory_gate.shape[1]):
            #     log_hist(f"memory_gate/{subject_name}/{i+1}", memory_gate[:, i])
            layer_selector = model.layer_selector[subject_name](
                coord_dict[subject_name], voxel_indices
            )
            for i in range(layer_selector.shape[1]):
                log_hist(
                    f"layer_selector/{subject_name}/{i+1}", layer_selector[:, i])
            retina_mapper = model.retina_mapper[subject_name](
                coord_dict[subject_name], voxel_indices
            )
            # log_hist(f"memory_gate/{subject_name}/all", memory_gate)
            log_hist(f"layer_selector/{subject_name}/all", layer_selector)
            log_hist(f"retina_mapper/{subject_name}/all", retina_mapper)


def get_callbacks_and_loggers(
    cfg: AutoConfig,
    sub_dir: str = None,
    log_dir: str = None,
    rm_log_dir: bool = True,
    version: str = ".",
    ckpt_dir="/data/ckpt/",
):
    log_dir = tune.get_trial_dir() if log_dir is None else log_dir
    log_dir = cfg.RESULTS_DIR if log_dir is None else log_dir
    if sub_dir is not None:
        log_dir = os.path.join(log_dir, sub_dir)

    logging.warning(f"Logging to {log_dir}")
    trial_dir = os.path.basename(os.path.dirname(log_dir))
    ckpt_dir = os.path.join(ckpt_dir, trial_dir)
    logging.warning(f"Checkpointing to {ckpt_dir}")
    shutil.rmtree(ckpt_dir, ignore_errors=True)  # TODO: maybe resume from ckpt
    # time_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    # ckpt_dir = os.path.join(ckpt_dir, time_str)

    metrics_name = "VAL/PearsonCorrCoef/mean"
    callbacks = []
    callbacks.append(EmptyCache())
    callbacks.append(
        EarlyStopping(
            monitor=metrics_name,
            min_delta=0.0,
            patience=cfg.TRAINER.CALLBACKS.EARLY_STOP.PATIENCE,
            verbose=False,
            mode="max",
        )
    )
    if cfg.TRAINER.CALLBACKS.CHECKPOINT.SAVE_TOP_K > 0:
        callbacks.append(
            ModelCheckpoint(
                monitor=metrics_name,
                dirpath=ckpt_dir,
                filename="{epoch:06d}",
                # auto_insert_metric_name=True,
                save_weights_only=True,
                save_top_k=cfg.TRAINER.CALLBACKS.CHECKPOINT.SAVE_TOP_K,
                mode="max",
            )
        )
    if cfg.MODEL.COORDS_MLP.LOG:
        callbacks.append(LogCoordsMLPCallback(cfg))

    loggers = []
    loggers.append(pl_loggers.TensorBoardLogger(log_dir, version=version))
    loggers.append(pl_loggers.CSVLogger(log_dir, version=version))

    return callbacks, loggers, log_dir, ckpt_dir


def log_metric(trainer, metric, keys, prefix=""):
    if isinstance(metric, list):
        assert len(metric) == 1
        metric = metric[-1]
    for key in keys:
        if key in metric:
            logger = trainer.logger.experiment
            logger.add_scalar(prefix + key, metric[key], trainer.global_step)


def validate_test_log(trainer, model, dm, prefix=""):
    metric = trainer.validate(model, datamodule=dm)[-1]
    log_metric(
        trainer,
        metric,
        ["VAL/PearsonCorrCoef/mean", "VAL/PearsonCorrCoef/challenge"],
        prefix=prefix,
    )
    val_score = metric["VAL/PearsonCorrCoef/mean"]
    metric = trainer.test(model, datamodule=dm)[-1]
    log_metric(
        trainer,
        metric,
        ["TEST/PearsonCorrCoef/mean", "TEST/PearsonCorrCoef/challenge"],
        prefix=prefix,
    )
    test_score = metric["TEST/PearsonCorrCoef/mean"]
    # model.log("hp_metric", test_score)
    return val_score, test_score


def greedy_soup_sh_voxel(
    trainer,
    dm,
    model,
    best_k_models: Dict[str, float],
    log_dir: str,
    target="heldout",
    prefix="SOUP_SH/greedy/",
):
    NUM_MODELS = len(best_k_models)
    best_k_models = sorted(best_k_models.items(), key=operator.itemgetter(1))
    best_k_models.reverse()
    sorted_models = [x[0] for x in best_k_models]
    greedy_soup_ingredients = [sorted_models[0]]
    greedy_soup_params = torch.load(sorted_models[0])
    if "state_dict" in greedy_soup_params:
        greedy_soup_params = greedy_soup_params["state_dict"]
    # best_score_so_far = best_k_models[0][1]
    best_score_so_far = 0.0
    for j in range(0, NUM_MODELS):
        print(f"Greedy soup: {j}/{NUM_MODELS}")

        # Get the potential greedy soup, which consists of the greedy soup with the new model added.
        new_ingredient_params = torch.load(sorted_models[j])
        if "state_dict" in new_ingredient_params:
            new_ingredient_params = new_ingredient_params["state_dict"]
        num_ingredients = len(greedy_soup_ingredients)
        potential_greedy_soup_params = {
            k: greedy_soup_params[k].clone()
            * (num_ingredients / (num_ingredients + 1.0))
            + new_ingredient_params[k].clone() * (1.0 / (num_ingredients + 1))
            for k in new_ingredient_params
        }
        model.load_state_dict(potential_greedy_soup_params)
        if target == "val":
            ret = trainer.validate(model, datamodule=dm)
            current_score = ret[-1]["VAL/PearsonCorrCoef/mean"]
        elif target == "heldout":
            ret = trainer.test(model, datamodule=dm)
            current_score = ret[-1]["TEST/PearsonCorrCoef/mean"]
        else:
            raise ValueError(f"Invalid target: {target}")
        print(
            f"Current score: {current_score}, best score so far: {best_score_so_far}")

        if current_score > best_score_so_far:
            greedy_soup_ingredients.append(sorted_models[j])
            best_score_so_far = current_score
            greedy_soup_params = potential_greedy_soup_params
            print(
                f"Greedy soup improved to {len(greedy_soup_ingredients)} models.")

    model.load_state_dict(greedy_soup_params)
    val_score, test_score = validate_test_log(
        trainer, model, dm, prefix=prefix)

    save_path = os.path.join(log_dir, "soup.pth")
    print(f"Saving greedy soup to {save_path}")
    torch.save(greedy_soup_params, save_path)
    torch.save(test_score, os.path.join(log_dir, "soup_test_score.pth"))
    torch.save(val_score, os.path.join(log_dir, "soup_val_score.pth"))
    torch.save(test_score, os.path.join(
        log_dir, f"soup_test_score={test_score:.6f}"))
    torch.save(val_score, os.path.join(
        log_dir, f"soup_val_score={val_score:.6f}"))

    return val_score, test_score


def freeze(model):
    model.requires_grad_(False)


def simple_train(
    cfg: AutoConfig, progress=True, rm_soup=False, topyneck_path=None, **kwargs
):
    dm = NSDDatamodule(cfg)
    dm.setup()

    plmodel = PlVEModel(cfg, dm.roi_dict, dm.neuron_coords_dict)
    if topyneck_path is not None and len(topyneck_path) > 0:
        if not os.path.exists(topyneck_path):
            raise ValueError(f"topyneck_path does not exist: {topyneck_path}")
        else:
            plmodel.load_state_dict(torch.load(topyneck_path), strict=False)
            freeze(plmodel.model.retina_mapper)
            freeze(plmodel.model.layer_selector)

    callbacks, loggers, log_dir, ckpt_dir = get_callbacks_and_loggers(
        cfg, ckpt_dir=cfg.CHECKPOINT_DIR)
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=loggers,
        max_epochs=cfg.TRAINER.MAX_EPOCHS,
        accelerator="gpu",
        devices=[0],
        precision=cfg.TRAINER.PRECISION,
        limit_train_batches=cfg.TRAINER.LIMIT_TRAIN_BATCHES,
        limit_val_batches=cfg.TRAINER.LIMIT_VAL_BATCHES,
        accumulate_grad_batches=cfg.TRAINER.ACCUMULATE_GRAD_BATCHES,
        log_every_n_steps=cfg.TRAINER.LOG_TRAIN_N_STEPS,
        gradient_clip_val=cfg.TRAINER.GRADIENT_CLIP_VAL,
        enable_progress_bar=progress,
    )

    trainer.fit(plmodel, datamodule=dm)

    dm.cfg.EXPERIMENTAL.SHUFFLE_VAL = False
    trainer.limit_val_batches = 1.0

    best_k_models = None
    best_model_path = None

    trainer.checkpoint_callback.to_yaml(
        os.path.join(log_dir, "checkpoint.yaml"))

    ckpt: ModelCheckpoint = trainer.checkpoint_callback
    best_k_models = ckpt.best_k_models
    best_model_path = ckpt.best_model_path

    val_score, test_score = greedy_soup_sh_voxel(
        trainer, dm, plmodel, best_k_models, log_dir, target="heldout"
    )
    if cfg.TRAINER.CALLBACKS.CHECKPOINT.REMOVE:
        shutil.rmtree(ckpt_dir, ignore_errors=True)

    soup_path = os.path.join(log_dir, "soup.pth")

    if rm_soup:
        os.remove(soup_path)


def modular_train(
    cfg: AutoConfig, progress=True, use_ddp=False, topyneck_path=None, **kwargs
):
    if cfg.LOSS.SYNC.USE and use_ddp:
        raise ValueError("Cannot use both DDP and LOSS.SYNC")

    dm = NSDDatamodule(cfg)
    dm.setup()

    plmodel = PlVEModel(cfg, dm.roi_dict, dm.neuron_coords_dict)
    if topyneck_path is not None and os.path.exists(topyneck_path):
        plmodel.load_state_dict(torch.load(topyneck_path), strict=False)
        freeze(plmodel.model.retina_mapper)
        freeze(plmodel.model.layer_selector)

    callbacks, loggers, log_dir, ckpt_dir = get_callbacks_and_loggers(
        cfg, ckpt_dir=cfg.CHECKPOINT_DIR)
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=loggers,
        max_epochs=cfg.TRAINER.MAX_EPOCHS,
        accelerator="gpu",
        devices=-1 if use_ddp else [0],
        strategy=DDPStrategy(find_unused_parameters=True) if use_ddp else None,
        # use_amp=True,
        precision=cfg.TRAINER.PRECISION,
        limit_train_batches=cfg.TRAINER.LIMIT_TRAIN_BATCHES,
        limit_val_batches=cfg.TRAINER.LIMIT_VAL_BATCHES,
        accumulate_grad_batches=cfg.TRAINER.ACCUMULATE_GRAD_BATCHES,
        log_every_n_steps=cfg.TRAINER.LOG_TRAIN_N_STEPS,
        gradient_clip_val=cfg.TRAINER.GRADIENT_CLIP_VAL,
        enable_progress_bar=progress,
    )

    trainer.fit(plmodel, datamodule=dm)

    trainer.checkpoint_callback.to_yaml(
        os.path.join(log_dir, "checkpoint.yaml"))
    
    # mark ckpt_dir as done
    with open(os.path.join(ckpt_dir, "done"), "w") as f:
        f.write("done")