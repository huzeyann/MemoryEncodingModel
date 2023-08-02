from copy import copy
import operator
import os
import time
from typing import Dict, List, Union
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

from topyneck import NeuronProjector, TopyNeck, VoxelOutBlock

plt.style.use("dark_background")
# matplotlib.use("Agg")

import numpy as np
import pytorch_lightning as pl
import torch
from einops import rearrange, reduce
from pyparsing import Any, Optional
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.finetuning import BaseFinetuning
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor, nn

from metrics import EpochMetric, vectorized_correlation
from plmodels import PlVEModel

import logging
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Union

import torch
from torch.nn import Module, ModuleDict
from torch.nn.modules.batchnorm import _BatchNorm
from torch.optim.optimizer import Optimizer

from pytorch_lightning.callbacks import ModelCheckpoint

import pytorch_lightning as pl
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import rank_zero_warn

log = logging.getLogger(__name__)


class DisableBN(Callback):
    def __init__(self):
        super().__init__()

    def on_train_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        def disable_bn(model):
            for module in model.modules():
                if (
                    isinstance(module, nn.BatchNorm3d)
                    or isinstance(module, nn.BatchNorm2d)
                    or isinstance(module, nn.BatchNorm1d)
                ):
                    module.eval()

        pl_module.backbone.apply(disable_bn)


class ModifyBNMoment(Callback):
    def __init__(self, momentum: float):
        super().__init__()
        self.momentum = momentum

    def on_fit_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        def modify_bn_moment(model):
            for module in model.modules():
                if (
                    isinstance(module, nn.BatchNorm3d)
                    or isinstance(module, nn.BatchNorm2d)
                    or isinstance(module, nn.BatchNorm1d)
                ):
                    module.momentum = self.momentum

        pl_module.backbone.apply(modify_bn_moment)


class SaveFinalFC(Callback):
    def __init__(self, log_dir: str):
        super().__init__()
        self.log_dir = log_dir

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: PlVEModel
    ) -> None:
        """_summary_

        Args:
            trainer (pl.Trainer): _description_
            pl_module (pl.LightningModule): _description_
        """
        step = trainer.global_step
        save_dir = os.path.join(self.log_dir, f"last_linear")
        os.makedirs(save_dir, exist_ok=True)
        torch.save(
            pl_module.neck.final_fc.state_dict(),
            os.path.join(save_dir, f"{step:012d}.pt"),
        )


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


class LoadBestCheckpointOnVal(Callback):
    # this may be a bad idea
    def __init__(self):
        super().__init__()
        self.prev_best_path = None

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: PlVEModel
    ) -> None:
        best_path = trainer.checkpoint_callback.best_model_path
        if best_path is None or best_path == "":
            return
        if best_path == self.prev_best_path:
            log.warning(f"reloading best checkpoint: {best_path}")
            state_dict = torch.load(best_path)["state_dict"]
            pl_module.load_state_dict(state_dict)
        else:
            # log.warning(f"best checkpoint not changed: {best_path}")
            self.prev_best_path = best_path


class LoadBestCheckpointOnEnd(Callback):
    def __init__(self):
        super().__init__()

    def on_train_end(self, trainer: "pl.Trainer", pl_module: PlVEModel) -> None:
        checkpoint_path = trainer.checkpoint_callback.best_model_path
        state_dict = torch.load(checkpoint_path)["state_dict"]
        pl_module.load_state_dict(state_dict)


# class MoudelSoup(Callback):
#     def __init__(self, recipes: str = 'greedy'):
#         super().__init__()
#         self.recipes = recipes

#     def on_train_end(self, trainer: "pl.Trainer", pl_module: VEModel) -> None:
#         ckpt: ModelCheckpoint = trainer.checkpoint_callback
#         best_k_models = ckpt.best_k_models
#         NUM_MODELS = len(best_k_models)

#         if "uniform" == self.recipes:
#             for j, path in enumerate(best_k_models):
#                 state_dict = torch.load(path)["state_dict"]
#                 if j == 0:
#                     uniform_soup = {
#                         k: v * (1.0 / NUM_MODELS) for k, v in state_dict.items()
#                     }
#                 else:
#                     uniform_soup = {
#                         k: v * (1.0 / NUM_MODELS) + uniform_soup[k]
#                         for k, v in state_dict.items()
#                     }
#             pl_module.load_state_dict(uniform_soup)
#             # trainer.test(pl_module)

#         if 'greedy' == self.recipes:
#             greedy_soup = {}
#             best_k_models = sorted(best_k_models.items(), key=operator.itemgetter(1))
#             best_k_models.reverse()


class RemoveCheckpoint(Callback):
    def __init__(self, log_dir: str):
        super().__init__()
        self.log_dir = log_dir

    def on_train_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        import shutil

        shutil.rmtree(self.log_dir, ignore_errors=True)


class SaveOutput(Callback):
    def __init__(self, log_dir: str) -> None:
        super().__init__()
        self.save_dir = os.path.join(log_dir, "outputs")
        os.makedirs(self.save_dir, exist_ok=True)

    def on_train_end(self, trainer: "pl.Trainer", pl_module: PlVEModel) -> None:
        # checkpoint_path = trainer.checkpoint_callback.best_model_path
        # state_dict = torch.load(checkpoint_path)["state_dict"]
        # pl_module.load_state_dict(state_dict)

        for sub in pl_module.cfg.DATASET.SUBJECT_LIST:
            dl = trainer.datamodule.val_dataloader(subject=sub)
            out = trainer.predict(pl_module, dataloaders=dl)
            out = sum(out, [])
            out = torch.cat(out, 0)

            torch.save(out, os.path.join(self.save_dir, f"{sub}.val.pt"))

            dl = trainer.datamodule.test_dataloader(subject=sub)
            out = trainer.predict(pl_module, dataloaders=dl)
            out = sum(out, [])
            out = torch.cat(out, 0)
            torch.save(out, os.path.join(self.save_dir, f"{sub}.test.pt"))


class SaveNeuronLocation(Callback):
    def __init__(self, log_dir: str, save=True, draw=True) -> None:
        super().__init__()
        self.save = save
        self.draw = draw
        self.save_dir = os.path.join(log_dir, "neuron_location")
        self.tmp_png = "/tmp/tmp.png"
        os.makedirs(self.save_dir, exist_ok=True)

    def scatter_plot_mu(self, mu):
        # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        fig = plt.figure(figsize=(5, 5))
        plt.scatter(mu[:, 0], mu[:, 1], s=1, c="pink", alpha=0.9)
        # plt.scatter(mu[:, 0], mu[:, 1], s=sigma[:, 0] * 10, c="aqua", alpha=0.1)
        for spine in ["top", "right"]:
            plt.gca().spines[spine].set_visible(False)
        plt.xlim(-1.05, 1.05)
        plt.ylim(-1.05, 1.05)

        return fig

    def scatter_plot_gate_mu(self, mu, gate, argmax=False):
        # np.random.seed(0)
        # random_indices = np.random.choice(mu.shape[0], 1000, replace=False)
        # mu = mu[random_indices]
        # gate = gate[random_indices]

        if argmax:
            labels = np.argmax(gate, axis=1) + 1
        else:
            arr = np.arange(gate.shape[1]) + 1
            arr = arr.reshape(1, -1)
            labels = np.sum(gate * arr, axis=1)
        # cm = plt.get_cmap("jet")
        # norm = matplotlib.colors.Normalize(vmin=0, vmax=gate.shape[1])
        fig = plt.figure(figsize=(5, 5))
        plt.scatter(
            mu[:, 0],
            mu[:, 1],
            s=1,
            c=labels,
            alpha=0.9,
            cmap="gist_rainbow",
            vmin=1,
            vmax=4,
        )
        plt.colorbar()
        # plt.scatter(mu[:, 0], mu[:, 1], s=sigma[:, 0] * 10, c="aqua", alpha=0.1)
        for spine in ["top", "right"]:
            plt.gca().spines[spine].set_visible(False)
        plt.xlim(-1.05, 1.05)
        plt.ylim(-1.05, 1.05)
        return fig

    def scatter_plot_shift_th(self, mu, shift_th):
        fig = plt.figure(figsize=(5, 5))
        plt.scatter(
            mu[:, 0],
            mu[:, 1],
            s=1,
            c=shift_th,
            alpha=0.9,
            cmap="viridis",
            vmin=-1,
            vmax=1,
        )
        for spine in ["top", "right"]:
            plt.gca().spines[spine].set_visible(False)
        plt.xlim(-1.05, 1.05)
        plt.ylim(-1.05, 1.05)
        return fig

    @torch.no_grad()
    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: PlVEModel
    ) -> None:
        try:
            start_time = time.time()
            subject_list = pl_module.subject_list
            step = trainer.global_step
            neck: TopyNeck = pl_module.neck

            mu_outs = {}
            # sigma_outs = {}
            gate_outs = {}
            # shift_th_outs = {}
            for sub in subject_list:
                block: NeuronProjector = neck.neuron_projectors[sub]
                mus, gate, reg_mu = block(batch_size=1)
                mus = {k: v.squeeze(0).squeeze(1).cpu().numpy() for k, v in mus.items()}
                gate = gate.cpu().numpy()
                # mu_outs[sub] = mus
                gate_outs[sub] = gate
                pl_module.logger.experiment.add_histogram(f"gate/{sub}", gate, step)
                pl_module.logger.experiment.add_histogram(
                    f"gate_max/{sub}", np.argmax(gate, axis=1), step
                )
                for i in range(gate.shape[1]):
                    pl_module.logger.experiment.add_histogram(
                        f"gate_{i}/{sub}", gate[:, i], step
                    )

                layers = list(mus.keys())
                for layer in layers[:1]:
                    mu = mus[layer]
                    mu_outs[sub] = mu
                    pl_module.logger.experiment.add_histogram(
                        f"mu_x/{layer}/{sub}", mu[:, 0], step
                    )
                    pl_module.logger.experiment.add_histogram(
                        f"mu_y/{layer}/{sub}", mu[:, 1], step
                    )

                if self.draw:
                    layers = list(mus.keys())
                    for layer in layers[:1]:
                        mu = mus[layer]

                        # fig = self.scatter_plot_gate_mu(mu, gate)
                        # pl_module.logger.experiment.add_figure(
                        #     f"neuron_gate/{layer}/{sub}", fig, step
                        # )
                        # plt.close(fig)
                        fig = self.scatter_plot_gate_mu(mu, gate, argmax=True)
                        pl_module.logger.experiment.add_figure(
                            f"neuron_gate_argmax/{layer}/{sub}", fig, step
                        )
                        plt.close(fig)

                        # fig = self.scatter_plot_mu(mu)
                        # pl_module.logger.experiment.add_figure(
                        #     f"neuron_location/{layer}/{sub}", fig, step
                        # )
                        # plt.close(fig)

            #         # shift_th = neck.neuron_shifters[sub](neuron_coords)
            #         # shift_th = shift_th.detach().cpu().flatten().numpy()
            #         # shift_th_outs[sub] = shift_th
            #         # fig = self.scatter_plot_shift_th(mu, shift_th)
            #         # pl_module.logger.experiment.add_figure(
            #         #     f"neuron_location_shift_th/{sub}", fig, step
            #         # )
            #         # plt.close(fig)

            if self.save:
                torch.save(mu_outs, os.path.join(self.save_dir, f"{step:012d}.mu.pt"))
                torch.save(gate_outs, os.path.join(self.save_dir, f"{step:012d}.gate.pt"))
                # torch.save(
                #     shift_th_outs,
                #     os.path.join(self.save_dir, f"{step:012d}.shift_th.pt"),
                # )

            # print(f"on_validation_epoch_end: {time.time() - start_time}")
        except Exception as e:
            print(e)


class SaveTopWeightEachVoxel(pl.Callback):
    def __init__(self, top_n=10):
        self.top_n = top_n
        self.w_queue = {}  # [N], N is the number voxels, [10], [depth]
        self.b_queue = {}
        self.s_queue = {}

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: PlVEModel
    ) -> None:
        step = trainer.global_step
        if step == 0:
            return

        for subject_id in pl_module.voxel_score.keys():
            current_score = pl_module.voxel_score[subject_id]
            current_score = current_score.tolist()
            block: VoxelOutBlock = pl_module.neck.voxel_outs[subject_id]
            n = len(current_score)
            if subject_id not in self.s_queue.keys():
                dummy_w = [None for _ in range(self.top_n)]
                self.w_queue[subject_id] = [copy(dummy_w) for _ in range(n)]
                self.b_queue[subject_id] = [copy(dummy_w) for _ in range(n)]
                dummy_score = [-114514 for _ in range(self.top_n)]
                self.s_queue[subject_id] = [copy(dummy_score) for _ in range(n)]

            for i in range(n):
                min_score_idx = np.argmin(self.s_queue[subject_id][i])
                min_score = self.s_queue[subject_id][i][min_score_idx]

                if current_score[i] > min_score:
                    self.w_queue[subject_id][i][min_score_idx] = [
                        w[i].detach().clone().cpu() for w in block.weight
                    ]
                    self.b_queue[subject_id][i][min_score_idx] = [
                        b[i].detach().clone().cpu() for b in block.bias
                    ]
                    self.s_queue[subject_id][i][min_score_idx] = current_score[i]

    def reorder_by_score(self):
        for subject_id in self.s_queue.keys():
            n = len(self.s_queue[subject_id])  # number of voxels
            for i in range(n):
                sorted_idx = np.argsort(self.s_queue[subject_id][i])[::-1]
                self.w_queue[subject_id][i] = [
                    self.w_queue[subject_id][i][idx] for idx in sorted_idx
                ]
                self.b_queue[subject_id][i] = [
                    self.b_queue[subject_id][i][idx] for idx in sorted_idx
                ]
                self.s_queue[subject_id][i] = [
                    self.s_queue[subject_id][i][idx] for idx in sorted_idx
                ]

    def on_fit_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.reorder_by_score()


from ema_pytorch import EMA
class EMAModel(pl.Callback):
    def __init__(self, model: PlVEModel, beta=0.999):

        self.beta = beta
        # self.ema = EMA(model.neck.voxel_outs, beta=self.beta)

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        self.ema.update()

    def on_fit_start(self, trainer, pl_module: PlVEModel) -> None:
        self.ema = EMA(pl_module.neck.voxel_outs, beta=self.beta)
    
    def on_fit_end(self, trainer: pl.Trainer, pl_module: PlVEModel) -> None:
        pl_module.neck.voxel_outs = self.ema.ema_model

    @property
    def ema_model(self):
        return self.ema.ema_model


class StageFinetuning(BaseFinetuning):
    r"""Finetune a backbone model based on a learning rate user-defined scheduling.

    When the backbone learning rate reaches the current model learning rate
    and ``should_align`` is set to True, it will align with it for the rest of the training.

    Args:
        unfreeze_backbone_at_epoch: Epoch at which the backbone will be unfreezed.
        lambda_func: Scheduling function for increasing backbone learning rate.
        backbone_initial_ratio_lr:
            Used to scale down the backbone learning rate compared to rest of model
        backbone_initial_lr: Optional, Initial learning rate for the backbone.
            By default, we will use ``current_learning /  backbone_initial_ratio_lr``
        should_align: Whether to align with current learning rate when backbone learning
            reaches it.
        initial_denom_lr: When unfreezing the backbone, the initial learning rate will
            ``current_learning_rate /  initial_denom_lr``.
        train_bn: Whether to make Batch Normalization trainable.
        verbose: Display current learning rate for model and backbone
        rounding: Precision for displaying learning rate
        freeze_modules: List of modules to freeze when unfreezing the backbone. By default, it's for resnet.
        unfreeze_modules: List of modules to unfreeze when unfreezing the backbone.

    Example::

        >>> from pytorch_lightning import Trainer
        >>> from pytorch_lightning.callbacks import BackboneFinetuning
        >>> multiplicative = lambda epoch: 1.5
        >>> backbone_finetuning = BackboneFinetuning(200, multiplicative)
        >>> trainer = Trainer(callbacks=[backbone_finetuning])

    """

    def __init__(
        self,
        unfreeze_backbone_at_epoch: int = 10,
        lambda_func: Callable = lambda epoch: 1.5,
        backbone_initial_ratio_lr: float = 10e-2,
        backbone_initial_lr: Optional[float] = None,
        should_align: bool = True,
        initial_denom_lr: float = 10.0,
        train_bn: bool = True,
        verbose: bool = False,
        rounding: int = 12,
        unfreeze_modules: Optional[List[str]] = [
            "layer3",
        ],
    ) -> None:
        super().__init__()

        self.unfreeze_backbone_at_epoch: int = unfreeze_backbone_at_epoch
        self.lambda_func: Callable = lambda_func
        self.backbone_initial_ratio_lr: float = backbone_initial_ratio_lr
        self.backbone_initial_lr: Optional[float] = backbone_initial_lr
        self.should_align: bool = should_align
        self.initial_denom_lr: float = initial_denom_lr
        self.train_bn: bool = train_bn
        self.verbose: bool = verbose
        self.rounding: int = rounding
        self.previous_backbone_lr: Optional[float] = None
        self.unfreeze_modules = unfreeze_modules

    def state_dict(self) -> Dict[str, Any]:
        return {
            "internal_optimizer_metadata": self._internal_optimizer_metadata,
            "previous_backbone_lr": self.previous_backbone_lr,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.previous_backbone_lr = state_dict["previous_backbone_lr"]
        super().load_state_dict(state_dict)

    def on_fit_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """
        Raises:
            MisconfigurationException:
                If LightningModule has no nn.Module `backbone` attribute.
        """
        for module_name in self.unfreeze_modules:
            if not hasattr(pl_module.backbone, module_name):
                raise MisconfigurationException(
                    f"The LightningModule should have a nn.Module `{module_name}` attribute"
                )
            if not isinstance(getattr(pl_module.backbone, module_name), Module):
                raise MisconfigurationException(
                    f"The LightningModule attribute `{module_name}` should be a nn.Module"
                )
        return super().on_fit_start(trainer, pl_module)

    def freeze_before_training(self, pl_module: "pl.LightningModule") -> None:
        # for module_name in self.all_modules:
        #     self.freeze(getattr(pl_module.backbone, module_name))
        self.freeze(pl_module.backbone)

    def finetune_function(
        self,
        pl_module: "pl.LightningModule",
        epoch: int,
        optimizer: Optimizer,
        opt_idx: int,
    ) -> None:
        """Called when the epoch begins."""
        if epoch == self.unfreeze_backbone_at_epoch:
            current_lr = optimizer.param_groups[0]["lr"]
            initial_backbone_lr = (
                self.backbone_initial_lr
                if self.backbone_initial_lr is not None
                else current_lr * self.backbone_initial_ratio_lr
            )
            self.previous_backbone_lr = initial_backbone_lr

            for module_name in self.unfreeze_modules:
                self.unfreeze_and_add_param_group(
                    getattr(pl_module.backbone, module_name),
                    optimizer,
                    initial_backbone_lr,
                    train_bn=self.train_bn,
                    initial_denom_lr=self.initial_denom_lr,
                )

            if self.verbose:
                log.info(
                    f"Current lr: {round(current_lr, self.rounding)}, "
                    f"Backbone lr: {round(initial_backbone_lr, self.rounding)}"
                )

        elif epoch > self.unfreeze_backbone_at_epoch:
            current_lr = optimizer.param_groups[0]["lr"]
            next_current_backbone_lr = (
                self.lambda_func(epoch + 1) * self.previous_backbone_lr
            )
            next_current_backbone_lr = (
                current_lr
                if (self.should_align and next_current_backbone_lr > current_lr)
                else next_current_backbone_lr
            )
            optimizer.param_groups[-1]["lr"] = next_current_backbone_lr
            self.previous_backbone_lr = next_current_backbone_lr
            if self.verbose:
                log.info(
                    f"Current lr: {round(current_lr, self.rounding)}, "
                    f"Backbone lr: {round(next_current_backbone_lr, self.rounding)}"
                )


class FreezeBackbone(Callback):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def on_fit_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        BaseFinetuning.freeze(pl_module.backbone)


if __name__ == "__main__":
    pass
