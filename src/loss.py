import os
from typing import Tuple

import numpy as np
import torch

from config import AutoConfig
from registry import Registry

LOSS = Registry()


@LOSS.register("MSELoss")
def _mse(cfg: AutoConfig):
    return torch.nn.MSELoss(reduction='none')


@LOSS.register("L1Loss")
def _l1(cfg: AutoConfig):
    return torch.nn.L1Loss(reduction='none')


@LOSS.register("SmoothL1Loss")
def _smooth_l1(cfg: AutoConfig):
    return torch.nn.SmoothL1Loss(beta=cfg.LOSS.SMOOTH_L1_BETA, reduction='none')


@LOSS.register("PoissonNLLLoss")
def _poisson_nll(cfg: AutoConfig):
    return torch.nn.PoissonNLLLoss(reduction='none')


def build_loss(cfg: AutoConfig):
    return LOSS[cfg.LOSS.NAME](cfg)
