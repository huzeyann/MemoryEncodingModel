from einops import rearrange
from typing import Any, Dict, List, Optional, Tuple
from config import AutoConfig
import torch
from torch import Tensor, nn

from topyneck import TopyNeck

from registry import Registry

NECK = Registry()

NECK.register("TopyNeck", TopyNeck)

def build_neck(
    cfg: AutoConfig,
    c_dict: Dict[str, int],
    num_voxel_dict: Dict[str, int],
    neuron_coords_dict: Dict[str, Tensor],
):
    neck = NECK[cfg.MODEL.NECK.NAME](cfg, c_dict, num_voxel_dict, neuron_coords_dict)

    return neck
