# %%
import os
from config_utils import get_cfg_defaults, save_to_yaml
from config import AutoConfig
# # %%


_C = get_cfg_defaults()

# path = "/workspace/configs/dino_t1.yaml"
# _C.merge_from_file(path)
path = "/workspace/configs/xvaa.yaml"
save_to_yaml(_C, path)