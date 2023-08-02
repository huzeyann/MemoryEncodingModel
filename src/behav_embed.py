from torch import nn, Tensor
from einops import rearrange
from typing import Dict, Optional
from config import AutoConfig


from timm.layers.mlp import Mlp
from timm.layers.norm import LayerNorm



class SubjectBehaviorEmbed(nn.Module):
    def __init__(
        self,
        subject_list,
        in_dim,
        dim,
        dropout=0.2, # dropout for handle behavior data free case
    ):
        super().__init__()
        self.subject_list = subject_list

        self.embed = nn.ModuleDict()
        for subject in self.subject_list:
            block = nn.Sequential(
                nn.Linear(in_dim, dim),
                nn.GELU(),
            )
            self.embed[subject] = block

        self.mlp = Mlp(dim, out_features=dim)
        
        self.dropout = nn.Sequential(
            nn.Unflatten(1, (dim, 1)),  # [B, D, 1]
            nn.Dropout1d(dropout),  # dropout on the entire D
            nn.Flatten(1, -1),  # [B, D]
        )
    def forward(self, c: Tensor, subject: str):
        if c is not None:
            c = self.embed[subject](c)
            c = self.mlp(c)
            c = self.dropout(c)
            # dropout in training but not validation
        return c
    

def build_behavior_embed(cfg: AutoConfig, out_dim=None):
    out_dim = out_dim or cfg.MODEL.COND.DIM
    return SubjectBehaviorEmbed(
        subject_list=cfg.DATASET.SUBJECT_LIST,
        in_dim=cfg.MODEL.COND.IN_DIM,
        dim=out_dim,
        dropout=cfg.MODEL.COND.DROPOUT,
    )
