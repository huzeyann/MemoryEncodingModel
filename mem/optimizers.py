from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import (
    StepLR,
    MultiStepLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
)

from registry import Registry

OPTIMIZER_REGISTRY = Registry()


@OPTIMIZER_REGISTRY.register("AdamW")
def _adamw(cfg, optimizer_grouped_parameters):
    return AdamW(
        optimizer_grouped_parameters,
        lr=cfg.OPTIMIZER.LR,
    )


@OPTIMIZER_REGISTRY.register("AdaBelief")
def _adamb(cfg, optimizer_grouped_parameters):
    from adabelief_pytorch import AdaBelief

    return AdaBelief(
        optimizer_grouped_parameters,
        lr=cfg.OPTIMIZER.LR,
        print_change_log=False,
    )


@OPTIMIZER_REGISTRY.register("SGD")
def _sgd(cfg, optimizer_grouped_parameters):
    return SGD(
        optimizer_grouped_parameters,
        momentum=0.9,
        lr=cfg.OPTIMIZER.LR,
    )


def build_optimizer(cfg, optimizer_grouped_parameters):
    p_list = list(optimizer_grouped_parameters)
    optimizer = OPTIMIZER_REGISTRY[cfg.OPTIMIZER.NAME](cfg, p_list)

    ### milestone scheduler
    # warmup_steps = cfg.OPTIMIZER.WARMUP_STEPS
    # milemilestones = cfg.OPTIMIZER.LR_DECAY_STEP
    # decay = cfg.OPTIMIZER.LR_DECAY_RATE
    # assert len(milemilestones) == len(decay)

    # def warmup(current_step: int):
    #     if current_step < warmup_steps:  # current_step / warmup_steps * base_lr
    #         return float(current_step / warmup_steps)
    #     if current_step in milemilestones:
    #         return decay[milemilestones.index(current_step)]
    #     return 1.0

    # from torch.optim.lr_scheduler import LambdaLR

    # scheduler = LambdaLR(optimizer, lr_lambda=warmup)

    from timm.scheduler.cosine_lr import CosineLRScheduler

    # scheduler = CosineLRScheduler(
    #     optimizer,
    #     t_initial=30,
    #     lr_min=1e-4,
    #     cycle_mul=1,
    #     cycle_decay=0.3,
    #     cycle_limit=100,
    #     warmup_t=10,
    #     warmup_lr_init=1e-3,
    #     k_decay=1,
    # )
    if cfg.OPTIMIZER.SCHEDULER.T_INITIAL == 1:
        cfg.OPTIMIZER.SCHEDULER.LR_MIN = cfg.OPTIMIZER.LR

    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=cfg.OPTIMIZER.SCHEDULER.T_INITIAL,
        lr_min=cfg.OPTIMIZER.SCHEDULER.LR_MIN,
        cycle_mul=cfg.OPTIMIZER.SCHEDULER.T_MULT,
        cycle_decay=cfg.OPTIMIZER.SCHEDULER.CYCLE_DECAY,
        cycle_limit=cfg.OPTIMIZER.SCHEDULER.CYCLE_LIMIT,
        warmup_t=cfg.OPTIMIZER.SCHEDULER.WARMUP_T,
        warmup_lr_init=cfg.OPTIMIZER.SCHEDULER.LR_MIN_WARMUP,
        warmup_prefix=False,
        k_decay=cfg.OPTIMIZER.SCHEDULER.K_DECAY,
    )
    return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
