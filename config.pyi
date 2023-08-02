import typing as T

from yacs.config import CfgNode as CN

_C: AutoConfig

class Experimental(CN):
    SHUFFLE_IMAGES: bool
    BLANK_IMAGE: bool
    T_IMAGE: int
    USE_PREV_FRAME: bool
    USE_RETINA_MAPPER: bool
    USE_LAYER_SELECTOR: bool
    USE_BHV: bool
    USE_BHV_PASSTHROUGH: bool
    BEHV_ONLY: bool
    BEHV_SELECTION: T.Sequence
    BACKBONE_NOGRAD: bool
    STRAIGHT_FORWARD: bool
    STRAIGHT_FORWARD_BUT_KEEP_BACKBONE_GRAD: bool
    ANOTHER_SPLIT: bool
    SHUFFLE_VAL: bool
    NO_SPLIT: bool
    USE_DEV_MODEL: bool

class Datamodule(CN):
    BATCH_SIZE: int
    NUM_WORKERS: int
    PIN_MEMORY: bool
    FEATURE_EXTRACTOR_MODE: bool

class Dataset(CN):
    IMAGE_RESOLUTION: T.Sequence
    N_PREV_FRAMES: int
    CACHE_DIR: str
    SUBJECT_LIST: T.Sequence
    ROIS: T.Sequence
    FMRI_SPACE: str
    FILTER_BY_SESSION: T.Sequence
    ROOT: str
    DARK_POSTFIX: str

class Position_encoding(CN):
    IN_DIM: int
    MAX_STEPS: int
    FEATURES: int
    PERIODS: int

class Lora(CN):
    SCALE: float
    RANK: int

class Adaptive_ln(CN):
    SCALE: float

class Backbone(CN):
    NAME: str
    CACHE_DIR: str
    LAYERS: T.Sequence
    FEATURE_DIMS: T.Sequence
    CLS_DIMS: T.Sequence
    LORA: Lora
    ADAPTIVE_LN: Adaptive_ln

class Lora_1(CN):
    SCALE: float
    RANK: int

class Adaptive_ln_1(CN):
    SCALE: float

class Backbone_small(CN):
    NAME: str
    LAYERS: T.Sequence
    CLS_DIMS: T.Sequence
    T_DIM: int
    WIDTH: int
    MERGE_WIDTH: int
    LORA: Lora_1
    ADAPTIVE_LN: Adaptive_ln_1

class Prev_feat(CN):
    DIM: int

class Conv_head(CN):
    MAX_DIM: int
    KERNEL_SIZES: T.Sequence
    DEPTHS: T.Sequence
    WIDTH: int
    SIMPLE: bool

class Cond(CN):
    USE: bool
    DROPOUT: float
    IN_DIM: int
    DIM: int
    PASSTHROUGH_DIM: int

class Coords_mlp(CN):
    WIDTH: int
    DEPTH: int
    LOG: bool

class Retina_mapper(CN):
    CONSTANT_SIGMA: float

class Layer_selector(CN): {}

class Bottleneck(CN):
    RANK: int
    OUT_DIM: int

class Mlp(CN):
    DEPTH: int
    WIDTH: int

class Shared(CN):
    USE: bool
    MLP: Mlp

class Voxel_outs(CN):
    SHARED: Shared

class Model(CN):
    WIDTH_RATIO: float
    BACKBONE: Backbone
    BACKBONE_SMALL: Backbone_small
    PREV_FEAT: Prev_feat
    CONV_HEAD: Conv_head
    COND: Cond
    MAX_TRAIN_VOXELS: int
    CHUNK_SIZE: int
    COORDS_MLP: Coords_mlp
    RETINA_MAPPER: Retina_mapper
    LAYER_SELECTOR: Layer_selector
    BOTTLENECK: Bottleneck
    VOXEL_OUTS: Voxel_outs

class Sync(CN):
    USE: bool
    STAGE: str
    SKIP_EPOCHS: int
    EMA_BETA: float
    EMA_BIAS_CORRECTION: bool
    UPDATE_RULE: str
    EXP_SCALE: float
    EXP_SHIFT: float
    LOG_SHIFT: float
    EMA_KEY: str

class Anneal(CN):
    T: int

class Dark(CN):
    USE: bool
    MAX_EPOCH: int
    GT_ROIS: T.Sequence
    GT_SCALE_UP_COEF: float
    ANNEAL: Anneal

class Loss(CN):
    NAME: str
    SMOOTH_L1_BETA: float
    SYNC: Sync
    DARK: Dark

class Regularizer(CN):
    LAYER: float

class Scheduler(CN):
    T_INITIAL: int
    T_MULT: float
    CYCLE_DECAY: float
    CYCLE_LIMIT: int
    WARMUP_T: int
    K_DECAY: float
    LR_MIN: float
    LR_MIN_WARMUP: float

class Optimizer(CN):
    NAME: str
    LR: float
    WEIGHT_DECAY: float
    SCHEDULER: Scheduler

class Early_stop(CN):
    PATIENCE: int

class Checkpoint(CN):
    SAVE_TOP_K: int
    REMOVE: bool
    LOAD_BEST_ON_VAL: bool
    LOAD_BEST_ON_END: bool

class Callbacks(CN):
    EARLY_STOP: Early_stop
    CHECKPOINT: Checkpoint

class Trainer(CN):
    DDP: bool
    PRECISION: int
    GRADIENT_CLIP_VAL: float
    MAX_EPOCHS: int
    MAX_STEPS: int
    ACCUMULATE_GRAD_BATCHES: int
    VAL_CHECK_INTERVAL: float
    LIMIT_TRAIN_BATCHES: float
    LIMIT_VAL_BATCHES: float
    LOG_TRAIN_N_STEPS: int
    CALLBACKS: Callbacks

class Model_soup(CN):
    USE: bool
    RECIPE: str
    GREEDY_TARGET: str

class Analysis(CN):
    SAVE_NEURON_LOCATION: bool
    DRAW_NEURON_LOCATION: bool

class AutoConfig(CN):
    DESCRIPTION: str
    EXPERIMENTAL: Experimental
    DATAMODULE: Datamodule
    DATASET: Dataset
    POSITION_ENCODING: Position_encoding
    MODEL: Model
    LOSS: Loss
    REGULARIZER: Regularizer
    OPTIMIZER: Optimizer
    TRAINER: Trainer
    MODEL_SOUP: Model_soup
    RESULTS_DIR: str
    CHECKPOINT_DIR: str
    ANALYSIS: Analysis
