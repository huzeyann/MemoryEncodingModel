import logging
import os
import warnings
from pathlib import Path
from typing import List

from dotenv import find_dotenv, load_dotenv
from yacs.config import CfgNode
from yacs.config import CfgNode as CN
from yacs.config import _assert_with_logging, _valid_type

from config import _C


def check_cfg(C):
    pass


def get_cfg_defaults():
    """
    Get a yacs CfgNode object with default values
    """
    # Return a clone so that the defaults will not be altered
    # It will be subsequently overwritten with local YAML.
    return _C.clone()

def save_to_yaml(cfg, path_output):
    """
    Save the current config to a YAML file.
    :param cfg: CfgNode object to be saved
    :param path_output: path to output files
    """
    path_output = Path(path_output)
    path_output.parent.mkdir(parents=True, exist_ok=True)
    with open(path_output, "w") as f:
        f.write(cfg.dump())
        
def load_from_yaml(path_cfg_data, path_cfg_override=None, list_cfg_override=None):
    """
    Load a config from a YAML file.
    :param path_cfg_data: path to path_cfg_data files
    :param path_cfg_override: path to path_cfg_override actual
    :param list_cfg_override: [key1, value1, key2, value2, ...]
    :return: cfg_base incorporating the overwrite.
    """
    cfg_base = get_cfg_defaults()
    if path_cfg_data is not None:
        cfg_base.merge_from_file(path_cfg_data)
    if path_cfg_override is not None:
        cfg_base.merge_from_file(path_cfg_override)
    if list_cfg_override is not None:
        cfg_base.merge_from_list(list_cfg_override)
    return cfg_base

def convert_to_dict(cfg_node):
    def _convert_to_dict(cfg_node, key_list):
        _VALID_TYPES = {tuple, list, str, int, float, bool, type(None)}
        if not isinstance(cfg_node, CfgNode):
            _assert_with_logging(
                _valid_type(cfg_node),
                "Key {} with value {} is not a valid type; valid types: {}".format(
                    ".".join(key_list), type(cfg_node), _VALID_TYPES
                ),
            )
            return cfg_node
        else:
            cfg_dict = dict(cfg_node)
            for k, v in cfg_dict.items():
                cfg_dict[k] = _convert_to_dict(v, key_list + [k])
            return cfg_dict

    return _convert_to_dict(cfg_node, [])


def combine_cfgs(
    path_cfg_data: Path = None,
    path_cfg_override: Path = None,
    list_cfg_override: List = None,
):
    """
    An internal facing routine thaat combined CFG in the order provided.
    :param path_output: path to output files
    :param path_cfg_data: path to path_cfg_data files
    :param path_cfg_override: path to path_cfg_override actual
    :param list_cfg_override: [key1, value1, key2, value2, ...]
    :return: cfg_base incorporating the overwrite.
    """
    if path_cfg_data is not None:
        path_cfg_data = Path(path_cfg_data)
    if path_cfg_override is not None:
        path_cfg_override = Path(path_cfg_override)
    # Path order of precedence is:
    # Priority 1, 2, 3, 4, 5 respectively
    # .env > List > other CFG YAML > data.yaml > default.yaml

    # Load default lowest tier one:
    # Priority 5:
    cfg_base = get_cfg_defaults()

    # Merge from the path_data
    # Priority 4:
    if path_cfg_data is not None and path_cfg_data.exists():
        cfg_base.merge_from_file(path_cfg_data.absolute())

    # Merge from other cfg_path files to further reduce effort
    # Priority 3:
    if path_cfg_override is not None and path_cfg_override.exists():
        cfg_base.merge_from_file(path_cfg_override.absolute())

    # Merge from List
    # Priority 2:
    if list_cfg_override is not None:
        cfg_base.merge_from_list(list_cfg_override)

    # Merge from .env
    # Priority 1:
    list_cfg = update_cfg_using_dotenv()
    if list_cfg is not []:
        cfg_base.merge_from_list(list_cfg)

    check_cfg(cfg_base)

    return cfg_base


def update_cfg_using_dotenv() -> list:
    """
    In case when there are dotenvs, try to return list of them.
    # It is returning a list of hard overwrite.
    :return: empty list or overwriting information
    """
    # If .env not found, bail
    if find_dotenv() == "":
        warnings.warn(".env files not found. YACS config file merging aborted.")
        return []

    # Load env.
    load_dotenv(find_dotenv(), verbose=True)

    # Load variables
    list_key_env = {
        "DATASET.ROOT_DIR",
        "DATASET.VOXEL_INDEX_DIR",
        "MODEL.BACKBONE.PRETRAINED_WEIGHT_DIR",
        "TRAINER.CALLBACKS.CHECKPOINT.ROOT_DIR",
        "RESULTS_DIR",
    }

    # Instantiate return list.
    path_overwrite_keys = []
    logging.info("merge from .env")
    # Go through the list of key to be overwritten.
    for key in list_key_env:

        # Get value from the env.
        value = os.getenv(key)
        logging.info(f"{key}={value}")
        # If it is none, skip. As some keys are only needed during training and others during the prediction stage.
        if value is None:
            continue

        # Otherwise, adding the key and the value to the dictionary.
        path_overwrite_keys.append(key)
        path_overwrite_keys.append(value)

    return path_overwrite_keys


# The flatten and unflatten snippets are from an internal lfads_tf2 implementation.


def flatten_dict(dictionary, level=[]):
    """Flattens a dictionary by placing '.' between levels.

    This function flattens a hierarchical dictionary by placing '.'
    between keys at various levels to create a single key for each
    value. It is used internally for converting the configuration
    dictionary to more convenient formats. Implementation was
    inspired by `this StackOverflow post
    <https://stackoverflow.com/questions/6037503/python-unflatten-dict>`_.

    Parameters
    ----------
    dictionary : dict
        The hierarchical dictionary to be flattened.
    level : str, optional
        The string to append to the beginning of this dictionary,
        enabling recursive calls. By default, an empty string.

    Returns
    -------
    dict
        The flattened dictionary.

    See Also
    --------
    lfads_tf2.utils.unflatten : Performs the opposite of this operation.

    """

    tmp_dict = {}
    for key, val in dictionary.items():
        if type(val) == dict:
            tmp_dict.update(flatten_dict(val, level + [key]))
        else:
            tmp_dict[".".join(level + [key])] = val
    return tmp_dict


def unflatten(dictionary):
    """Unflattens a dictionary by splitting keys at '.'s.

    This function unflattens a hierarchical dictionary by splitting
    its keys at '.'s. It is used internally for converting the
    configuration dictionary to more convenient formats. Implementation was
    inspired by `this StackOverflow post
    <https://stackoverflow.com/questions/6037503/python-unflatten-dict>`_.

    Parameters
    ----------
    dictionary : dict
        The flat dictionary to be unflattened.

    Returns
    -------
    dict
        The unflattened dictionary.

    See Also
    --------
    lfads_tf2.utils.flatten : Performs the opposite of this operation.

    """

    resultDict = dict()
    for key, value in dictionary.items():
        parts = key.split(".")
        d = resultDict
        for part in parts[:-1]:
            if part not in d:
                d[part] = dict()
            d = d[part]
        d[parts[-1]] = value
    return resultDict


def dict_to_list(config):
    config_list = []
    for key, val in config.items():
        # print(key, val, type(val))
        config_list.append(key)
        config_list.append(val)
    return config_list
