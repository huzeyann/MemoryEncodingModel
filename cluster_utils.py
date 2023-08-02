from pathlib import Path
from typing import Dict
from ray import tune
from filelock import Timeout, FileLock
import os
import logging

# monkey patching
import ray
from ray.tune.experiment import Trial


def my_create_unique_logdir_name(root: str, relative_logdir: str) -> str:
    candidate = Path(root).expanduser().joinpath(relative_logdir)
    if candidate.exists():
        # relative_logdir_old = relative_logdir
        # relative_logdir += "_" + uuid.uuid4().hex[:4]
        # logger.info(
        #     f"Creating a new dirname {relative_logdir} because "
        #     f"trial dirname '{relative_logdir_old}' already exists."
        # )
        pass
    return relative_logdir


ray.tune.experiment.trial._create_unique_logdir_name = my_create_unique_logdir_name


def my_nfs_cluster_job(func):
    def inner(*args, **kwargs):
        log_dir = tune.get_trial_dir()
        done_path = os.path.join(log_dir, "done")
        if os.path.exists(done_path):
            logging.warning(f"Experiment {log_dir} already done, skipping.")
            return
        lock_path = os.path.join(log_dir, "lockfile")
        lock = FileLock(lock_path, timeout=1)
        try:
            with lock.acquire(timeout=1):
                func(*args, **kwargs)
                with open(done_path, "w") as f:
                    f.write("done")
        except Timeout:
            import sys
            sys.tracebacklimit = -1

            raise RuntimeError(
                "Failed to acquire lock, another process is running this experiment."
            )

    return inner


def trial_dirname_creator(trial : Trial):
    config : Dict = trial.config
    # config_str = "_".join([f"{k}={'_'.join(v)}" for k, v in config.items()])
    config_str = ""
    for k, v in config.items():
        if isinstance(v, list):
            v = [str(x) for x in v]
            config_str += f"{k}={'_'.join(v)},"
        else:
            config_str += f"{k}={v},"
    max_len = 60
    config_str = config_str[:max_len]
    return f"t{trial.trial_id}_{config_str}"

def trail_name_creator(trial):
    return f"t{trial.trial_id}"

# def trial_name_creator(trial):
# return f"t"
