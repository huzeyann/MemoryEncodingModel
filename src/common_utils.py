from itertools import chain, combinations
from torch import Tensor
import torch

def count_nan(x: Tensor):
    count = torch.sum(torch.isnan(x))
    percent = count / x.numel()
    return count, percent

def all_subsets(ss):
    """
    Returns all non-empty subsets of a set.
    """
    return list(chain(*map(lambda x: combinations(ss, x), range(1, len(ss) + 1))))


def subsets(
    ss,
    exclude: list = [
        0,
    ],
):
    """
    Returns all non-empty subsets of a set.
    """
    return list(
        chain(
            *map(
                lambda x: combinations(ss, x),
                [i for i in range(0, len(ss) + 1) if i not in exclude],
            )
        )
    )


if __name__ == "__main__":
    subjects = ["E1", "E2", "E3", "M1", "M2"]
    full_ss = subsets(subjects)
    ss = subsets(subjects, exclude=[0, 3])
    print(len(full_ss))
    pass
