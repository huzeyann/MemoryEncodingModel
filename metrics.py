import torch
from torch import Tensor


def vectorized_correlation(x: Tensor, y: Tensor) -> Tensor:
    """

    :param x: Tensor shape [num_samples, num_voxels]
    :param y: Tensor shape [num_samples, num_voxels]
    :return: shape [num_voxels, ]
    """

    dim = 0
    centered_x = x - x.mean(dim, keepdims=True)
    centered_y = y - y.mean(dim, keepdims=True)

    covariance = (centered_x * centered_y).sum(dim, keepdims=True)

    bessel_corrected_covariance = covariance / (x.shape[dim] - 1)

    x_std = x.std(dim, keepdims=True) + 1e-8
    y_std = y.std(dim, keepdims=True) + 1e-8

    corr = bessel_corrected_covariance / (x_std * y_std)

    return corr.squeeze(0)


class EpochMetric:
    def __init__(
        self,
        fn: vectorized_correlation = None,
        device=None,
    ):
        self.reset()
        self.fn = fn
        self.device = device

    def reset(self):
        self._preds = []
        self._targets = []

    def update(self, pred: Tensor, target: Tensor):
        self._preds.append(pred.detach().to(self.device))
        self._targets.append(target.detach().to(self.device))

    def compute(self):
        if self._preds[0].ndim == 1:
            preds = torch.stack(self._preds, dim=0)
        elif self._preds[0].ndim == 2 and self._preds[0].shape[0] == 1:
            preds = torch.cat(self._preds, dim=0)
        else:
            raise ValueError("preds must be 1D or 2D with shape [1, num_voxels]")

        if self._targets[0].ndim == 1:
            targets = torch.stack(self._targets, dim=0)
        elif self._targets[0].ndim == 2 and self._targets[0].shape[0] == 1:
            targets = torch.cat(self._targets, dim=0)
        else:
            raise ValueError("targets must be 1D or 2D with shape [1, num_voxels]")

        return self.fn(preds, targets)

    def __call__(self, value):
        self.update(value)
        return self.compute()
