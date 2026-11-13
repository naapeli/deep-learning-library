import torch

from ._TreeUtils import _leaves


def _refit_leaves_to_quantile(root, X, residual, quantile):
    """
    Sets the value of every leaf to the chosen quantile of the residuals of the samples in the leaf. The gradients of
    the quantile loss are only -quantile or 1 - quantile, so the leaf values fitted to them are poor steps. The
    quantile of the residuals is the exact minimiser of the quantile loss in every leaf, which is also what
    scikit-learn and LightGBM use.
    """
    for leaf, indices in _leaves(root, X):
        leaf.value = torch.quantile(residual[indices], quantile)
