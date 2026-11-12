import torch


def _find_leaf(node, x):
    while not node.is_leaf():
        node = node.left if x[node.feature_index] <= node.threshold else node.right
    return node


def _refit_leaves_to_quantile(root, X, residual, quantile):
    """
    Sets the value of every leaf to the chosen quantile of the residuals of the samples in the leaf. The gradients of
    the quantile loss are only -quantile or 1 - quantile, so the leaf values fitted to them are poor steps. The
    quantile of the residuals is the exact minimiser of the quantile loss in every leaf, which is also what
    scikit-learn and LightGBM use.
    """
    leaves = {}
    for i, x in enumerate(X):
        leaf = _find_leaf(root, x)
        leaves.setdefault(id(leaf), (leaf, []))[1].append(i)
    for leaf, indices in leaves.values():
        leaf.value = torch.quantile(residual[indices], quantile)
