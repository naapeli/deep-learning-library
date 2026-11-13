import torch


def _split_statistics(feature_values, statistics, candidates=None):
    """
    Calculates the statistics of the left child (the samples with feature_values <= threshold) for every candidate
    threshold of a feature at once. The feature is sorted once, after which the statistics of the left child of every
    threshold are the cumulative sums of the statistics of the sorted samples. This takes O(n log n) time for all
    thresholds together, while calculating the statistics of every threshold separately would take O(n^2) time.

    Args:
        feature_values (torch.Tensor of shape (n_samples,)): The values of the feature.
        statistics (torch.Tensor of shape (n_samples, n_statistics)): The statistics of every sample that are summed,
            for instance the targets and the squared targets.
        candidates (torch.Tensor | None): The candidate thresholds. If None, every unique value of the feature is a
            candidate, which is the same as trying every split of the samples.

    Returns:
        candidates (torch.Tensor of shape (n_candidates,)): The candidate thresholds in ascending order.
        n_left (torch.Tensor of shape (n_candidates,)): The number of samples in the left child of every threshold.
        left (torch.Tensor of shape (n_candidates, n_statistics)): The summed statistics of the left child.
        total (torch.Tensor of shape (n_statistics,)): The summed statistics of all samples.
    """
    values, order = torch.sort(feature_values)
    if candidates is None:
        candidates = torch.unique(values)
    n_left = torch.searchsorted(values, candidates.to(values.dtype).contiguous(), right=True)
    zero = torch.zeros((1, statistics.shape[1]), dtype=statistics.dtype)
    cumulative = torch.cat([zero, torch.cumsum(statistics[order], dim=0)])
    return candidates, n_left, cumulative[n_left], cumulative[-1]


def _best_threshold(candidates, gains, valid):
    """
    Returns the largest gain among the valid thresholds and its threshold, or (-inf, None) if no threshold is valid.
    Of equal gains, the smallest threshold is chosen.
    """
    gains = torch.where(valid, gains, torch.tensor(-float("inf"), dtype=gains.dtype))
    if len(gains) == 0:
        return -float("inf"), None
    index = torch.argmax(gains)  # the first of the maximal values
    return gains[index].item(), candidates[index]


def _leaves(root, X, is_leaf=lambda node: node.is_leaf()):
    """
    Routes every sample of X through the tree at once and returns the reached leaves with the indices of the samples
    in them. At every node, a single comparison of the feature column decides which samples go to the left child, so
    the number of loops is the number of nodes instead of the number of samples.

    Returns:
        list[tuple[Node, torch.Tensor]]: The leaves and the indices of the samples in each of them.
    """
    leaves = []
    stack = [(root, torch.arange(len(X)))]
    while stack:
        node, indices = stack.pop()
        if len(indices) == 0:
            continue
        if is_leaf(node):
            leaves.append((node, indices))
            continue
        goes_left = X[indices, node.feature_index] <= node.threshold
        stack.append((node.right, indices[~goes_left]))
        stack.append((node.left, indices[goes_left]))
    return leaves


def _predict_values(root, X, attribute="value", is_leaf=lambda node: node.is_leaf()):
    """
    Returns the chosen attribute of the leaf of every sample of X, for instance the predicted value or the predicted
    class probabilities.
    """
    leaves = _leaves(root, X, is_leaf)
    if len(leaves) == 0:
        return torch.tensor([])
    values = torch.stack([torch.as_tensor(getattr(leaf, attribute)) for leaf, _ in leaves])
    output = torch.empty((len(X), *values.shape[1:]), dtype=values.dtype)
    for (_, indices), value in zip(leaves, values):
        output[indices] = value
    return output
