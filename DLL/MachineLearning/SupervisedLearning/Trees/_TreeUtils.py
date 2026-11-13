import torch


def _best_split(X, feature_indicies, statistics, gain_function, min_child_size):
    """
    Finds the best split of a node over every feature and every threshold at once. All features are sorted together,
    after which the statistics of the left child of every threshold are the cumulative sums of the statistics of the
    sorted samples, so the gains of all splits come from one tensor expression. The candidate thresholds are the
    unique values of every feature, which is the same as trying every split of the samples.

    Args:
        X (torch.Tensor of shape (n_samples, n_features)): The samples of the node.
        feature_indicies (torch.Tensor of shape (n_candidate_features,)): The features that are searched, in order.
        statistics (torch.Tensor of shape (n_samples, n_statistics)): The statistics of every sample that are summed,
            for instance the targets and the squared targets.
        gain_function (Callable): Takes the summed statistics of the left children (shape (n_samples, n_candidate_features,
            n_statistics)), the summed statistics of the whole node (shape (n_statistics,)), the sizes of the left children
            (shape (n_samples, 1)) and the size of the node, and returns the gains of shape (n_samples, n_candidate_features).
        min_child_size (int): The minimum number of samples in each child.

    Returns:
        gain (float): The largest gain, or -inf if there is no valid split.
        threshold (torch.Tensor | None): The threshold of the best split. Of equal gains, the smallest threshold of the
            first feature in feature_indicies is chosen.
        index (torch.Tensor | None): The feature of the best split.
    """
    n = len(X)
    values, order = torch.sort(X[:, feature_indicies], dim=0, stable=True)
    left = torch.cumsum(statistics[order], dim=0)  # left[i, j] sums the i + 1 smallest samples of feature j
    total = left[-1, 0]
    n_left = torch.arange(1, n + 1, dtype=statistics.dtype).unsqueeze(1)
    # a threshold is a unique value, so only the last occurrence of every value is a split (ties go left together)
    last_occurrence = torch.ones_like(values, dtype=torch.bool)
    last_occurrence[:-1] = values[1:] != values[:-1]
    valid = last_occurrence & (n_left >= min_child_size) & (n - n_left >= min_child_size)
    gains = torch.where(valid, gain_function(left, total, n_left, n), torch.tensor(-float("inf"), dtype=statistics.dtype))
    feature_gains, positions = torch.max(gains, dim=0)  # the first (smallest) threshold of the largest gain
    feature = torch.argmax(feature_gains)  # the first feature of the largest gain
    gain = feature_gains[feature].item()
    if gain == -float("inf"):
        return gain, None, None
    return gain, values[positions[feature], feature], feature_indicies[feature]


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
