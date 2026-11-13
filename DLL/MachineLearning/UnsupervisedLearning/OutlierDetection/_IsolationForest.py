import torch
from math import log, ceil, log2

from ...SupervisedLearning.Trees._DecisionTree import Node
from ...SupervisedLearning.Trees._TreeUtils import _predict_values
from ....Exceptions import NotFittedError


def _average_path_length(n):
    """
    The average path length of an unsuccessful search in a binary search tree of n samples, which is the expected
    remaining depth of a leaf that still contains n samples. c(n) = 2 H(n - 1) - 2 (n - 1) / n with the harmonic number
    H(i) ~ ln(i) + 0.5772156649, c(2) = 1 and c(n) = 0 for n < 2.
    """
    if n > 2:
        return 2 * (log(n - 1) + 0.5772156649) - 2 * (n - 1) / n
    return 1.0 if n == 2 else 0.0


class IsolationForest:
    """
    IsolationForest implements an algorithm to detect outliers in the data by fitting many isolation trees to random subsamples of the data. The implementation follows `the original paper <https://doi.org/10.1109/ICDM.2008.17>`_: outliers are isolated with fewer random splits than the other points, so their average path length from the root to their leaf is shorter.

    Every tree is fitted to a random subsample of max_samples points and grown until a maximum depth of :math:`\\lceil\\log_2(\\psi)\\rceil`, where :math:`\\psi` is the size of the subsample. A leaf that still contains :math:`m` points adds :math:`c(m)`, the average path length of a binary search tree of :math:`m` points, to the path length. The anomaly score of a point :math:`x` is

    .. math::

        s(x) = 2^{-\\mathbb{E}[h(x)] / c(\\psi)},

    where :math:`h(x)` is the path length of :math:`x` in a tree. The score is close to 1 for outliers and below 0.5 for normal points.

    Args:
        n_trees (int, optional): The number of trees used for predicting. Defaults to 100. Must be a positive integer.
        max_samples (int | float, optional): The number of samples drawn from the data to fit every tree. If an int, the number of samples (at most the number of samples in the data). If a float, the proportion of the data. Defaults to 256, as recommended in the paper. Must be an integer of at least 2 or a float in range (0, 1].
        max_depth (int | None, optional): The maximum depth of the trees. If None, :math:`\\lceil\\log_2(\\psi)\\rceil` is used, which is the average depth of a tree of :math:`\\psi` points. Defaults to None. Must be a positive integer or None.
        min_samples_split (int, optional): The minimum required samples in a leaf to make a split. Defaults to 2. Must be a positive integer.
        bootstrap (bool, optional): Determines if the subsamples are drawn with replacement. Must be a boolean. Defaults to False.
        threshold (int | float, optional): Determines how many standard deviations away from the mean score a datapoint must be to be considered an outlier. Must be a non-negative real number. Defaults to 4.
    Attributes:
        n_samples_tree (int): The number of samples :math:`\\psi` used to fit every tree. Available after fitting.
    """
    def __init__(self, n_trees=100, max_samples=256, max_depth=None, min_samples_split=2, bootstrap=False, threshold=4):
        if not isinstance(n_trees, int) or n_trees < 1:
            raise ValueError("n_trees must be a positive integer.")
        if isinstance(max_samples, bool) or not ((isinstance(max_samples, int) and max_samples >= 2) or (isinstance(max_samples, float) and 0 < max_samples <= 1)):
            raise ValueError("max_samples must be an integer of at least 2 or a float in range (0, 1].")
        if max_depth is not None and (not isinstance(max_depth, int) or max_depth < 1):
            raise ValueError("max_depth must be a positive integer or None.")
        if not isinstance(min_samples_split, int) or min_samples_split < 1:
            raise ValueError("min_samples_split must be a positive integer.")
        if not isinstance(bootstrap, bool):
            raise TypeError("bootstrap must be a boolean.")
        if not isinstance(threshold, int | float) or threshold < 0:
            raise ValueError("threshold must be a non-negative real number.")

        self.n_trees = n_trees
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.bootstrap = bootstrap
        self.threshold = threshold

    def fit(self, X):
        """
        Fits the IsolationForest model to the input data by fitting every tree to a random subsample of the data.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data, where each row is a sample and each column is a feature.
        Returns:
            None
        Raises:
            TypeError: If the input matrix is not a PyTorch tensor.
            ValueError: If the input matrix is not the correct shape.
        """
        if not isinstance(X, torch.Tensor):
            raise TypeError("The input matrix must be a PyTorch tensor.")
        if X.ndim != 2 or len(X) < 2:
            raise ValueError("The input matrix must be a 2 dimensional tensor with at least 2 samples.")

        n = len(X)
        self.n_samples_tree = min(self.max_samples, n) if isinstance(self.max_samples, int) else max(2, int(self.max_samples * n))
        max_depth = self.max_depth if self.max_depth is not None else ceil(log2(self.n_samples_tree))
        self.trees = []
        for _ in range(self.n_trees):
            tree = IsolationTree(max_depth, self.min_samples_split)
            tree.fit(X[self._subsample_indices(n)])
            self.trees.append(tree)

    def _subsample_indices(self, n):
        if self.bootstrap:
            return torch.randint(high=n, size=(self.n_samples_tree,))
        return torch.randperm(n)[:self.n_samples_tree]

    def predict(self, X, return_scores=False):
        """
        Predicts the outliers in the input by considering scores, which are threshold standard deviations away from the mean.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data, where each row is a sample and each column is a feature.
            return_scores (bool, optional): Determines if the anomaly scores of each datapoint are returned instead of the predictions. Defaults to False.
        Returns:
            torch.Tensor of shape (n_samples,): A boolean tensor, which is True for the outliers, or the anomaly scores if return_scores=True.
        Raises:
            NotFittedError: If the IsolationForest model has not been fitted before predicting.
            TypeError: If the input matrix is not a PyTorch tensor or return_scores is not a boolean.
            ValueError: If the input matrix is not the correct shape.
        """
        if not hasattr(self, "trees"):
            raise NotFittedError("IsolationForest.fit() must be called before predicting.")
        if not isinstance(X, torch.Tensor):
            raise TypeError("The input matrix must be a PyTorch tensor.")
        if X.ndim != 2:
            raise ValueError("The input matrix must be a 2 dimensional tensor.")
        if not isinstance(return_scores, bool):
            raise TypeError("return_scores must be a boolean.")

        path_lengths = torch.stack([tree._path_lengths(X) for tree in self.trees])
        scores = 2 ** -(torch.mean(path_lengths, dim=0) / _average_path_length(self.n_samples_tree))
        if return_scores: return scores

        threshold = torch.mean(scores) + self.threshold * torch.std(scores)
        return scores > threshold

    def fit_predict(self, X, return_scores=False):
        """
        First fits the model to the input and then predicts, which of the inputs are outliers.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data, where each row is a sample and each column is a feature.
            return_scores (bool, optional): Determines if the scores of each datapoint are returned. Defaults to False.
        """
        self.fit(X)
        return self.predict(X, return_scores=return_scores)


class IsolationTree:
    """
    IsolationTree implements an algorithm to detect outliers in the data by splitting it randomly.

    Args:
        max_depth (int, optional): The maximum depth of the tree. Defaults to 25. Must be a positive integer.
        min_samples_split (int, optional): The minimum required samples in a leaf to make a split. Defaults to 2. Must be a positive integer.
    """
    def __init__(self, max_depth=25, min_samples_split=2):
        if not isinstance(max_depth, int) or max_depth < 1:
            raise ValueError("max_depth must be a positive integer.")
        if not isinstance(min_samples_split, int) or min_samples_split < 1:
            raise ValueError("min_samples_split must be a positive integer.")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X):
        """
        Fits the IsolationTree model to the input data by generating a tree, which splits the data randomly.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data, where each row is a sample and each column is a feature.
        Returns:
            None
        Raises:
            TypeError: If the input matrix is not a PyTorch tensor.
            ValueError: If the input matrix is not the correct shape.
        """
        if not isinstance(X, torch.Tensor):
            raise TypeError("The input matrix must be a PyTorch tensor.")
        if X.ndim != 2:
            raise ValueError("The input matrix must be a 2 dimensional tensor.")
        self.n_features = X.shape[1]
        self.root = self._grow_tree(X, 0)

    def _leaf(self, n_samples, depth):
        node = Node(size=n_samples, depth=depth)
        # the samples that were not separated would on average need c(n_samples) more splits
        node.path_length = depth + _average_path_length(n_samples)
        return node

    def _grow_tree(self, x, depth):
        n_samples = len(x)

        if depth >= self.max_depth or n_samples < self.min_samples_split:
            return self._leaf(n_samples, depth)

        split_threshold, split_index = self._random_split(x)
        if split_threshold is None:
            return self._leaf(n_samples, depth)

        left_indicies, right_indicies = self._split(x[:, split_index], split_threshold)
        left = self._grow_tree(x[left_indicies], depth + 1)
        right = self._grow_tree(x[right_indicies], depth + 1)
        return Node(left, right, split_threshold, split_index)

    def _random_split(self, x):
        split_index = torch.randint(x.shape[1], size=tuple())
        feature_min, feature_max = torch.min(x[:, split_index]), torch.max(x[:, split_index])
        if feature_min == feature_max:
            return None, None

        split_threshold = (feature_max - feature_min) * torch.rand(size=tuple()) + feature_min
        return split_threshold, split_index

    def _split(self, feature_values, threshold):
        left_indicies = torch.argwhere(feature_values <= threshold).flatten()
        right_indicies = torch.argwhere(feature_values > threshold).flatten()
        return left_indicies, right_indicies

    def _path_lengths(self, X):
        # the path length of every sample, with all samples routed through the tree at once
        return _predict_values(self.root, X, attribute="path_length", is_leaf=lambda node: node.depth is not None)
