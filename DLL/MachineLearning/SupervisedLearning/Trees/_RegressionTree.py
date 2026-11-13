import torch

from ._DecisionTree import Node
from ._TreeUtils import _split_statistics, _best_threshold, _predict_values
from ....Exceptions import NotFittedError


class RegressionTree:
    """
    RegressionTree implements a regression algorithm splitting the data along features minimizing the variance.

    Args:
        max_depth (int, optional): The maximum depth of the tree. Defaults to 25. Must be a positive integer.
        min_samples_split (int, optional): The minimum required samples in a leaf to make a split. Defaults to 2. Must be a positive integer.
        ccp_alpha (non-negative float, optional): Determines how easily subtrees are pruned in cost-complexity pruning. The larger the value, more subtrees are pruned. Defaults to 0.0.
    """
    def __init__(self, max_depth=25, min_samples_split=2, ccp_alpha=0.0):
        if not isinstance(max_depth, int) or max_depth < 1:
            raise ValueError("max_depth must be a positive integer.")
        if not isinstance(min_samples_split, int) or min_samples_split < 1:
            raise ValueError("min_samples_split must be a positive integer.")
        if not isinstance(ccp_alpha, int | float) or ccp_alpha < 0:
            raise ValueError("ccp_alpha must be non-negative.")
        
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.ccp_alpha = ccp_alpha
        self.root = None
    
    def fit(self, X, y):
        """
        Fits the RegressionTree model to the input data by generating a tree, which splits the data appropriately.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data, where each row is a sample and each column is a feature.
            y (torch.Tensor of shape (n_samples,)): The target values corresponding to each sample.
        Returns:
            None
        Raises:
            TypeError: If the input matrix or the target matrix is not a PyTorch tensor.
            ValueError: If the input matrix or the target matrix is not the correct shape.
        """
        if not isinstance(X, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise TypeError("The input matrix and the target matrix must be a PyTorch tensor.")
        if X.ndim != 2:
            raise ValueError("The input matrix must be a 2 dimensional tensor.")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError("The target must be 1 dimensional with the same number of samples as the input data")
        self.n_features = X.shape[1]
        self.root = self._grow_tree(X, y, 0)
        if self.ccp_alpha > 0:
            self.root, _, _ = self._prune(self.root, X, y)

    def _grow_tree(self, x, y, depth):
        n_samples, n_features = x.size()
        classes = torch.unique(y)

        if depth >= self.max_depth or len(classes) == 1 or n_samples < self.min_samples_split:
            return Node(value=torch.mean(y))

        feature_indicies = torch.randperm(n_features)[:self.n_features]
        split_threshold, split_index = self._best_split(x, y, feature_indicies)
        # if no split gains more information
        if split_threshold is None:
            return Node(value=torch.mean(y))
        
        left_indicies, right_indicies = self._split(x[:, split_index], split_threshold)
        left = self._grow_tree(x[left_indicies], y[left_indicies], depth + 1)
        right = self._grow_tree(x[right_indicies], y[right_indicies], depth + 1)
        return Node(left, right, split_threshold, split_index)
    
    def _best_split(self, x, y, feature_indicies):
        max_variance_reduction = -float("inf")
        split_index = None
        split_threshold = None

        y = y.to(torch.float64)
        y = y - y.mean()  # the variance does not change, but the sums of y^2 lose less precision
        statistics = torch.stack([y, y ** 2], dim=1)
        for index in feature_indicies:
            variance_reduction, threshold = self._variance_reduction(x[:, index], statistics)
            if variance_reduction > max_variance_reduction:
                max_variance_reduction = variance_reduction
                split_index = index
                split_threshold = threshold
        if max_variance_reduction <= 0: return None, None
        return split_threshold, split_index

    def _split(self, feature_values, threshold):
        left_indicies = torch.argwhere(feature_values <= threshold).flatten()
        right_indicies = torch.argwhere(feature_values > threshold).flatten()
        return left_indicies, right_indicies

    def _variance_reduction(self, feature_values, statistics):
        # The reduction of the (unbiased) variance for every threshold at once from the sums of y and y^2 of the children.
        candidates, n_left, left, total = _split_statistics(feature_values, statistics)
        n = len(feature_values)
        n_right = n - n_left
        right = total - left

        def variance(sums, counts):
            return (sums[..., 1] - sums[..., 0] ** 2 / counts) / (counts - 1)

        n_left, n_right = n_left.to(torch.float64), n_right.to(torch.float64)
        reduction = variance(total, torch.tensor(n, dtype=torch.float64)) - n_left / n * variance(left, n_left) - n_right / n * variance(right, n_right)
        # both children must contain at least two samples for their variance to be defined
        return _best_threshold(candidates, reduction, (n_left > 1) & (n_right > 1))

    def predict(self, X):
        """
        Applies the fitted RegressionTree model to the input data, predicting the correct values.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data to be regressed.
        Returns:
            target values (torch.Tensor of shape (n_samples,)): The predicted values corresponding to each sample.
        Raises:
            NotFittedError: If the RegressionTree model has not been fitted before predicting.
            TypeError: If the input matrix is not a PyTorch tensor.
            ValueError: If the input matrix is not the correct shape.
        """
        if self.root is None:
            raise NotFittedError("RegressionTree.fit() must be called before predicting.")
        if not isinstance(X, torch.Tensor):
            raise TypeError("The input matrix must be a PyTorch tensor.")
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError("The input matrix must be a 2 dimensional tensor with the same number of features as the fitted tensor.")
        return _predict_values(self.root, X)
    
    def _prune(self, node, X, y):
        if node.is_leaf():
            return node, torch.sum((y - node.value) ** 2).item(), 1
        
        left_indicies, right_indicies = self._split(X[:, node.feature_index], node.threshold)
        node.left, left_cost, left_leaf_nodes = self._prune(node.left, X[left_indicies], y[left_indicies])
        node.right, right_cost, right_leaf_nodes = self._prune(node.right, X[right_indicies], y[right_indicies])
        subtree_cost = left_cost + right_cost
        total_subleaves = left_leaf_nodes + right_leaf_nodes

        leaf_value = torch.mean(y)
        new_leaf_nodes = 1
        cost_of_node_replacement = torch.sum((y - leaf_value) ** 2).item()

        subtree_cost_with_n_nodes_penalty = subtree_cost + self.ccp_alpha * total_subleaves
        
        if cost_of_node_replacement + self.ccp_alpha * new_leaf_nodes < subtree_cost_with_n_nodes_penalty:
            return Node(value=leaf_value), cost_of_node_replacement, 1
        return node, subtree_cost, total_subleaves
