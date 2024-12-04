import torch

from ._DecisionTree import Node
from ....Exceptions import NotFittedError


class RegressionTree:
    """
    RegressionTree implements a regression algorithm splitting the data along features minimizing the variance.

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
    
    """
    x.shape = (n_samples, n_features)
    y.shape = (n_samples)
    """
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
        max_variance_reduction = -1
        split_index = None
        split_threshold = None

        for index in feature_indicies:
            feature_values = x[:, index]
            possible_thresholds = torch.unique(feature_values)
            for threshold in possible_thresholds:
                variance_reduction = self._variance_reduction(y, feature_values, threshold)
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

    def _variance_reduction(self, y, feature_values, threshold):
        left_indicies, right_indicies = self._split(feature_values, threshold)
        if len(left_indicies) <= 1 or len(right_indicies) <= 1:
            return 0
        n = len(y)
        return torch.var(y) - len(left_indicies) / n * torch.var(y[left_indicies]) - len(right_indicies) / n * torch.var(y[right_indicies])

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
        return torch.tensor([self._predict_single(datapoint, self.root) for datapoint in X])

    def _predict_single(self, x, current_node):
        if current_node.is_leaf():
            return current_node.value
        
        if x[current_node.feature_index] <= current_node.threshold:
            return self._predict_single(x, current_node.left)
        
        return self._predict_single(x, current_node.right)