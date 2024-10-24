import torch
from collections import Counter

from .RegressionTree import RegressionTree
from ....Exceptions import NotFittedError


class RandomForestRegressor:
    """
    RandomForestRegressor implements a regression algorithm fitting many :class:`RegressionTrees <DLL.MachineLearning.SupervisedLearning.Trees.RegressionTree>` to bootstrapped data.

    Args:
        n_trees (int, optional): The number of trees used for predictiong. Defaults to 10. Must be a positive integer.
        max_depth (int, optional): The maximum depth of the tree. Defaults to 10. Must be a positive integer.
        min_samples_split (int, optional): The minimum required samples in a leaf to make a split. Defaults to 2. Must be a positive integer.
    """
    def __init__(self, n_trees=10, max_depth=25, min_samples_split=2):
        if not isinstance(n_trees, int) or n_trees < 1:
            raise ValueError("n_trees must be a positive integer.")
        if not isinstance(max_depth, int) or max_depth < 1:
            raise ValueError("max_depth must be a positive integer.")
        if not isinstance(min_samples_split, int) or min_samples_split < 1:
            raise ValueError("min_samples_split must be a positive integer.")
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = [RegressionTree(max_depth=max_depth, min_samples_split=min_samples_split) for _ in range(n_trees)]

    def fit(self, X, y):
        """
        Fits the RandomForestRegressor model to the input data by generating trees, which split the data appropriately.

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
        for tree in self.trees:
            tree.fit(*self._bootstrap_sample(X, y))

    def predict(self, X):
        """
        Applies the fitted RandomForestRegressor model to the input data, predicting the correct values.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data to be regressed.
        Returns:
            labels (torch.Tensor of shape (n_samples,)): The predicted target values corresponding to each sample.
        Raises:
            NotFittedError: If the RandomForestRegressor model has not been fitted before predicting.
            TypeError: If the input matrix is not a PyTorch tensor.
            ValueError: If the input matrix is not the correct shape.
        """
        if self.trees is None:
            raise NotFittedError("RandomForestRegressor.fit() must be called before predicting.")
        if not isinstance(X, torch.Tensor):
            raise TypeError("The input matrix must be a PyTorch tensor.")
        if X.ndim != 2 or X.shape[1] != self.trees[0].n_features:
            raise ValueError("The input matrix must be a 2 dimensional tensor with the same number of features as the fitted tensor.")
        predictions = torch.stack([tree.predict(X) for tree in self.trees]).T
        return torch.tensor([Counter(sample_prediction).most_common(1)[0][0] for sample_prediction in predictions])
    
    def _bootstrap_sample(self, x, y):
        indices = torch.randint(high=len(y), size=(len(y), 1)).flatten()
        return x[indices], y[indices]
