import torch

from joblib import Parallel, delayed
import os

from ._RegressionTree import RegressionTree
from ....Exceptions import NotFittedError


class RandomForestRegressor:
    """
    RandomForestRegressor implements a regression algorithm fitting many :class:`RegressionTrees <DLL.MachineLearning.SupervisedLearning.Trees.RegressionTree>` to bootstrapped data.

    Args:
        n_trees (int, optional): The number of trees used for predicting. Defaults to 10.
        max_depth (int, optional): The maximum depth of the tree. Defaults to 25.
        min_samples_split (int, optional): The minimum required samples in a leaf to split. Defaults to 2.
        ccp_alpha (non-negative float, optional): Determines how easily subtrees are pruned in cost-complexity pruning. The larger the value, more subtrees are pruned. Defaults to 0.0.
        n_jobs (int, optional): The number of parallel workers. Defaults to -1.
    """
    def __init__(self, n_trees=10, max_depth=25, min_samples_split=2, ccp_alpha=0.0, n_jobs=-1):
        if not isinstance(n_trees, int) or n_trees < 1:
            raise ValueError("n_trees must be a positive integer.")
        if not isinstance(max_depth, int) or max_depth < 1:
            raise ValueError("max_depth must be a positive integer.")
        if not isinstance(min_samples_split, int) or min_samples_split < 1:
            raise ValueError("min_samples_split must be a positive integer.")
        if not isinstance(ccp_alpha, int | float) or ccp_alpha < 0:
            raise ValueError("ccp_alpha must be non-negative.")
        if not isinstance(n_jobs, int) or n_jobs <= -2 or n_jobs > os.cpu_count():
            raise ValueError("n_jobs must be an integer between -1 and os.cpu_count()")

        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.ccp_alpha = ccp_alpha
        self.n_jobs = n_jobs 
        self.trees = None

    def _fit_single_tree(self, X, y):
        X_boot, y_boot = self._bootstrap_sample(X, y)
        tree = RegressionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split, ccp_alpha=self.ccp_alpha)
        tree.fit(X_boot, y_boot)
        return tree

    def fit(self, X, y):
        """
        Fits the RandomForestRegressor model to the input data by generating trees, which split the data appropriately.
        """
        if not isinstance(X, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise TypeError("The input matrix and the target matrix must be a PyTorch tensor.")
        if X.ndim != 2:
            raise ValueError("The input matrix must be a 2 dimensional tensor.")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError("The target must be 1D with the same number of samples as X.")
        
        self.trees = Parallel(n_jobs=self.n_jobs)(delayed(self._fit_single_tree)(X, y) for _ in range(self.n_trees))

    def predict(self, X):
        """
        Applies the fitted RandomForestRegressor model to the input data, predicting the correct values.
        """
        if self.trees is None:
            raise NotFittedError("RandomForestRegressor.fit() must be called before predicting.")
        if not isinstance(X, torch.Tensor):
            raise TypeError("The input matrix must be a PyTorch tensor.")
        if X.ndim != 2 or X.shape[1] != self.trees[0].n_features:
            raise ValueError("The input matrix must be 2D with the correct number of features.")

        tree_preds = Parallel(n_jobs=self.n_jobs)(delayed(tree.predict)(X) for tree in self.trees)
        return torch.stack(tree_preds).mean(dim=0)
    
    def _bootstrap_sample(self, X, y):
        indices = torch.randint(high=len(y), size=(len(y),)).to(X.device)
        return X[indices], y[indices]
