import torch

from ._DecisionTree import Node
from ._TreeUtils import _best_split, _predict_values
from ....Exceptions import NotFittedError


class _XGBoostTree:
    def __init__(self, max_depth=25, min_samples_split=2, reg_lambda=1, gamma=1):
        if not isinstance(max_depth, int) or max_depth < 1:
            raise ValueError("max_depth must be a positive integer.")
        if not isinstance(min_samples_split, int) or min_samples_split < 1:
            raise ValueError("min_samples_split must be a positive integer.")
        if not isinstance(reg_lambda, int | float) or reg_lambda <= 0:
            raise ValueError("reg_lambda must be a positive real number.")
        if not isinstance(gamma, int | float) or gamma < 0:
            raise ValueError("gamma must be a non-negative real number.")
        
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.reg_lambda  = reg_lambda
        self.gamma = gamma
        self.root = None
    
    def _value(self, gradient, hessian):
        return -gradient.sum() / (hessian.sum() + self.reg_lambda)
    
    def fit(self, X, gradient, hessian):
        if not isinstance(X, torch.Tensor) or not isinstance(gradient, torch.Tensor):
            raise TypeError("The input matrix and the target matrix must be a PyTorch tensor.")
        if X.ndim != 2:
            raise ValueError("The input matrix must be a 2 dimensional tensor.")
        if gradient.ndim != 1 or gradient.shape[0] != X.shape[0]:
            raise ValueError("The gradient must be 1 dimensional with the same number of samples as the input data")
        if hessian.ndim != 1 or hessian.shape[0] != X.shape[0]:
            raise ValueError("The hessian must be 1 dimensional with the same number of samples as the input data")
        
        self.n_features = X.shape[1]
        self.root = self._grow_tree(X, gradient, hessian, 0)

    def _grow_tree(self, X, gradient, hessian, depth):
        n_samples, n_features = X.size()
        vals = torch.unique(gradient)

        if depth >= self.max_depth or len(vals) == 1 or n_samples < self.min_samples_split:
            return Node(value=self._value(gradient, hessian))

        feature_indicies = torch.randperm(n_features)[:self.n_features]
        split_threshold, split_index = self._best_split(X, gradient, hessian, feature_indicies)
        # if no split gains more information
        if split_threshold is None:
            return Node(value=self._value(gradient, hessian))
        
        left_indicies, right_indicies = self._split(X[:, split_index], split_threshold)
        left = self._grow_tree(X[left_indicies], gradient[left_indicies], hessian[left_indicies], depth + 1)
        right = self._grow_tree(X[right_indicies], gradient[right_indicies], hessian[right_indicies], depth + 1)
        return Node(left, right, split_threshold, split_index)
    
    def _best_split(self, X, gradient, hessian, feature_indicies):
        statistics = torch.stack([gradient, hessian], dim=1).to(torch.float64)
        # both children must contain at least two samples
        gain, split_threshold, split_index = _best_split(X, feature_indicies, statistics, self._gain, min_child_size=2)
        if gain <= 0: return None, None
        return split_threshold, split_index

    def _split(self, feature_values, threshold):
        left_indicies = torch.argwhere(feature_values <= threshold).flatten()
        right_indicies = torch.argwhere(feature_values > threshold).flatten()
        return left_indicies, right_indicies

    def _gain(self, left, total, n_left, n):
        # The gain of every split from the sums of the gradients and the hessians of the children.
        return self._gain_node(total) - self._gain_node(left) - self._gain_node(total - left) - self.gamma

    def _gain_node(self, sums):
        # sums[..., 0] is the sum of the gradients and sums[..., 1] the sum of the hessians
        return -(sums[..., 0] ** 2) / (sums[..., 1] + self.reg_lambda)

    def predict(self, X):
        if self.root is None:
            raise NotFittedError("_XGBoostTree.fit() must be called before predicting.")
        if not isinstance(X, torch.Tensor):
            raise TypeError("The input matrix must be a PyTorch tensor.")
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError("The input matrix must be a 2 dimensional tensor with the same number of features as the fitted tensor.")
        return _predict_values(self.root, X)
