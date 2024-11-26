import torch

from .DecisionTree import Node
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
        max_gain = -1
        split_index = None
        split_threshold = None

        for index in feature_indicies:
            feature_values = X[:, index]
            possible_thresholds = torch.unique(feature_values)
            for threshold in possible_thresholds:
                gain = self._gain(gradient, hessian, feature_values, threshold)
                if gain > max_gain:
                    max_gain = gain
                    split_index = index
                    split_threshold = threshold
        if max_gain <= 0: return None, None
        return split_threshold, split_index

    def _split(self, feature_values, threshold):
        left_indicies = torch.argwhere(feature_values <= threshold).flatten()
        right_indicies = torch.argwhere(feature_values > threshold).flatten()
        return left_indicies, right_indicies

    def _gain(self, gradient, hessian, feature_values, threshold):
        left_indicies, right_indicies = self._split(feature_values, threshold)
        if len(left_indicies) <= 1 or len(right_indicies) <= 1:
            return 0
        return self._gain_node(gradient, hessian) - self._gain_node(gradient[left_indicies], hessian[left_indicies]) - self._gain_node(gradient[right_indicies], hessian[right_indicies]) - self.gamma

    def _gain_node(self, gradient, hessian):
        return -(gradient.sum() ** 2) / (hessian.sum() + self.reg_lambda)

    def predict(self, X):
        if self.root is None:
            raise NotFittedError("_XGBoostTree.fit() must be called before predicting.")
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