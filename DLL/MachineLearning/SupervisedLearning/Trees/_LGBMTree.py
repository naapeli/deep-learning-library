import torch
import multiprocessing

from ._DecisionTree import Node
from ._TreeUtils import _split_statistics, _best_threshold, _predict_values
from ....Exceptions import NotFittedError
from ....Data.Preprocessing import OneHotEncoder


class _LGBMTree:
    def __init__(self, max_depth=25, min_samples_split=2, n_bins=30, reg_lambda=1, gamma=1):
        if not isinstance(max_depth, int) or max_depth < 1:
            raise ValueError("max_depth must be a positive integer.")
        if not isinstance(min_samples_split, int) or min_samples_split < 1:
            raise ValueError("min_samples_split must be a positive integer.")
        if not isinstance(reg_lambda, int | float) or reg_lambda <= 0:
            raise ValueError("reg_lambda must be a positive real number.")
        if not isinstance(n_bins, int) or n_bins <= 1:
            raise ValueError("n_bins must be an integer greater than 1.")
        if not isinstance(gamma, int | float) or gamma < 0:
            raise ValueError("gamma must be a non-negative real number.")
        
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.reg_lambda  = reg_lambda
        self.gamma = gamma
        self.n_bins = n_bins
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
        max_gain = -float("inf")
        split_index = None
        split_threshold = None

        statistics = torch.stack([gradient, hessian], dim=1).to(torch.float64)
        for index in feature_indicies:
            feature_values = X[:, index]
            possible_thresholds = torch.unique(feature_values)
            if len(possible_thresholds) > self.n_bins: _, possible_thresholds = torch.histogram(feature_values, bins=self.n_bins, weight=gradient)
            gain, threshold = self._gain(feature_values, statistics, possible_thresholds)
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

    def _gain(self, feature_values, statistics, candidates):
        # The gain of every candidate threshold at once from the sums of the gradients and the hessians of the children.
        candidates, n_left, left, total = _split_statistics(feature_values, statistics, candidates)
        right = total - left
        gain = self._gain_node(total) - self._gain_node(left) - self._gain_node(right) - self.gamma
        # both children must contain at least two samples
        return _best_threshold(candidates, gain, (n_left > 1) & (len(feature_values) - n_left > 1))

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


# _merge_features and _greedy_bundling functions based on https://github.com/meanxai/machine_learning/blob/main/12.LGBM/5.efb_onehot.py
class _ExclusiveFeatureBundling:
    def __init__(self, max_conflict_rate=0.0, n_bins=30):
        self.max_conflict_rate = max_conflict_rate
        self.n_bins = n_bins
        self.bundled_features = []
        self.mapping = {}
        self.non_sparse_features = None

    def fit(self, X):
        self.mins = torch.min(X, dim=0).values
        self.maxs = torch.max(X, dim=0).values
        X = self._discretize(X)

        self.bundles = self._greedy_bundling(X)
        result = self._merge_bundles(X, self.bundles)
        return result

    def _merge_bundles(self, X, bundles):
        result = torch.zeros(len(X), len(bundles))
        for i, bundle in enumerate(bundles):
            combined_features = torch.stack([X[:, i] for i in bundle])
            new_feature, _ = self._merge_features(combined_features)
            result[:, i] = new_feature - 1
        return result
    
    def _merge_features(self, combined_features):
        bin_ranges = [0]
        total_bin = 0
        for feature in combined_features:
            total_bin += torch.max(feature)
            bin_ranges.append(total_bin)

        new_feature = combined_features[0]
        for i in range(len(combined_features)):
            for j in range(1, len(combined_features)):
                if combined_features[j][i] != 0:
                    new_feature[i] = combined_features[j][i] + bin_ranges[j]
        return new_feature, bin_ranges

    def _greedy_bundling(self, X):
        n_features = X.shape[1]
        conflictCnt = torch.zeros((n_features, n_features))
        
        for i in range(n_features):
            for j in range(i+1, n_features):
                n_conflicts = torch.sum(X[:, i] * X[:, j] > 0)
                conflictCnt[i, j] = n_conflicts
                conflictCnt[j, i] = n_conflicts

        degree = conflictCnt.sum(axis=0)
        searchOrder = torch.argsort(degree, descending=True)
        
        bundles = []
        bundlesConflict = []
        for i in searchOrder:
            needNew = True
            for j in range(len(bundles)):
                cnt = conflictCnt[bundles[j][-1], i]
                if cnt + bundlesConflict[j] <= self.max_conflict_rate * len(X):  # K from the paper but different parametrisation
                    bundles[j].append(i)
                    bundlesConflict[j] += cnt     
                    needNew = False
                    break
            if needNew:
                bundles.append([i])
                bundlesConflict.append(0)
        return bundles
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def transform(self, X):
        X = self._discretize(X)
        return self._merge_bundles(X, self.bundles)

    def _discretize(self, X):
        X_discrete = torch.zeros_like(X)
        for i in range(X.shape[1]):
            bins = torch.linspace(self.mins[i], self.maxs[i], self.n_bins)
            descrete = torch.bucketize(X[:, i].contiguous(), bins)
            X_discrete[:, i] = torch.clamp(descrete, 0, self.n_bins - 1)  # handle the possible values outside the bounds during transformation phase
        return X_discrete
