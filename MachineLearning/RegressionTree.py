import torch
from MachineLearning.DecisionTree import Node


class RegressionTree:
    def __init__(self, max_depth=25, min_samples_split=2, n_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.n_features = n_features
    
    """
    x.shape = (n_samples, n_features)
    y.shape = (n_samples)
    """
    def fit(self, x, y):
        self.n_features = x.shape[1] if self.n_features is None else min(x.shape[1], self.n_features)
        self.root = self._grow_tree(x, y, 0)
    
    def _fit_for_random_forest(self, x, y, i, q):
        self.n_features = x.shape[1] if self.n_features is None else min(x.shape[1], self.n_features)
        self.root = self._grow_tree(x, y, 0)
        q.put((i, self.root, self.n_features))

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
        if len(left_indicies) == 0 or len(right_indicies) == 0:
            return 0
        n = len(y)
        return torch.var(y) - len(left_indicies) / n * torch.var(y[left_indicies]) - len(right_indicies) / n * torch.var(y[right_indicies])

    def predict(self, x):
        assert self.root is not None, "DecisionTreeClassifier.fit() must be called before trying to predict"
        assert x.shape[1] == self.n_features, "DecisionTreeClassifier.fit() must be called with the same number of features"
        return torch.tensor([self._predict_single(datapoint, self.root) for datapoint in x])

    def _predict_single(self, x, current_node):
        if current_node.is_leaf():
            return current_node.value
        
        if x[current_node.feature_index] <= current_node.threshold:
            return self._predict_single(x, current_node.left)
        
        return self._predict_single(x, current_node.right)