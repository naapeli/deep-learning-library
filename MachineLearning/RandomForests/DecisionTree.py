import torch
from collections import Counter


class Node:
    def __init__(self, left=None, right=None, threshold=None, feature_index=None, value=None):
        self.left = left
        self.right = right
        self.threshold = threshold
        self.feature_index = feature_index
        self.value = value

    def is_leaf(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2, n_features=None):
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
            largest_class = Counter(y).most_common(1)[0][0]
            return Node(value=largest_class)

        feature_indicies = torch.randperm(n_features)[:self.n_features]
        split_threshold, split_index = self._best_split(x, y, feature_indicies)
        # if no split gains more information
        if split_threshold is None:
            largest_class = Counter(y).most_common(1)[0][0]
            return Node(value=largest_class)
        
        left_indicies, right_indicies = self._split(x[:, split_index], split_threshold)
        left = self._grow_tree(x[left_indicies], y[left_indicies], depth + 1)
        right = self._grow_tree(x[right_indicies], y[right_indicies], depth + 1)
        return Node(left, right, split_threshold, split_index)
    
    def _best_split(self, x, y, feature_indicies):
        max_entropy_gain = -1
        split_index = None
        split_threshold = None

        for index in feature_indicies:
            feature_values = x[:, index]
            possible_thresholds = torch.unique(feature_values)
            for threshold in possible_thresholds:
                entropy_gain = self._entropy_gain(y, feature_values, threshold)
                if entropy_gain > max_entropy_gain:
                    max_entropy_gain = entropy_gain
                    split_index = index
                    split_threshold = threshold
        if max_entropy_gain <= 0: return None, None
        return split_threshold, split_index

    def _split(self, feature_values, threshold):
        left_indicies = torch.argwhere(feature_values <= threshold).flatten()
        right_indicies = torch.argwhere(feature_values > threshold).flatten()
        return left_indicies, right_indicies

    def _entropy_gain(self, y, feature_values, threshold):
        left_indicies, right_indicies = self._split(feature_values, threshold)
        if len(left_indicies) == 0 or len(right_indicies) == 0:
            return 0
        n = len(y)
        return self._entropy(y) - len(left_indicies) / n * self._entropy(y[left_indicies]) - len(right_indicies) / n * self._entropy(y[right_indicies])
    
    def _entropy(self, values):
        n = len(values)
        data_type = values.dtype
        p = torch.bincount(values.to(dtype=torch.int32)).to(dtype=data_type) / n
        return -torch.sum(p * torch.log(p))

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
