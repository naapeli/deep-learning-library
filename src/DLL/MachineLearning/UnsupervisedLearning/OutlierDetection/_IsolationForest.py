import torch
from math import log

from ...SupervisedLearning.Trees._DecisionTree import Node
from ....Exceptions import NotFittedError


class IsolationForest:
    """
    IsolationForest implements an algorithm to detect outliers in the data by fitting may isolation trees to the data.

    Args:
        n_trees (int, optional): The number of trees used for predictiong. Defaults to 10. Must be a positive integer.
        max_depth (int, optional): The maximum depth of the tree. Defaults to 10. Must be a positive integer.
        min_samples_split (int, optional): The minimum required samples in a leaf to make a split. Defaults to 2. Must be a positive integer.
        bootstrap (bool, optional): Determines if the samples for fitting are boostrapped from the given data. Must be a boolean. Defaults to False.
        threshold (int | float, optional): Determines how many standard deviations away from the mean score a datapoint must be to be considered an outlier. Must be a non-ngeative real number. Defaults to 4.
    """
    def __init__(self, n_trees=10, max_depth=25, min_samples_split=2, bootstrap=False, threshold=4):
        if not isinstance(n_trees, int) or n_trees < 1:
            raise ValueError("n_trees must be a positive integer.")
        if not isinstance(max_depth, int) or max_depth < 1:
            raise ValueError("max_depth must be a positive integer.")
        if not isinstance(min_samples_split, int) or min_samples_split < 1:
            raise ValueError("min_samples_split must be a positive integer.")
        if not isinstance(bootstrap, bool):
            raise TypeError("bootstrap must be a boolean.")
        if not isinstance(threshold, int | float) or threshold < 0:
            raise ValueError("threshold must be a non-negative real number.")
        
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.bootstrap = bootstrap
        self.threshold = threshold
        self.trees = [IsolationTree(max_depth, min_samples_split) for _ in range(n_trees)]
    
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

        for tree in self.trees:
            _X = self._bootstrap_sample(X) if self.bootstrap else X
            tree.fit(_X)
    
    def _bootstrap_sample(self, x):
        indices = torch.randint(high=len(x), size=(len(x),))
        return x[indices]
    
    def predict(self, X, return_scores=False):
        """
        Predicts the outliers in the input by considering scores, which are threshold standard deviations away from the mean.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data, where each row is a sample and each column is a feature.
            return_scores (bool, optional): Determines if the scores of each datapoint are returned. Defaults to False.
        """
        if not isinstance(X, torch.Tensor):
            raise TypeError("The input matrix must be a PyTorch tensor.")
        if X.ndim != 2:
            raise ValueError("The input matrix must be a 2 dimensional tensor.")
        if not isinstance(return_scores, bool):
            raise TypeError("return_scores must be a boolean.")

        scores = torch.zeros((self.n_trees, len(X)))
        for i, tree in enumerate(self.trees):
            tree_scores = torch.tensor([tree._path_length(point, tree.root) for point in X])
            scores[i] = tree_scores
        scores = 2 ** -(torch.mean(scores, dim=0) / self._c(len(X)))
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

    def _c(self, n):
        return 2 * (log(n - 1) + 0.5772156649) - (2 * (n - 1) / n)


class IsolationTree:
    """
    IsolationTree implements an algorithm to detect outliers in the data.

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

    def _grow_tree(self, x, depth):
        self.n_samples, n_features = x.size()

        if depth >= self.max_depth or self.n_samples < self.min_samples_split:
            return Node(size=self.n_samples, depth=depth)

        split_threshold, split_index = self._random_split(x)
        if split_threshold is None:
            return Node(size=self.n_samples, depth=depth)
        
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

    def _path_length(self, point, current_node):
        if current_node.depth is not None:
            return current_node.depth
        
        if point[current_node.feature_index] <= current_node.threshold:
            return self._path_length(point, current_node.left)
        return self._path_length(point, current_node.right)
