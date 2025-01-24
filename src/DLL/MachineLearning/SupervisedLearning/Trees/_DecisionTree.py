import torch

from ....Exceptions import NotFittedError


class Node:
    def __init__(self, left=None, right=None, threshold=None, feature_index=None, value=None, probabilities=None):
        self.left = left
        self.right = right
        self.threshold = threshold
        self.feature_index = feature_index
        self.value = value
        self.probabilities = probabilities

    def is_leaf(self):
        return self.value is not None


class DecisionTree:
    """
    DecisionTree implements a classification algorithm splitting the data along features yielding the maximum entropy.

    Args:
        max_depth (int, optional): The maximum depth of the tree. Defaults to 10. Must be a positive integer.
        min_samples_split (int, optional): The minimum required samples in a leaf to make a split. Defaults to 2. Must be a positive integer.
        criterion (str, optional): The information criterion used to select optimal splits. Must be one of "entropy" or "gini". Defaults to "gini".
        ccp_alpha (non-negative float, optional): Determines how easily subtrees are pruned in cost-complexity pruning. The larger the value, more subtrees are pruned. Defaults to 0.0.
    Attributes:
        n_classes (int): The number of classes. A positive integer available after calling DecisionTree.fit().
        classes (torch.Tensor of shape (n_classes,)): The classes in an arbitrary order. Available after calling DecisionTree.fit().
    """
    def __init__(self, max_depth=10, min_samples_split=2, criterion="gini", ccp_alpha=0.0):
        if not isinstance(max_depth, int) or max_depth < 1:
            raise ValueError("max_depth must be a positive integer.")
        if not isinstance(min_samples_split, int) or min_samples_split < 1:
            raise ValueError("min_samples_split must be a positive integer.")
        if criterion not in ["entropy", "gini"]:
            raise ValueError('The chosen criterion must be one of "entropy" or "gini".')
        if not isinstance(ccp_alpha, int | float) or ccp_alpha < 0:
            raise ValueError("ccp_alpha must be non-negative.")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None
        self.ccp_alpha = ccp_alpha
    
    def fit(self, X, y):
        """
        Fits the DecisionTree model to the input data by generating a tree, which splits the data appropriately.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data, where each row is a sample and each column is a feature.
            y (torch.Tensor of shape (n_samples,)): The labels corresponding to each sample.
        Returns:
            None
        Raises:
            TypeError: If the input matrix or the label matrix is not a PyTorch tensor.
            ValueError: If the input matrix or the label matrix is not the correct shape.
        """
        if not isinstance(X, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise TypeError("The input matrix and the label matrix must be a PyTorch tensor.")
        if X.ndim != 2:
            raise ValueError("The input matrix must be a 2 dimensional tensor.")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError("The labels must be 1 dimensional with the same number of samples as the input data")
        self.n_samples, self.n_features = X.shape
        self.classes = torch.unique(y)
        self.n_classes = len(self.classes)
        self.root = self._grow_tree(X, y, 0)
        if self.ccp_alpha > 0:
            self.root, _, _ = self._prune(self.root, X, y)

    def _grow_tree(self, x, y, depth):
        n_samples, n_features = x.size()
        classes, counts = torch.unique(y, return_counts=True)

        if depth >= self.max_depth or len(classes) == 1 or n_samples < self.min_samples_split:
            largest_class = classes[torch.argmax(counts)]
            probabilities = torch.tensor([(counts[torch.where(classes == _class)[0]] / len(y) if _class in classes else 0) for _class in self.classes])
            return Node(value=largest_class, probabilities=probabilities)

        feature_indicies = torch.randperm(n_features)[:self.n_features]
        split_threshold, split_index = self._best_split(x, y, feature_indicies)
        # if no split gains more information
        if split_threshold is None:
            largest_class = classes[torch.argmax(counts)]
            probabilities = torch.tensor([(counts[torch.where(classes == _class)[0]] / len(y) if _class in classes else 0) for _class in self.classes])
            return Node(value=largest_class, probabilities=probabilities)
        
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
                entropy_gain = self._information_gain(y, feature_values, threshold)
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

    def _information_gain(self, y, feature_values, threshold):
        if self.criterion == "gini":
            information_func = self._gini
        elif self.criterion == "entropy":
            information_func = self._entropy

        left_indicies, right_indicies = self._split(feature_values, threshold)
        if len(left_indicies) == 0 or len(right_indicies) == 0:
            return 0
        n = len(y)
        return information_func(y) - len(left_indicies) / n * information_func(y[left_indicies]) - len(right_indicies) / n * information_func(y[right_indicies])
    
    def _entropy(self, values):
        n = len(values)
        data_type = values.dtype
        p = torch.bincount(values.to(dtype=torch.int32)).to(dtype=data_type) / n
        p = p[p != 0]  # remove non-zero classes from messing up the calculations
        return -torch.sum(p * torch.log(p))
    
    def _gini(self, values):
        n = len(values)
        data_type = values.dtype
        p = torch.bincount(values.to(dtype=torch.int32)).to(dtype=data_type) / n
        return torch.sum(p * (1 - p))

    def predict(self, X):
        """
        Applies the fitted DecisionTree model to the input data, predicting the correct classes.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data to be classified.
        Returns:
            labels (torch.Tensor of shape (n_samples,)): The predicted labels corresponding to each sample.
        Raises:
            NotFittedError: If the DecisionTree model has not been fitted before predicting.
            TypeError: If the input matrix is not a PyTorch tensor.
            ValueError: If the input matrix is not the correct shape.
        """
        if self.root is None:
            raise NotFittedError("DecisionTree.fit() must be called before predicting.")
        if not isinstance(X, torch.Tensor):
            raise TypeError("The input matrix must be a PyTorch tensor.")
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError("The input matrix must be a 2 dimensional tensor with the same number of features as the fitted tensor.")
        return torch.tensor([self._predict_single(datapoint, self.root) for datapoint in X])

    def _predict_single(self, x, current_node, proba=False):
        if current_node.is_leaf():
            if proba:
                return current_node.probabilities
            return current_node.value
        
        if x[current_node.feature_index] <= current_node.threshold:
            return self._predict_single(x, current_node.left, proba=proba)
        
        return self._predict_single(x, current_node.right, proba=proba)
    
    def predict_proba(self, X):
        """
        Applies the fitted DecisionTree model to the input data, predicting the probabilities of each class.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data to be classified.
        Returns:
            probabilities (torch.Tensor of shape (n_samples, n_classes)): The predicted probabilities corresponding to each sample.
        Raises:
            NotFittedError: If the DecisionTree model has not been fitted before predicting.
            TypeError: If the input matrix is not a PyTorch tensor.
            ValueError: If the input matrix is not the correct shape.
        """
        if self.root is None:
            raise NotFittedError("DecisionTree.fit() must be called before predicting.")
        if not isinstance(X, torch.Tensor):
            raise TypeError("The input matrix must be a PyTorch tensor.")
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError("The input matrix must be a 2 dimensional tensor with the same number of features as the fitted tensor.")
        return torch.stack([self._predict_single(datapoint, self.root, proba=True) for datapoint in X])

    def _prune(self, node, X, y):
        classes, counts = torch.unique(y, return_counts=True)
        m = len(y) - torch.min(counts).item()
        if node.is_leaf():
            return node, m / self.n_samples, 1
        
        left_indicies, right_indicies = self._split(X[:, node.feature_index], node.threshold)
        node.left, left_cost, left_leaf_nodes = self._prune(node.left, X[left_indicies], y[left_indicies])
        node.right, right_cost, right_leaf_nodes = self._prune(node.right, X[right_indicies], y[right_indicies])
        subtree_cost = left_cost + right_cost
        total_subleaves = left_leaf_nodes + right_leaf_nodes

        leaf_miss_classification_error_rate = m / self.n_samples
        new_leaf_nodes = 1

        subtree_cost_with_n_nodes_penalty = subtree_cost + self.ccp_alpha * total_subleaves
        
        if leaf_miss_classification_error_rate + self.ccp_alpha * new_leaf_nodes < subtree_cost_with_n_nodes_penalty:
            largest_class = classes[torch.argmax(counts)]
            probabilities = torch.tensor([(counts[torch.where(classes == _class)[0]] / len(y) if _class in classes else 0) for _class in self.classes])
            return Node(value=largest_class, probabilities=probabilities), leaf_miss_classification_error_rate, 1
        return node, subtree_cost, total_subleaves
