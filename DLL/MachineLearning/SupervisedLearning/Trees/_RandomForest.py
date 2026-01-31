import torch

from joblib import Parallel, delayed
import os

from ._DecisionTree import DecisionTree
from ....Exceptions import NotFittedError


class RandomForestClassifier:
    """
    RandomForestClassifier implements a classification algorithm fitting many :class:`DecisionTrees <DLL.MachineLearning.SupervisedLearning.Trees.DecisionTree>` to bootstrapped data.

    Args:
        n_trees (int, optional): The number of trees used for predicting. Defaults to 10. Must be a positive integer.
        max_depth (int, optional): The maximum depth of the tree. Defaults to 10. Must be a positive integer.
        min_samples_split (int, optional): The minimum required samples in a leaf to make a split. Defaults to 2. Must be a positive integer.
        n_jobs (int, optional): The number of parallel workers used to fit the model.
    """
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, criterion="gini", ccp_alpha=0, n_jobs=-1):
        if not isinstance(n_trees, int) or n_trees < 1:
            raise ValueError("n_trees must be a positive integer.")
        if not isinstance(max_depth, int) or max_depth < 1:
            raise ValueError("max_depth must be a positive integer.")
        if not isinstance(min_samples_split, int) or min_samples_split < 1:
            raise ValueError("min_samples_split must be a positive integer.")
        if not isinstance(n_jobs, int) or n_jobs <= -2 or n_jobs > os.cpu_count():
            raise ValueError("n_jobs must be an integer between -1 and os.cpu_count()")
        
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.ccp_alpha = ccp_alpha
        self.n_jobs = n_jobs 
        self.trees = None

    def _fit_single_tree(self, X, y):
        X_boot, y_boot = self._bootstrap_sample(X, y)
        tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
        tree.fit(X_boot, y_boot)
        return tree

    def fit(self, X, y):
        """
        Fits the RandomForestClassifier model to the input data by generating trees, which split the data appropriately.

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
        
        self.trees = Parallel(n_jobs=self.n_jobs)(delayed(self._fit_single_tree)(X, y) for _ in range(self.n_trees))

    def predict(self, X):
        """
        Applies the fitted RandomForestClassifier model to the input data, predicting the correct classes.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data to be classified.
        Returns:
            labels (torch.Tensor of shape (n_samples,)): The predicted labels corresponding to each sample.
        Raises:
            NotFittedError: If the RandomForestClassifier model has not been fitted before predicting.
            TypeError: If the input matrix is not a PyTorch tensor.
            ValueError: If the input matrix is not the correct shape.
        """
        if self.trees is None:
            raise NotFittedError("RandomForestClassifier.fit() must be called before predicting.")
        if not isinstance(X, torch.Tensor):
            raise TypeError("The input matrix must be a PyTorch tensor.")
        if X.ndim != 2 or X.shape[1] != self.trees[0].n_features:
            raise ValueError("The input matrix must be a 2 dimensional tensor with the same number of features as the fitted tensor.")

        tree_preds = Parallel(n_jobs=self.n_jobs)(delayed(tree.predict)(X) for tree in self.trees)
        predictions = torch.stack(tree_preds).T
        final_preds = torch.mode(predictions, dim=1).values
        return final_preds
    
    def predict_proba(self, X):
        """
        Applies the fitted RandomForestClassifier model to the input data, predicting the probabilities of each class. Is calculated as the average of each individual trees predicted probabilities.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data to be classified.
        Returns:
            probabilities (torch.Tensor of shape (n_samples, n_classes)): The predicted probabilities corresponding to each sample.
        Raises:
            NotFittedError: If the RandomForestClassifier model has not been fitted before predicting.
            TypeError: If the input matrix is not a PyTorch tensor.
            ValueError: If the input matrix is not the correct shape.
        """
        if self.trees is None:
            raise NotFittedError("RandomForestClassifier.fit() must be called before predicting.")
        if not isinstance(X, torch.Tensor):
            raise TypeError("The input matrix must be a PyTorch tensor.")
        if X.ndim != 2 or X.shape[1] != self.trees[0].n_features:
            raise ValueError("The input matrix must be a 2 dimensional tensor with the same number of features as the fitted tensor.")

        tree_probs = Parallel(n_jobs=self.n_jobs)(delayed(tree.predict_proba)(X) for tree in self.trees)
        return torch.stack(tree_probs).mean(dim=0)

    def _bootstrap_sample(self, X, y):
        indices = torch.randint(high=len(y), size=(len(y), 1)).flatten()
        return X[indices], y[indices]
