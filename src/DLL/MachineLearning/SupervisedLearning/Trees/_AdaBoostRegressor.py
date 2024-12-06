import torch
from warnings import warn

from ._RegressionTree import RegressionTree
from ....Exceptions import NotFittedError


class AdaBoostRegressor:
    """
    AdaBoostRegressor implements a regression algorithm fitting many consecutive :class:`RegressionTrees <DLL.MachineLearning.SupervisedLearning.Trees.RegressionTree>` to previously incorrectly predicted samples.

    Args:
        n_trees (int, optional): The number of trees used for predicting. Defaults to 10. Must be a positive integer.
        max_depth (int, optional): The maximum depth of the tree. Defaults to 25. Must be a positive integer.
        min_samples_split (int, optional): The minimum required samples in a leaf to make a split. Defaults to 2. Must be a positive integer.
        loss (str, optional): The loss function used. Must be in ["linear", "square", "exponential"]. Defaults to "square".
    Attributes:
        n_features (int): The number of features. Available after fitting.
    """
    def __init__(self, n_trees=10, max_depth=25, min_samples_split=2, loss="square"):
        if not isinstance(n_trees, int) or n_trees < 1:
            raise ValueError("n_trees must be a positive integer.")
        if not isinstance(max_depth, int) or max_depth < 1:
            raise ValueError("max_depth must be a positive integer.")
        if not isinstance(min_samples_split, int) or min_samples_split < 1:
            raise ValueError("min_samples_split must be a positive integer.")
        if loss not in ["linear", "square", "exponential"]:
            raise ValueError('loss must be in ["linear", "square", "exponential"].')
        
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.loss = loss
        self.trees = None

    def fit(self, X, y):
        """
        Fits the AdaBoostRegressor model to the input data by fitting trees to the errors made by previous trees.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data, where each row is a sample and each column is a feature.
            y (torch.Tensor of shape (n_samples,)): The target values corresponding to each sample.
        Returns:
            The average errors after each tree.
        Raises:
            TypeError: If the input matrix or the label vector is not a PyTorch tensor or if the problem is binary and metrics is not a list or a tuple.
            ValueError: If the input matrix or the label vector is not the correct shape or the label vector contains wrong values.
        """
        if not isinstance(X, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise TypeError("The input matrix and the label matrix must be a PyTorch tensor.")
        if X.ndim != 2:
            raise ValueError("The input matrix must be a 2 dimensional tensor.")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError("The labels must be 1 dimensional with the same number of samples as the input data")
        
        y = y.to(X.dtype)
        self.n_features = X.shape[1]

        trees = []
        weights = torch.full_like(y, 1 / len(y))
        self.confidences = []
        errors = []

        for i in range(self.n_trees):
            indices = torch.multinomial(weights, len(y), replacement=True)
            X_sample = X[indices]
            y_sample = y[indices]
            
            tree = RegressionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_sample, y_sample)
            prediction = tree.predict(X)
            max_absolute_error = torch.abs(y - prediction).max()
            if self.loss == "linear":
                error = torch.abs(y - prediction) / max_absolute_error
            elif self.loss == "square":
                error = (y - prediction) ** 2 / max_absolute_error ** 2
            elif self.loss == "exponential":
                error = 1 - torch.exp(-torch.abs(y - prediction) / max_absolute_error)
            
            average_loss = torch.sum(weights * error)
            if average_loss > 0.5:
                warn(f"The average error exceeds 0.5. The training is stopped to reduce over fitting. Only {i} trees are used.")
                break

            beta = average_loss / (1 - average_loss)
            weights = weights * beta ** (1 - error)
            weights /= weights.sum()

            self.confidences.append(torch.log(1 / beta))
            trees.append(tree)
            errors.append(average_loss)
        
        self.confidences = torch.stack(self.confidences)
        self.confidences /= self.confidences.sum()
        self.trees = trees
        return errors

    def predict(self, X, method="average"):
        """
        Applies the fitted AdaBoostRegressor model to the input data, predicting the correct values.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data to be regressed.
            method (str, optional): The method for computing the prediction. Must be one of "average" or "weighted_median". Defaults to average.

        Returns:
            values (torch.Tensor of shape (n_samples,)): The predicted values corresponding to each sample.

        Raises:
            NotFittedError: If the AdaBoostRegressor model has not been fitted before predicting.
            TypeError: If the input matrix is not a PyTorch tensor.
            ValueError: If the input matrix is not the correct shape.
        """
        if self.trees is None:
            raise NotFittedError("AdaBoostRegressor.fit() must be called before predicting.")
        if not isinstance(X, torch.Tensor):
            raise TypeError("The input matrix must be a PyTorch tensor.")
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError("The input matrix must be a 2 dimensional tensor with the same number of features as the fitted tensor.")
        if method not in ["average", "weighted_median"]:
            raise ValueError('method must be one of "average" or "weighted_median".')

        if method == "average":
            preds = torch.zeros(len(X))
            for log_beta, tree in zip(self.confidences, self.trees):
                pred = tree.predict(X)
                preds += log_beta * pred
        elif method == "weighted_median":
            tree_preds = torch.stack([tree.predict(X) for tree in self.trees]).T  # n_samples, n_trees
            indicies = torch.argsort(tree_preds, dim=1)
            accumulated = torch.cumsum(self.confidences[indicies], dim=1)
            after_middle = accumulated >= 0.5 * accumulated[:, -1]
            middle_indicies = torch.argmax(after_middle, dim=1)
            original_middle_indicies = indicies[:, middle_indicies]
            preds = tree_preds[:, original_middle_indicies]
        
        return preds
