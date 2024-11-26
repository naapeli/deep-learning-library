import torch

from ._XGBoostTree import _XGBoostTree
from ....DeepLearning.Losses import mse, mae, Huber
from ....Exceptions import NotFittedError
from ....Data.Metrics import calculate_metrics


class XGBoostingRegressor:
    """
    XGBoostingRegressor implements a regression algorithm fitting many consecutive trees to gradients and hessians of the loss function.

    Args:
        n_trees (int, optional): The number of trees used for predicting. Defaults to 50. Must be a positive integer.
        learning_rate (float, optional): The number multiplied to each additional trees residuals. Must be a real number in range (0, 1). Defaults to 0.5.
        max_depth (int, optional): The maximum depth of the tree. Defaults to 3. Must be a positive integer.
        min_samples_split (int, optional): The minimum required samples in a leaf to make a split. Defaults to 2. Must be a positive integer.
        reg_lambda (float | int, optional): The regularisation parameter used in fitting the trees. The larger the parameter, the smaller the trees. Must be a positive real number. Defaults to 1.
        gamma (float | int, optional): The minimum gain to make a split. Must be a non-negative real number. Defaults to 0.
        loss (string, optional): The loss function used in calculations of the gradients and hessians. Must be one of mse, mae, Huber. Defaults to mse.
    Attributes:
        n_features (int): The number of features. Available after fitting.
    """
    def __init__(self, n_trees=50, learning_rate=0.5, max_depth=3, min_samples_split=2, reg_lambda=1, gamma=0, loss=mse(reduction="sum")):
        if not isinstance(n_trees, int) or n_trees < 1:
            raise ValueError("n_trees must be a positive integer.")
        if not isinstance(learning_rate, float) or learning_rate <= 0 or learning_rate >= 1:
            raise ValueError("learning_rate must be a float in range (0, 1).")
        if not isinstance(max_depth, int) or max_depth < 1:
            raise ValueError("max_depth must be a positive integer.")
        if not isinstance(min_samples_split, int) or min_samples_split < 1:
            raise ValueError("min_samples_split must be a positive integer.")
        if not isinstance(reg_lambda, int | float) or reg_lambda <= 0:
            raise ValueError("reg_lambda must be a positive real number.")
        if not isinstance(gamma, int | float) or gamma < 0:
            raise ValueError("gamma must be a non-negative real number.")
        if not isinstance(loss, mse) and not isinstance(loss, mae) and not isinstance(loss, Huber):
            raise ValueError("loss must be one of mse, mae, Huber.")

        self.n_trees = n_trees
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.trees = None
        self.loss = loss

    def fit(self, X, y, metrics=["loss"]):
        """
        Fits the XGBoostingRegressor model to the input data by fitting trees to the errors made by previous trees.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data, where each row is a sample and each column is a feature.
            y (torch.Tensor of shape (n_samples,)): The target values corresponding to each sample.
            metrics (dict[str, torch.Tensor]): Contains the metrics that will be calculated between fitting each tree and returned.
        Returns:
            metrics (dict[str, torch.Tensor]): The calculated metrics.
        Raises:
            TypeError: If the input matrix or the target vector is not a PyTorch tensor.
            ValueError: If the input matrix or the target vector is not the correct shape.
        """
        if not isinstance(X, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise TypeError("The input matrix and the target matrix must be a PyTorch tensor.")
        if X.ndim != 2:
            raise ValueError("The input matrix must be a 2 dimensional tensor.")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError("The target must be 1 dimensional with the same number of samples as the input data")
        if not isinstance(metrics, list | tuple):
            raise ValueError("metrics must be a list or tuple containing the shorthand names of each wanted metric.")
        
        self.n_features = X.shape[1]
        self.initial_pred = y.mean()
        pred = torch.full(y.shape, self.initial_pred)
        trees = []
        history = {metric: torch.zeros(self.n_trees) for metric in metrics}

        for i in range(self.n_trees):
            gradient = self.loss.gradient(pred, y)
            hessian = self.loss.hessian(pred, y)

            tree = _XGBoostTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split, reg_lambda=self.reg_lambda, gamma=self.gamma)
            tree.fit(X, gradient, hessian)
            prediction = tree.predict(X)

            pred += self.learning_rate * prediction
            trees.append(tree)

            values = calculate_metrics(data=(pred, y), metrics=metrics, loss=self.loss.loss)
            for metric, value in values.items():
                history[metric][i] = value

        self.trees = trees
        return history

    def predict(self, X):
        """
        Applies the fitted XGBoostingRegressor model to the input data, predicting the target values.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data to be classified.
        Returns:
            targets (torch.Tensor of shape (n_samples,)): The predicted target values corresponding to each sample.
        Raises:
            NotFittedError: If the XGBoostingRegressor model has not been fitted before predicting.
            TypeError: If the input matrix is not a PyTorch tensor.
            ValueError: If the input matrix is not the correct shape.
        """
        if not hasattr(self, "initial_pred"):
            raise NotFittedError("XGBoostingRegressor.fit() must be called before predicting.")
        if not isinstance(X, torch.Tensor):
            raise TypeError("The input matrix must be a PyTorch tensor.")
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError("The input matrix must be a 2 dimensional tensor with the same number of features as the fitted tensor.")

        pred = torch.full((X.shape[0],), self.initial_pred)
        
        for tree in self.trees:
            pred += self.learning_rate * tree.predict(X)
        return pred
