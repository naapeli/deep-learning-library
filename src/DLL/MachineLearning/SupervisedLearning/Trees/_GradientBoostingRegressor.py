import torch

from . import RegressionTree
from ....DeepLearning.Losses import MSE, MAE, Huber
from ....Exceptions import NotFittedError
from ....Data.Metrics import calculate_metrics


class GradientBoostingRegressor:
    """
    GradientBoostingRegressor implements a regression algorithm fitting many consecutive :class:`RegressionTrees <DLL.MachineLearning.SupervisedLearning.Trees.RegressionTree>` to residuals of the model.

    Args:
        n_trees (int, optional): The number of trees used for predicting. Defaults to 50. Must be a positive integer.
        learning_rate (float, optional): The number multiplied to each additional trees residuals. Must be a real number in range (0, 1). Defaults to 0.5.
        max_depth (int, optional): The maximum depth of the tree. Defaults to 3. Must be a positive integer.
        min_samples_split (int, optional): The minimum required samples in a leaf to make a split. Defaults to 2. Must be a positive integer.
        loss (string, optional): The loss function used in calculations of the gradients. Must be one of "squared", "absolute" or "huber". Defaults to "squared".
        huber_delta (float | int, optional): The delta parameter for the possibly used huber loss. If loss is not "huber", this parameter is ignored.
    Attributes:
        n_features (int): The number of features. Available after fitting.
    """
    def __init__(self, n_trees=50, learning_rate=0.5, max_depth=3, min_samples_split=2, loss="squared", huber_delta=1):
        if not isinstance(n_trees, int) or n_trees < 1:
            raise ValueError("n_trees must be a positive integer.")
        if not isinstance(learning_rate, float) or learning_rate <= 0 or learning_rate >= 1:
            raise ValueError("learning_rate must be a float in range (0, 1).")
        if not isinstance(max_depth, int) or max_depth < 1:
            raise ValueError("max_depth must be a positive integer.")
        if not isinstance(min_samples_split, int) or min_samples_split < 1:
            raise ValueError("min_samples_split must be a positive integer.")
        if loss not in ["squared", "absolute", "huber"]:
            raise ValueError('loss must be one in ["squared", "absolute", "huber"].')
        if not isinstance(huber_delta, int | float) or huber_delta <= 0:
            raise ValueError("huber_delta must be a positive real number.")
        
        self.n_trees = n_trees
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = None
        self.loss = self._choose_loss(loss, huber_delta)

    def _choose_loss(self, loss, huber_delta):
        if loss == "squared":
            return MSE(reduction="sum")
        elif loss == "absolute":
            return MAE(reduction="sum")
        elif loss == "huber":
            return Huber(reduction="sum", delta=huber_delta)

    def fit(self, X, y, metrics=["loss"]):
        """
        Fits the GradientBoostingRegressor model to the input data by fitting trees to the errors made by previous trees.

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
            residual = -self.loss.gradient(pred, y)

            tree = RegressionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X, residual)
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
        Applies the fitted GradientBoostingRegressor model to the input data, predicting the target values.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data to be classified.
        Returns:
            targets (torch.Tensor of shape (n_samples,)): The predicted target values corresponding to each sample.
        Raises:
            NotFittedError: If the GradientBoostingRegressor model has not been fitted before predicting.
            TypeError: If the input matrix is not a PyTorch tensor.
            ValueError: If the input matrix is not the correct shape.
        """
        if not hasattr(self, "initial_pred"):
            raise NotFittedError("GradientBoostingRegressor.fit() must be called before predicting.")
        if not isinstance(X, torch.Tensor):
            raise TypeError("The input matrix must be a PyTorch tensor.")
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError("The input matrix must be a 2 dimensional tensor with the same number of features as the fitted tensor.")

        pred = torch.full((X.shape[0],), self.initial_pred)
        
        for tree in self.trees:
            pred += self.learning_rate * tree.predict(X)
        return pred
