import torch

from ._XGBoostTree import _XGBoostTree
from ....DeepLearning.Layers.Activations import Sigmoid, SoftMax
from ....DeepLearning.Losses import BCE, CCE, Exponential
from ....Data.Preprocessing import OneHotEncoder
from ....Exceptions import NotFittedError
from ....Data.Metrics import calculate_metrics, prob_to_pred


class XGBoostingClassifier:
    """
    XGBoostingClassifier implements a classification algorithm fitting many consecutive trees to gradients and hessians of the predictions.

    Args:
        n_trees (int, optional): The number of trees used for predicting. Defaults to 10. Must be a positive integer.
        learning_rate (float, optional): The number multiplied to each additional trees residuals. Must be a real number in range (0, 1). Defaults to 0.5.
        max_depth (int, optional): The maximum depth of the tree. Defaults to 25. Must be a positive integer.
        min_samples_split (int, optional): The minimum required samples in a leaf to make a split. Defaults to 2. Must be a positive integer.
        reg_lambda (float | int, optional): The regularisation parameter used in fitting the trees. The larger the parameter, the smaller the trees. Must be a positive real number. Defaults to 1.
        gamma (float | int, optional): The minimum gain to make a split. Must be a non-negative real number. Defaults to 0.
        loss (string, optional): The loss function used in calculations of the residuals. Must be one of "log_loss" or "exponential". Defaults to "log_loss". "exponential" can only be used for binary classification.
    Attributes:
        n_features (int): The number of features. Available after fitting.
        n_classes (int): The number of classes. 2 for binary classification. Available after fitting.
    """
    def __init__(self, n_trees=10, learning_rate=0.5, max_depth=25, min_samples_split=2, reg_lambda=1, gamma=0, loss="log_loss"):
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
        if loss not in ["log_loss", "exponential"]:
            raise ValueError('loss must be one of ["log_loss", "exponential"]')

        self.n_trees = n_trees
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.trees = None
        self.loss_ = loss
    
    def _get_activation_and_loss(self, classes):
        self.n_classes = len(classes)
        if self.loss_ == "log_loss":
            if self.n_classes == 2:
                self.loss = BCE(reduction="sum")
                self.activation = Sigmoid()
            else:
                self.loss = CCE(reduction="sum")
                self.activation = SoftMax()
        elif self.loss_ == "exponential":
            if self.n_classes != 2:
                raise ValueError("The exponential loss is only applicable in binary classification. Use log_loss for multiclass classification instead.")
            self.loss = Exponential(reduction="sum")
            self.activation = Sigmoid()

    def fit(self, X, y, metrics=["loss"]):
        """
        Fits the XGBoostingClassifier model to the input data by fitting trees to the errors made by previous trees.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data, where each row is a sample and each column is a feature.
            y (torch.Tensor of shape (n_samples,)): The labels corresponding to each sample. Every element must be in [0, ..., n_classes - 1].
            metrics (dict[str, torch.Tensor]): Contains the metrics that will be calculated between fitting each tree and returned. Only available for binary classification.
        Returns:
            metrics if binary classification else None
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
        vals = torch.unique(y).numpy()
        if set(vals) != {*range(len(vals))}:
            raise ValueError("y must only contain the values in [0, ..., n_classes - 1].")
        
        self._get_activation_and_loss(torch.unique(y))
        y = y.to(X.dtype)
        self.n_features = X.shape[1]

        if self.n_classes == 2:
            if not isinstance(metrics, list | tuple):
                raise ValueError("metrics must be a list or tuple containing the shorthand names of each wanted metric.")

            return self._binary_fit(X, y, metrics=metrics)
        else:
            self._multi_fit(X, y)
    
    def _binary_hessian_diag(self, prob, true_output):
        loss_gradient = self.loss.gradient(prob, true_output)
        loss_hessian = self.loss.hessian(prob, true_output)
        sigmoid_value = prob
        sigmoid_gradient = self.activation.backward(loss_gradient)
        
        first_term = sigmoid_gradient * (1 - sigmoid_value) * loss_gradient
        second_term = -sigmoid_value * sigmoid_gradient * loss_gradient
        third_term = sigmoid_value * (1 - sigmoid_value) * loss_hessian  # is the same as self.activation.backward(loss_hessian)
        return first_term + second_term + third_term
    
    def _multi_hessian_diag(self, prob, true_output):
        loss_gradient = self.loss.gradient(prob, true_output)
        loss_hessian = self.loss.hessian(prob, true_output)
        softmax_value = prob
        softmax_gradient = self.activation.backward(loss_gradient)

        softmax_hessian = softmax_value * (1 - softmax_value) * (1 - 2 * softmax_value)
        
        # From Mathematica: D[f[g[x]], {x, 2}] = Derivative[1][g][x]^2 (f^\[Prime]\[Prime])[g[x]] + Derivative[1][f][g[x]] (g^\[Prime]\[Prime])[x]
        return softmax_gradient ** 2 * loss_hessian + loss_gradient * softmax_hessian

    def _binary_fit(self, X, y, metrics=["loss"]):
        positive_ratio = y.mean()
        self.initial_log_odds = torch.log(positive_ratio / (1 - positive_ratio))
        pred = torch.full(y.shape, self.initial_log_odds)
        trees = []
        history = {metric: torch.zeros(self.n_trees) for metric in metrics}

        for i in range(self.n_trees):
            prob = self.activation.forward(pred)
            gradient = self.activation.backward(self.loss.gradient(prob, y))
            hessian = self._binary_hessian_diag(prob, y)

            tree = _XGBoostTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split, reg_lambda=self.reg_lambda, gamma=self.gamma)
            tree.fit(X, gradient, hessian)
            prediction = tree.predict(X)
            pred += self.learning_rate * prediction

            trees.append(tree)

            values = calculate_metrics(data=(self.activation.forward(pred), y), metrics=metrics, loss=self.loss.loss)
            for metric, value in values.items():
                history[metric][i] = value
        
        self.trees = trees
        return history
    
    def _multi_fit(self, X, y):
        encoder = OneHotEncoder()
        y = encoder.fit_encode(y)
        
        self.initial_log_odds = 0.0
        pred = torch.full(y.shape, self.initial_log_odds)
        trees = []

        for class_index in range(self.n_classes):
            class_trees = []
            for _ in range(self.n_trees):
                prob = self.activation.forward(pred)
                gradient = self.activation.backward(self.loss.gradient(prob, y))[:, class_index]
                hessian = self._multi_hessian_diag(prob, y)[:, class_index]

                tree = _XGBoostTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split, reg_lambda=self.reg_lambda, gamma=self.gamma)
                tree.fit(X, gradient, hessian)
                prediction = tree.predict(X)
                pred[:, class_index] += self.learning_rate * prediction

                class_trees.append(tree)
            trees.append(class_trees)
        
        self.trees = trees
    
    def predict_proba(self, X):
        """
        Applies the fitted XGBoostingClassifier model to the input data, predicting the probabilities of each class.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data to be classified.
        Returns:
            probabilities (torch.Tensor of shape (n_samples, n_classes) or for binary classification (n_samples,)): The predicted probabilities corresponding to each sample.
        Raises:
            NotFittedError: If the XGBoostingClassifier model has not been fitted before predicting.
            TypeError: If the input matrix is not a PyTorch tensor.
            ValueError: If the input matrix is not the correct shape.
        """
        if not hasattr(self, "initial_log_odds"):
            raise NotFittedError("XGBoostingClassifier.fit() must be called before predicting.")
        if not isinstance(X, torch.Tensor):
            raise TypeError("The input matrix must be a PyTorch tensor.")
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError("The input matrix must be a 2 dimensional tensor with the same number of features as the fitted tensor.")

        if self.n_classes > 2:
            return self._multi_predict_proba(X)
        
        pred = torch.full((X.shape[0],), self.initial_log_odds)
        
        for tree in self.trees:
            prediction = tree.predict(X)
            pred += self.learning_rate * prediction

        return self.activation.forward(pred)

    def predict(self, X):
        """
        Applies the fitted XGBoostingClassifier model to the input data, predicting the correct classes.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data to be classified.
        Returns:
            labels (torch.Tensor of shape (n_samples,)): The predicted labels corresponding to each sample.
        Raises:
            NotFittedError: If the XGBoostingClassifier model has not been fitted before predicting.
            TypeError: If the input matrix is not a PyTorch tensor.
            ValueError: If the input matrix is not the correct shape.
        """
        if not hasattr(self, "initial_log_odds"):
            raise NotFittedError("XGBoostingClassifier.fit() must be called before predicting.")
        if not isinstance(X, torch.Tensor):
            raise TypeError("The input matrix must be a PyTorch tensor.")
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError("The input matrix must be a 2 dimensional tensor with the same number of features as the fitted tensor.")
        
        prob = self.predict_proba(X)
        return prob_to_pred(prob)
    
    def _multi_predict_proba(self, X):
        pred = torch.full((X.shape[0], self.n_classes), self.initial_log_odds)

        for i in range(self.n_classes):
            class_trees = self.trees[i]
            for tree in class_trees:
                pred[:, i] += self.learning_rate * tree.predict(X)

        return self.activation.forward(pred)

