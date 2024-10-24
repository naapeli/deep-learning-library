import torch

from .RegressionTree import RegressionTree
from ....DeepLearning.Layers.Activations import Sigmoid, SoftMax
from ....DeepLearning.Losses import bce, cce, exponential
from ....Data.Preprocessing import OneHotEncoder
from ....Exceptions import NotFittedError



class GradientBoostingClassifier:
    """
    GradientBoostingClassifier implements a classification algorithm fitting many consecutive :class:`RegressionTrees <DLL.MachineLearning.SupervisedLearning.Trees.RegressionTree>` to residuals of the model.

    Args:
        n_trees (int, optional): The number of trees used for predicting. Defaults to 10. Must be a positive integer.
        learning_rate (float, optional): The number multiplied to each additional trees residuals. Must be a real number in range (0, 1). Defaults to 0.5.
        max_depth (int, optional): The maximum depth of the tree. Defaults to 25. Must be a positive integer.
        min_samples_split (int, optional): The minimum required samples in a leaf to make a split. Defaults to 2. Must be a positive integer.
        loss (string, optional): The loss function used in calculations of the residuals. Must be one of "log_loss" or "exponential". Defaults to "log_loss". "exponential" can only be used for binary classification.
    Attributes:
        n_features (int): The number of features. Available after fitting.
        n_classes (int): The number of classes. 2 for binary classification. Available after fitting.
        classes (torch.Tensor of shape (n_classes,)): The classes used as labels. Available after fitting. Contains integers [0, ..., n_classes - 1].
    """
    def __init__(self, n_trees=10, learning_rate=0.5, max_depth=25, min_samples_split=2, loss="log_loss"):
        if not isinstance(n_trees, int) or n_trees < 1:
            raise ValueError("n_trees must be a positive integer.")
        if not isinstance(learning_rate, float) or learning_rate <= 0 or learning_rate >= 1:
            raise ValueError("learning_rate must be a float in range (0, 1).")
        if not isinstance(max_depth, int) or max_depth < 1:
            raise ValueError("max_depth must be a positive integer.")
        if not isinstance(min_samples_split, int) or min_samples_split < 1:
            raise ValueError("min_samples_split must be a positive integer.")
        if loss not in ["log_loss", "exponential"]:
            raise ValueError('loss must be one of ["log_loss", "exponential"]')
        self.n_trees = n_trees
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.loss_ = loss
    
    def _get_activation_and_loss(self, classes):
        self.classes = classes
        self.n_classes = len(classes)
        if self.loss_ == "log_loss":
            if self.n_classes == 2:
                self.loss = bce()
                self.activation = Sigmoid()
            else:
                self.loss = cce()
                self.activation = SoftMax()
        elif self.loss_ == "exponential":
            if self.n_classes != 2:
                raise ValueError("The exponential loss is only applicable in binary classification. Use log_loss for multiclass classification instead.")
            self.loss = exponential()
            self.activation = Sigmoid()

    def fit(self, X, y):
        """
        Fits the GradientBoostingClassifier model to the input data by fitting trees too the errors made by previous trees.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data, where each row is a sample and each column is a feature.
            y (torch.Tensor of shape (n_samples,)): The labels corresponding to each sample. Every element must be in [0, ..., n_classes - 1].
        Returns:
            None
        Raises:
            TypeError: If the input matrix or the label vector is not a PyTorch tensor.
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
            self._binary_fit(X, y)
        else:
            self._multi_fit(X, y)

    def _binary_fit(self, X, y):
        positive_ratio = y.mean()
        self.initial_log_odds = torch.log(positive_ratio / (1 - positive_ratio))
        pred = torch.full(y.shape, self.initial_log_odds)

        for _ in range(self.n_trees):
            prob = self.activation.forward(pred)
            residual = -self.activation.backward(self.loss.gradient(prob, y))

            tree = RegressionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X, residual)
            prediction = tree.predict(X)
            pred += self.learning_rate * prediction

            self.trees.append(tree)
    
    def _multi_fit(self, X, y):
        encoder = OneHotEncoder()
        encoder.fit(y)
        y = encoder.one_hot_encode(y)
        
        self.initial_log_odds = 0.0
        pred = torch.full(y.shape, self.initial_log_odds)

        for class_index in range(self.n_classes):
            class_trees = []
            for _ in range(self.n_trees):
                prob = self.activation.forward(pred)
                residual = -self.activation.backward(self.loss.gradient(prob, y))[:, class_index]

                tree = RegressionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
                tree.fit(X, residual)
                prediction = tree.predict(X)
                pred[:, class_index] += self.learning_rate * prediction

                class_trees.append(tree)
            self.trees.append(class_trees)
    
    def predict_proba(self, X):
        """
        Applies the fitted GradientBoostingClassifier model to the input data, predicting the probabilities of each class.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data to be classified.
        Returns:
            probabilities (torch.Tensor of shape (n_samples, n_classes) or for binary classification (n_samples,)): The predicted probabilities corresponding to each sample.
        Raises:
            NotFittedError: If the GradientBoostingClassifier model has not been fitted before predicting.
            TypeError: If the input matrix is not a PyTorch tensor.
            ValueError: If the input matrix is not the correct shape.
        """
        if not hasattr(self, "initial_log_odds"):
            raise NotFittedError("GradientBoostingClassifier.fit() must be called before predicting.")
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
        Applies the fitted GradientBoostingClassifier model to the input data, predicting the correct classes.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data to be classified.
        Returns:
            labels (torch.Tensor of shape (n_samples,)): The predicted labels corresponding to each sample.
        Raises:
            NotFittedError: If the GradientBoostingClassifier model has not been fitted before predicting.
            TypeError: If the input matrix is not a PyTorch tensor.
            ValueError: If the input matrix is not the correct shape.
        """
        if not hasattr(self, "initial_log_odds"):
            raise NotFittedError("GradientBoostingClassifier.fit() must be called before predicting.")
        if not isinstance(X, torch.Tensor):
            raise TypeError("The input matrix must be a PyTorch tensor.")
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError("The input matrix must be a 2 dimensional tensor with the same number of features as the fitted tensor.")
        
        prob = self.predict_proba(X)
        if self.n_classes == 2:
            return (prob >= 0.5).to(torch.int32)
        return self.classes[prob.argmax(dim=1)]
    
    def _multi_predict_proba(self, X):
        pred = torch.full((X.shape[0], self.n_classes), self.initial_log_odds)

        for i in range(self.n_classes):
            class_trees = self.trees[i]
            for tree in class_trees:
                pred[:, i] += self.learning_rate * tree.predict(X)

        return self.activation.forward(pred)

