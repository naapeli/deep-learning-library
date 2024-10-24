import torch

from .RegressionTree import RegressionTree
from ....DeepLearning.Layers.Activations import Sigmoid, SoftMax
from ....DeepLearning.Losses import bce, cce, exponential
from ....Data.Preprocessing import OneHotEncoder



class GradientBoostingClassifier:
    """
    GradientBoostingClassifier implements a classification algorithm fitting many consecutive :class:`DecisionTrees <DLL.MachineLearning.SupervisedLearning.Trees.DecisionTree>` to residuals of the model.

    Args:
        n_trees (int, optional): The number of trees used for predicting. Defaults to 10. Must be a positive integer.
        learning_rate (float, optional): The number multiplied to each additional trees residuals.
        max_depth (int, optional): The maximum depth of the tree. Defaults to 25. Must be a positive integer.
        min_samples_split (int, optional): The minimum required samples in a leaf to make a split. Defaults to 2. Must be a positive integer.
    """
    def __init__(self, n_trees=10, learning_rate=0.1, max_depth=25, min_samples_split=2, loss="log_loss"):
        # raise valueError if loss wrong
        self.n_trees = n_trees
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.loss_ = loss
    
    def _get_activation_and_loss(self, n_classes):
        self.n_classes = n_classes
        if self.loss_ == "log_loss":
            if n_classes == 2:
                self.loss = bce()
                self.activation = Sigmoid()
            else:
                self.loss = cce()
                self.activation = SoftMax()
        elif self.loss_ == "exponential":
            if n_classes != 2:
                raise ValueError("The exponential loss is only applicable in binary classification. Use log_loss for multiclass classification instead.")
            self.loss = exponential()
            self.activation = Sigmoid()

    def fit(self, X, y):
        assert y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1), "y has to be a 1 dimensional vector."
        assert set(torch.unique(y).numpy()) == {0, 1}, "The classes must be labelled 0 and 1."
        self._get_activation_and_loss(len(torch.unique(y)), y)

        y = y.squeeze().to(X.dtype)
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
        y = encoder.one_hot_encode(y)
        
        self.initial_log_odds = 0.0
        pred = torch.full(y.shape, self.initial_log_odds)

        for class_index in self.n_classes:
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
        assert hasattr(self, "initial_log_odds"), "GradientBoostingClassifier.fit(X, y) has to be called before predicting."
        pred = torch.full((X.shape[0],), self.initial_log_odds)
        
        for tree in self.trees:
            prediction = tree.predict(X)
            pred += self.learning_rate * prediction
        
        return self.activation.forward(pred)

    def predict(self, X):
        assert hasattr(self, "initial_log_odds"), "GradientBoostingClassifier.fit(X, y) has to be called before predicting."
        prob = self.predict_proba(X)
        return (prob >= 0.5).to(torch.int32)
