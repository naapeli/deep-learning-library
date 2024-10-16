import torch

from .RegressionTree import RegressionTree
from ....DeepLearning.Layers.Activations import Sigmoid


class GradientBoostingClassifier:
    def __init__(self, n_trees=10, learning_rate=0.1, max_depth=25, min_samples_split=2):
        self.n_trees = n_trees
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.sigmoid = Sigmoid(output_shape=None)

    def fit(self, X, y):
        assert y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1), "y has to be a 1 dimensional vector."
        assert set(torch.unique(y).numpy()) == {0, 1}, "The classes must be labelled 0 and 1."
        y = y.squeeze().to(X.dtype)
        positive_ratio = y.mean()
        self.initial_log_odds = torch.log(positive_ratio / (1 - positive_ratio))
        pred = torch.full(y.shape, self.initial_log_odds)

        for _ in range(self.n_trees):
            prob = self.sigmoid.forward(pred)
            residual = y - prob

            tree = RegressionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X, residual)
            pred += self.learning_rate * tree.predict(X)

            self.trees.append(tree)
    
    def predict_proba(self, X):
        assert hasattr(self, "initial_log_odds"), "GradientBoostingClassifier.fit(X, y) has to be called before predicting."
        pred = torch.full((X.shape[0],), self.initial_log_odds)
        
        for tree in self.trees:
            pred += self.learning_rate * tree.predict(X)
        
        return self.sigmoid.forward(pred)

    def predict(self, X):
        assert hasattr(self, "initial_log_odds"), "GradientBoostingClassifier.fit(X, y) has to be called before predicting."
        prob = self.predict_proba(X)
        return (prob >= 0.5).to(torch.int32)
