import torch
from .RegressionTree import RegressionTree


class GradientBoostingRegressor:
    def __init__(self, n_trees=10, learning_rate=0.1, max_depth=25, min_samples_split=2):
        self.n_trees = n_trees
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X, y):
        self.initial_pred = y.mean()
        residual = (y - self.initial_pred).squeeze()

        for _ in range(self.n_trees):
            tree = RegressionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X, residual)
            pred = tree.predict(X)

            residual -= self.learning_rate * pred
            self.trees.append(tree)

    def predict(self, X):
        assert hasattr(self, "initial_pred"), "GradientBoostingRegressor.fit(X, y) has to be called before predicting."
        pred = torch.full((X.shape[0],), self.initial_pred)
        
        for tree in self.trees:
            pred += self.learning_rate * tree.predict(X)
        return pred
