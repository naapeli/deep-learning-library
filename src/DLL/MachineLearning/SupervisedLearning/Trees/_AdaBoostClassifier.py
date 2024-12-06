import torch
from math import log
from warnings import warn

from ._DecisionTree import DecisionTree
from ....Exceptions import NotFittedError


class AdaBoostClassifier:
    """
    AdaBoostClassifier implements a classification algorithm fitting many consecutive :class:`DecisionTrees <DLL.MachineLearning.SupervisedLearning.Trees.DecisionTree>` to previously missclassified samples.

    Args:
        n_trees (int, optional): The number of trees used for predicting. Defaults to 10. Must be a positive integer.
        max_depth (int, optional): The maximum depth of the tree. Defaults to 25. Must be a positive integer.
        min_samples_split (int, optional): The minimum required samples in a leaf to make a split. Defaults to 2. Must be a positive integer.
        criterion (str, optional): The information criterion used to select optimal splits. Must be one of "entropy" or "gini". Defaults to "gini".
    Attributes:
        n_features (int): The number of features. Available after fitting.
        n_classes (int): The number of classes. 2 for binary classification. Available after fitting.
        confidences (torch.tensor of shape (n_trees,)): The confidence on each tree.
    """
    def __init__(self, n_trees=10, max_depth=25, min_samples_split=2, criterion="gini"):
        if not isinstance(n_trees, int) or n_trees < 1:
            raise ValueError("n_trees must be a positive integer.")
        if not isinstance(max_depth, int) or max_depth < 1:
            raise ValueError("max_depth must be a positive integer.")
        if not isinstance(min_samples_split, int) or min_samples_split < 1:
            raise ValueError("min_samples_split must be a positive integer.")
        if criterion not in ["entropy", "gini"]:
            raise ValueError('The chosen criterion must be one of "entropy" or "gini".')
        
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.trees = None

    def fit(self, X, y):
        """
        Fits the AdaBoostClassifier model to the input data by fitting trees to the errors made by previous trees.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data, where each row is a sample and each column is a feature.
            y (torch.Tensor of shape (n_samples,)): The labels corresponding to each sample. Every element must be in [0, ..., n_classes - 1].
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
        vals = torch.unique(y).numpy()
        if set(vals) != {*range(len(vals))}:
            raise ValueError("y must only contain the values in [0, ..., n_classes - 1].")
        
        self.n_classes = len(vals)
        self.classes = vals
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

            #  Make sure atleast 1 datapoint is present from each class
            sample_classes = torch.unique(y_sample)
            if len(sample_classes) < self.n_classes:
                unseen_classes = list(set(self.classes) - set(sample_classes))
                X_new = []
                y_new = []
                for class_ in unseen_classes:
                    index = torch.multinomial((y == class_).float(), 1)
                    X_new.append(X[index].squeeze(0))
                    y_new.append(y[index].squeeze(0))
                X_sample = torch.cat([X_sample, torch.stack(X_new)])
                y_sample = torch.cat([y_sample, torch.stack(y_new)])
            
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split, criterion=self.criterion)
            tree.fit(X_sample, y_sample)
            prediction = tree.predict(X)
            incorrect = prediction != y
            eps = torch.sum(incorrect * weights)
            errors.append(eps)

            #  If better than random quessing, continue
            if eps < 1 - 1 / self.n_classes:
                alpha = 0.5 * (torch.log((1 - eps) / (eps + 1e-8)) + log(self.n_classes - 1))
                weights = weights * torch.exp(alpha * incorrect)
                weights /= weights.sum()  # keep weights as a probability distribution
                self.confidences.append(alpha)
            else:
                warn(f"The average error exceeds 0.5. The training is stopped to reduce over fitting. Only {i} trees are used.")
                break

            trees.append(tree)
        
        self.trees = trees
        return errors

    def predict(self, X):
        """
        Applies the fitted AdaBoostClassifier model to the input data, predicting the correct classes.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data to be classified.

        Returns:
            labels (torch.Tensor of shape (n_samples,)): The predicted labels corresponding to each sample.

        Raises:
            NotFittedError: If the AdaBoostClassifier model has not been fitted before predicting.
            TypeError: If the input matrix is not a PyTorch tensor.
            ValueError: If the input matrix is not the correct shape.
        """
        if self.trees is None:
            raise NotFittedError("AdaBoostClassifier.fit() must be called before predicting.")
        if not isinstance(X, torch.Tensor):
            raise TypeError("The input matrix must be a PyTorch tensor.")
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError("The input matrix must be a 2 dimensional tensor with the same number of features as the fitted tensor.")

        preds = torch.zeros((len(X), self.n_classes))
        for alpha, tree in zip(self.confidences, self.trees):
            pred = tree.predict(X).int()
            one_hot_pred = torch.eye(self.n_classes)[pred]
            preds += alpha * one_hot_pred
        
        return preds.argmax(dim=1)
    
    def predict_proba(self, X):
        """
        Applies the fitted AdaBoostClassifier model to the input data, predicting the probabilities of each class.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data for which to predict probabilities.

        Returns:
            torch.Tensor of shape (n_samples, n_classes): The predicted probabilities for each class.
        
        Raises:
            NotFittedError: If the AdaBoostClassifier model has not been fitted before predicting.
            TypeError: If the input matrix is not a PyTorch tensor.
            ValueError: If the input matrix is not the correct shape.
        """
        if self.trees is None:
            raise NotFittedError("AdaBoostClassifier.fit() must be called before predicting.")
        if not isinstance(X, torch.Tensor):
            raise TypeError("The input matrix must be a PyTorch tensor.")
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError("The input matrix must be a 2 dimensional tensor with the same number of features as the fitted tensor.")

        preds = torch.zeros((len(X), self.n_classes))
        for alpha, tree in zip(self.confidences, self.trees):
            pred = tree.predict(X).int()
            one_hot_pred = torch.eye(self.n_classes)[pred]
            preds += alpha * one_hot_pred

        probs = torch.exp(preds - torch.max(preds, dim=1, keepdim=True).values)
        probs = probs / torch.sum(probs, dim=1, keepdim=True)
        if self.n_classes == 2:
            return probs[:, 1]
        return probs
