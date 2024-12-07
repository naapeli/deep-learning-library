import torch
from math import ceil
from copy import deepcopy

from . import LinearRegression
from ....Exceptions import NotFittedError


class RANSACRegression:
    """
    Implements the random sample consensus (RANSAC) regression model.

    Args:
        estimator (A regression model with fit and predict methods): A base model which is fit to random samples of the data. Defaults to LinearRegression.

    Attributes:
        best_estimator (estimator): The best model. Available after fitting.
    """
    def __init__(self, estimator=LinearRegression()):
        if not hasattr(estimator, "fit") or not hasattr(estimator, "predict"):
            raise TypeError("estimator must have fit and predict functions.")

        self.estimator = estimator

    def fit(self, X, y, min_samples=None, residual_threshold=None, max_trials=100, stop_inliers_prob=1, **kwargs):
        """
        Samples random subsamples of the datapoints and fits base estimators to the subsamples.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data, where each row is a sample and each column is a feature.
            y (torch.Tensor of shape (n_samples,)): The target values corresponding to each sample.
            min_samples (int | float | None, optional): The number of samples used to fit the base estimators. If float, ceil(n_samples * min_samples) is used and if None, n_features + 1 is used. Defaults to None
            residual_threshold (int | float | None, optional): The threshold for which larger absolute errors are considered outliers. If None, the median absolute deviation of y is used. Defaults to None.
            max_trials (int, optional): The number of tries to sample the data. Must be a positive integer. Defaults to 100.
            stop_inliers_prob (int | float, optional): If the proportion of inliers on an iteration exceeds this value, the random sampling is stopped early. Defaults to 1, i.e. the process is never stopped early as the max(n_inliers / number_of_samples_in_subsample) == 1.
            kwargs: Other parameters are passed to estimator.fit()
        """
        if not isinstance(X, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise TypeError("The input matrix and the target matrix must be a PyTorch tensor.")
        if X.ndim != 2:
            raise ValueError("The input matrix must be a 2 dimensional tensor.")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError("The targets must be 1 dimensional with the same number of samples as the input data")
        if min_samples is not None and not isinstance(min_samples, float) and not isinstance(min_samples, int):
            raise TypeError("min_samples must be one of None, float or int.")
        if isinstance(min_samples, float) and (min_samples < 0 or min_samples > 1):
            raise ValueError("If min_samples is a float, it must be in range [0, 1].")
        if residual_threshold is not None and not isinstance(residual_threshold, float) and not isinstance(residual_threshold, int):
            raise TypeError("residual_threshold must be one of None, float or int.")
        if not isinstance(max_trials, int) or max_trials < 1:
            raise ValueError("max_trials must be a positive integer.")
        if not isinstance(stop_inliers_prob, float | int) or (stop_inliers_prob < 0 or stop_inliers_prob > 1):
            raise ValueError("stop_inliers_prob must be a float in range [0, 1].")

        n_samples, self.n_features = X.shape

        if isinstance(min_samples, float):
            min_samples = ceil(n_samples * min_samples)
        min_samples = X.shape[1] + 1 if min_samples is None else min_samples

        residual_threshold = torch.median(torch.abs(y - torch.median(y))) if residual_threshold is None else residual_threshold

        max_inliers = -1

        for i in range(max_trials):
            indicies = torch.randperm(n_samples)[:min_samples]
            X_ = X[indicies]
            y_ = y[indicies]

            self.estimator.fit(X_, y_, **kwargs)
            preds = self.estimator.predict(X_)
            n_inliers = torch.sum(torch.abs(y_ - preds) < residual_threshold)
            if n_inliers > max_inliers:
                max_inliers = n_inliers
                best_model = deepcopy(self.estimator)
            
            if n_inliers / min_samples > stop_inliers_prob:
                break
        
        self.best_estimator = best_model
    
    def predict(self, X, **kwargs):
        """
        Predicts the values of the samples using the best estimator determined in the fit method.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data, where each row is a sample and each column is a feature.
            kwargs: Other parameters are passed to estimator.predict()
        """
        if not hasattr(self, "best_estimator"):
            raise NotFittedError("RANSACRegression.predict() must be called before predicting.")
        if not isinstance(X, torch.Tensor):
            raise TypeError("The input matrix must be a PyTorch tensor.")
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError("The input matrix must be a 2 dimensional tensor with the same number of features as the fitted tensor.")

        return self.best_estimator.predict(X, **kwargs)
