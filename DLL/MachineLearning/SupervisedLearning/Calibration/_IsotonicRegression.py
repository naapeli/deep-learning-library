import torch
from functools import partial


class IsotonicRegression:
    """
    Isotonic regression model.
    """
    def fit(self, X, y, weight=None, increasing=True):
        """
        Fits a monotonic function to the trainin data using the Pool-Adjacent-Violators (PAV) algorithm.

        Args:
            X (torch.Tensor of shape (n_samples,)): Independent variable (not necessarily sorted).
            y (torch.Tensor of shape (n_samples,)): Dependent variable.
            weight (torch.Tensor of shape (n_samples,) | None, optional): Sample weights. Must be a non-negative torch tensor. Defaults to None, which corresponds to uniform weights.
            increasing (bool, optional): Deterimines if the final predictions should be increasing or decreasing. Must be a boolean. Defaults to True.
        """
        if not isinstance(X, torch.Tensor) or not isinstance(y, torch.Tensor) or not isinstance(weight, torch.Tensor):
            raise TypeError("The inpu matrix X, the label matrix y and the weight matrix must be PyTorch tensors.")
        if X.ndim != 1:
            raise ValueError("The input matrix must be a 1 dimensional tensor.")
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("The labels must be 1 dimensional with the same number of samples as the input data.")
        if weight.ndim != 1 or len(weight) != len(X):
            raise ValueError("The weights must be 1 dimensional with the same number of samples as the input data.")
        if not isinstance(increasing, bool):
            raise TypeError("increasing must be a boolean.")

        sorted_idx = torch.argsort(X)
        X_sorted = X[sorted_idx]
        y_sorted = y[sorted_idx]
        if weight is None:
            weight = torch.ones_like(y_sorted)
        else:
            weight = weight[sorted_idx]
        if not increasing:
            y_sorted = y_sorted.flip(0)
            weight = weight.flip(0)

        solution, _ = self._pava(y_sorted, weight)
        if not increasing:
            solution = solution.flip(0)
        self._interp_func = partial(self._linear_interpolation, X_sorted, solution)
    
    def _pava(self, y, w):
        """
        Adapted from sklearn docs: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/_isotonic.pyx
        """
        n = len(y)
        y, w = y.clone(), w.clone()
        target = torch.arange(n, dtype=torch.int32)
        i = 0
        while i < n:
            k = target[i].item() + 1
            if k == n:
                break
            if y[i] < y[k]:
                i = k
                continue
            sum_wy = w[i] * y[i]
            sum_w = w[i]
            while True:
                prev_y = y[k]
                sum_wy += w[k] * y[k]
                sum_w += w[k]
                k = target[k].item() + 1
                if k == n or prev_y < y[k]:
                    y[i] = sum_wy / sum_w
                    w[i] = sum_w
                    target[i] = k - 1
                    target[k - 1] = i
                    if i > 0:
                        i = target[i - 1].item()
                    break
        i = 0
        while i < n:
            k = target[i] + 1
            y[i + 1:k] = y[i]
            i = k
        return y, w
    
    def _linear_interpolation(self, X_fit, y_fit, X_new):
        X_new = X_new.unsqueeze(0) if X_new.dim() == 0 else X_new
        y_interp = torch.zeros_like(X_new).float()

        for i, x in enumerate(X_new):
            if x <= X_fit[0]:
                y_interp[i] = y_fit[0]
            elif x >= X_fit[-1]:
                y_interp[i] = y_fit[-1]
            else:
                idx = torch.searchsorted(X_fit, x, right=True) - 1
                x0, x1 = X_fit[idx], X_fit[min(idx + 1, len(X_fit) - 1)]
                y0, y1 = y_fit[idx], y_fit[min(idx + 1, len(X_fit) - 1)]
                y_interp[i] = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        return y_interp

    def predict(self, X):
        """
        Predicts values using the fitted isotonic regression model with linear interpolation.

        Parameters:
            X (torch.Tensor): New independent variable values.

        Returns:
            torch.Tensor: Predicted values.
        """
        if not isinstance(X, torch.Tensor) or X.ndim != 1:
            raise ValueError("The input matrix must be a torch.Tensor of shape (n_samples,).")

        return self._interp_func(X)
