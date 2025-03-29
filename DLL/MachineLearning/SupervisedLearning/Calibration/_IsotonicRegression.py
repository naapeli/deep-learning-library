import torch
from functools import partial


class IsotonicRegression:
    """
    Isotonic regression model.
    """
    def fit(self, X, y, weight=None):
        """
        Fits amonotonic function to the trainin data using the Pool-Adjacent-Violators (PAV) algorithm.

        Parameters:
            X (torch.Tensor): Independent variable (not necessarily sorted).
            y (torch.Tensor): Dependent variable (to be monotonized).
            weight (torch.Tensor, optional): Sample weights (default is uniform weights).

        Returns:
            self (IsotonicRegression): The fitted model.
        """
        sorted_idx = torch.argsort(X)
        X_sorted = X[sorted_idx]
        y_sorted = y[sorted_idx]
        n = len(y_sorted)
        if weight is None:
            weight = torch.ones_like(y_sorted)
        else:
            weight = weight[sorted_idx]

        solution = self._pava(y_sorted, weight, torch.full(size=(n + 1,), fill_value=-1))[0]  # returns slightly different values than below, which doesn't make sense since implementation should just be a port of c++ from scipy source code to here!!!
        # print(solution)
        # from scipy.optimize import isotonic_regression
        # solution = torch.from_numpy(isotonic_regression(y_sorted.numpy(), weights=weight.numpy()).x)
        # print(solution)

        self._interp_func = partial(self._linear_interpolation, X_sorted, solution)

    def _pava(self, x, w, r):
        n = len(x)
        r[0], r[1] = 0, 1
        b = 0
        xb_prev, wb_prev = x[b], w[b]
        for i in range(1, n):
            b += 1
            xb, wb = x[i], w[i]
            sb = 0
            if xb_prev >= xb:
                b -= 1
                sb = wb_prev * xb_prev + wb * xb
                wb += wb_prev
                xb = sb / wb
                while i < n - 1 and xb >= x[i + 1]:
                    i += 1
                    sb += w[i] * x[i]
                    wb += w[i]
                    xb = sb / wb
                while b > 0 and x[b - 1] >= xb:
                    b -= 1
                    sb += w[b] * x[b]
                    wb += w[b]
                    xb = sb / wb
            x[b] = xb_prev = xb
            w[b] = wb_prev = wb
            r[b + 1] = i + 1
        f = n - 1
        for k in range(b, -1, -1):
            t, xk = r[k], x[k]
            for i in range(f, t - 1, -1):
                x[i] = xk
            f = t - 1
        return x, w, r, b + 1
    
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
        result = self._interp_func(X)
        return result
