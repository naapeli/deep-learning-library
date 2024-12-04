import torch

from . import LinearRegression


class SARIMA:
    """
    The Seasonal auto regressive moving average model for time series analysis.

    Args:
        series (torch.Tensor of shape (n_samples,)): The time series for fitting. Must be one dimensional.
        order (tuple of ints): The orders of the non-seasonal parts. Follows the format (p, d, q).
        seasonal_order (tuple of ints): The orders of the seasonal parts. Follows the format (P, D, Q, S). If a seasonal component is not needed, the seasonal order should be put as (0, 0, 0, 1).
    """
    def __init__(self, series, order, seasonal_order):
        if not isinstance(series, torch.Tensor) or series.ndim != 1:
            raise TypeError("series must be a one-dimensional torch tensor.")
        if not isinstance(order, tuple | list) or len(order) != 3:
            raise TypeError("order must be a tuple of length 3.")
        if any([not isinstance(val, int) or val < 0 for val in order]):
            raise ValueError("order must only contain non-negative integers.")
        if not isinstance(seasonal_order, tuple | list) or len(seasonal_order) != 4:
            raise TypeError("seasonal_order must be a tuple of length 4.")
        if any([not isinstance(val, int) or val < 0 for val in seasonal_order]):
            raise ValueError("seasonal_order must only contain non-negative integers.")

        self.p, self.d, self.q = order
        self.P, self.D, self.Q, self.S = seasonal_order

        self.series = series
        self._discarded = {}
        if self.d > 0: series = self._differentiate(series, order=self.d)
        if self.D > 0: series = self._differentiate(series, lag=self.S, order=self.D)
        self.diff_series = series

        min_length = max(self.p, self.q, self.P * self.S, self.Q * self.S)
        if len(self.diff_series) <= min_length:
            raise ValueError(f"Differentiated series' length {len(self.diff_series)} is less than or equal minimum required length {min_length} for the given orders.")

    def fit(self):
        """
        Fits the ARMA model to the given time series. Currently, the function fits two linear regression models separately for the AR and MA components.

        Note:
            This approach is suboptimal for the MA component, as it should be fitted using Kalman filters for correctness.
        """
        residuals = self._fit_ar()
        self._fit_ma(residuals)
    
    def _fit_ar(self):
        X, y = self._train_data(part="ar")

        self.ar_model = LinearRegression()
        self.ar_model.fit(X, y)
        return y - self.ar_model.predict(X)

    def _fit_ma(self, residuals):
        X, _ = self._train_data(part="ma")
        length = min(len(X), len(residuals))
        X, residuals = X[-length:], residuals[-length:]

        self.ma_model = LinearRegression()
        self.ma_model.fit(X, residuals)
    
    def _train_data(self, part):
        if part == "ar":
            X_series, X_targets = self._lagged_terms(order=self.p)
            seasonal_X_series, seasonal_X_targets = self._lagged_terms(order=self.P, lag=self.S)
        elif part == "ma":
            X_series, X_targets = self._lagged_terms(order=self.q)
            seasonal_X_series, seasonal_X_targets = self._lagged_terms(order=self.Q, lag=self.S)
        length = min(len(X_series), len(seasonal_X_series))
        X = torch.cat((X_series[-length:], seasonal_X_series[-length:]), dim=1)
        y = X_targets if len(X_targets) < len(seasonal_X_targets) else seasonal_X_targets
        return X, y

    def _lagged_terms(self, order, lag=1):
        points = []
        for i in range(lag * order, len(self.diff_series)):
            indicies = i - lag * torch.arange(1, order + 1)
            features = self.diff_series[indicies]
            points.append(features)
        targets = self.diff_series[lag * order:]
        return torch.stack(points, dim=0), targets

    def _differentiate(self, series, lag=1, order=1):
        discarded = []
        for _ in range(order):
            discarded.append(series[:lag])
            series = series[lag:] - series[:-lag]
        self._discarded[lag] = torch.stack(discarded, dim=0)
        return series

    def _integrate(self, differenced, lag=1, order=1):
        for j in range(order):
            restored = torch.zeros(len(differenced) + lag)
            restored[:lag] = self._discarded[lag][-j - 1]
            for i in range(lag, len(restored)):
                restored[i] = differenced[i - lag] + restored[i - lag]
            differenced = restored
        return differenced

    def predict(self, steps=1, fit_between_steps=False):
        """
        Predicts the next values of the given time series.

        Args:
            steps (int, optional): The number of next values to predict. Must be a positive integer. Defaults to 1.
            fit_between_steps (bool, optional): Determines if the model should be refitted between each prediction. Defaults to False.

        Returns:
            torch.Tensor: The predicted values as a one-dimensional torch Tensor.
        """
        if not isinstance(steps, int) or steps <= 0:
            raise ValueError("steps must be a positive integer.")
        if not isinstance(fit_between_steps, bool):
            raise TypeError("fit_between_steps must be a boolean.")

        diff_series = self.diff_series.clone()
        original_diff_series = self.diff_series.clone()
        for _ in range(steps):
            pred = self._predict_next(diff_series)
            diff_series = torch.cat((diff_series, pred), dim=0)

            if fit_between_steps:
                self.diff_series = diff_series
                residuals = self._fit_ar()
                self._fit_ma(residuals)
        
        if self.D > 0: diff_series = self._integrate(diff_series, lag=self.S, order=self.D)
        if self.d > 0: diff_series = self._integrate(diff_series, order=self.d)
        self.diff_series = original_diff_series
        result = diff_series
        return result[-steps:]

    def _predict_next(self, diff_series):
        indicies = -torch.arange(1, self.p + 1)
        indicies = torch.cat((-self.S * torch.arange(1, self.P + 1), indicies), dim=0)
        X_ar = diff_series[indicies].unsqueeze(0)

        indicies = -torch.arange(1, self.q + 1)
        indicies = torch.cat((-self.S * torch.arange(1, self.Q + 1), indicies), dim=0)
        X_ma = diff_series[indicies].unsqueeze(0)

        ar_pred = self.ar_model.predict(X_ar)
        ma_correction = self.ma_model.predict(X_ma)
        return ar_pred + ma_correction
