import torch
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from ....Exceptions import NotFittedError


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
        self.diff_series = series.numpy()

        min_length = max(self.p, self.q, self.P * self.S, self.Q * self.S)
        if len(self.diff_series) <= min_length:
            raise ValueError(f"Differentiated series' length {len(self.diff_series)} is less than or equal minimum required length {min_length} for the given orders.")
    
    def _expand_seasonal_poly(self, coefs):
        poly = np.array([1.0])
        for i, c in enumerate(coefs, 1):
            p = np.zeros(i * self.S + 1)
            p[0] = 1.0
            p[i * self.S] = c
            poly = np.convolve(poly, p)
        return poly
    
    def _difference_series(self, y):
        z = np.array(y, dtype=float)
        for _ in range(self.d):
            z = np.diff(z, n=1)
        for _ in range(self.D):
            z = z[self.S:] - z[:-self.S]
        return z

    def _differentiate(self, series, lag=1, order=1):
        discarded = []
        for _ in range(order):
            discarded.append(series[:lag])
            series = series[lag:] - series[:-lag]
        self._discarded[lag] = torch.stack(discarded, dim=0)
        return series
    
    def _arma_state_space(self, phi, theta):
        p = len(phi)
        q = len(theta)
        r = max(p, q + 1)
        T = np.zeros((r, r))
        if p > 0:
            T[0, :p] = phi
        if r > 1:
            T[1:, :-1] = np.eye(r - 1)
        Z = np.zeros((1, r)); Z[0, 0] = 1.0
        R = np.zeros((r, 1)); R[0, 0] = 1.0
        for i in range(1, min(q + 1, r)):
            R[i, 0] = theta[i - 1]
        return T, Z, R, r

    def _kalman_loglik(self, phi, theta, sigma2):
        T, Z, R, r = self._arma_state_space(phi, theta)
        n = len(self.diff_series)
        a = np.zeros((r, 1))
        P = np.eye(r) * 1e6
        loglik = 0.0
        fitted = np.zeros(n)
        for t in range(n):
            a_pred = T @ a
            P_pred = T @ P @ T.T + (R @ R.T) * sigma2
            v = self.diff_series[t] - (Z @ a_pred).item()
            f = (Z @ P_pred @ Z.T).item()
            f = max(f, 1e-9)
            K = (P_pred @ Z.T) / f
            a = a_pred + K * v
            P = P_pred - K @ (Z @ P_pred)
            loglik += -0.5 * (np.log(2 * np.pi * f) + v ** 2 / f)
            fitted[t] = (Z @ a_pred).item()
        return loglik, fitted
    
    def _sarima_loglik(self, phi, theta, Phi, Theta, sigma2):
        phi_ns = np.r_[1, -np.array(phi)] if len(phi) > 0 else np.array([1.0])
        theta_ns = np.r_[1, np.array(theta)] if len(theta) > 0 else np.array([1.0])
        Phi_seas = np.r_[1, -np.array(Phi)] if len(Phi) > 0 else np.array([1.0])
        Theta_seas = np.r_[1, np.array(Theta)] if len(Theta) > 0 else np.array([1.0])
        phi_full = np.convolve(phi_ns, self._expand_seasonal_poly(Phi_seas[1:]))
        theta_full = np.convolve(theta_ns, self._expand_seasonal_poly(Theta_seas[1:]))
        phi_final = -phi_full[1:]
        theta_final = theta_full[1:]
        ll, fit = self._kalman_loglik(phi_final, theta_final, sigma2)
        return ll, fit
    
    def _neg_loglik(self, params):
            i = 0
            phi = params[i:i + self.p]
            i += self.p
            theta = params[i:i + self.q]
            i += self.q
            Phi = params[i:i + self.P]
            i += self.P
            Theta = params[i:i + self.Q]
            i += self.Q
            sigma2 = np.exp(params[-1])
            ll, _ = self._sarima_loglik(phi, theta, Phi, Theta, sigma2)
            return -ll

    def fit(self):
        """
        Fits the ARMA model to the given time series. Currently, the function fits two linear regression models separately for the AR and MA components.

        Note:
            Due to some bug, this suffers from instability if the series is long.
        """
        init_params = np.r_[np.zeros(self.p + self.q + self.P + self.Q), np.log(1.0)]

        res = minimize(self._neg_loglik, init_params, method="L-BFGS-B")
        
        params = res.x
        self.phi = params[0:self.p]
        self.theta = params[self.p:self.p + self.q]
        self.Phi = params[self.p + self.q:self.p + self.q + self.P]
        self.Theta = params[self.p + self.q + self.P:self.p + self.q + self.P + self.Q]
        self.sigma2 = np.exp(params[-1])
    
        self.loglikelihood, self.fitted_values = self._sarima_loglik(self.phi, self.theta, self.Phi, self.Theta, self.sigma2)

    def summary(self):
        if not hasattr(self, "phi"):
            raise NotFittedError("One must call model.fit() before computing the summary")
        
        return f"Model:          ({self.p}, {self.d}, {self.q})x({self.P}, {self.D}, {self.Q}, {self.S})\nar:             {[round(p.item(), 3) for p in self.phi]}\nma:             {[round(p.item(), 3) for p in self.theta]}\nar.S:           {[round(p.item(), 3) for p in self.Phi]}\nma.S:           {[round(p.item(), 3) for p in self.Theta]}\nsigma2:         {self.sigma2:.3f}\nLog Likelihood: {self.loglikelihood:.3f}"

    def _integrate(self, differenced, lag=1, order=1):
        z = np.array(differenced, dtype=float)
        for j in range(order):
            # restored length = len(z) + lag
            restored = np.zeros(len(z) + lag, dtype=float)
            # pick the correct discarded values (stack of shape (order, lag))
            init = self._discarded[lag][-j - 1].numpy().astype(float)
            restored[:lag] = init
            for i in range(lag, len(restored)):
                restored[i] = z[i - lag] + restored[i - lag]
            z = restored
        return z

    def _integrate_var(self, var_differenced, lag=1, order=1):
        v = np.array(var_differenced, dtype=float)
        for _ in range(order):
            restored_var = np.zeros(len(v) + lag, dtype=float)
            # historical initial lag variances are zero (observations known)
            for i in range(lag, len(restored_var)):
                restored_var[i] = v[i - lag] + restored_var[i - lag]
            v = restored_var
        return v

    def predict(self, n_ahead, alpha=0.05):
        """
        Predicts the future values of the series.

        Args:
            n_ahead (int, optional): The number of next values to predict. Must be a positive integer. Defaults to 1.
        """
        if not isinstance(n_ahead, int) or n_ahead <= 0:
            raise ValueError("n_ahead must be a positive integer.")
        if not hasattr(self, "phi"):
            raise NotFittedError("One must call model.fit() before prediction")

        diff_hist = np.array(self.diff_series, dtype=float)
        y_hist = list(diff_hist)

        r = max(len(self.theta), len(self.Theta) * self.S, 1)
        eps_hist = [0.0] * r
        forecasts_diff = []
        ma_coeffs = np.r_[1, self.theta]
        if len(self.Theta) > 0:
            seas_ma = np.zeros(self.S * len(self.Theta) + 1)
            seas_ma[0] = 1
            seas_ma[self.S * np.arange(1, len(self.Theta) + 1)] = self.Theta
            ma_coeffs = np.convolve(ma_coeffs, seas_ma)

        psi = ma_coeffs.copy()
        forecasts_var_diff = np.zeros(n_ahead)

        for t in range(n_ahead):
            # AR part (on differenced series)
            ar_part = sum(self.phi[i] * y_hist[-i - 1] for i in range(len(self.phi)))
            ar_part += sum(self.Phi[i] * y_hist[-(i + 1) * self.S] for i in range(len(self.Phi)))

            # MA part (zero future epsilons)
            ma_part = sum(self.theta[i] * eps_hist[-i - 1] for i in range(len(self.theta)))
            ma_part += sum(self.Theta[i] * eps_hist[-(i + 1) * self.S] for i in range(len(self.Theta)))

            y_next = ar_part + ma_part
            y_hist.append(y_next)
            eps_hist.append(0.0)
            forecasts_diff.append(y_next)

            # update psi (psi-weights for forecast variance)
            if t >= len(ma_coeffs):
                next_psi = 0.0
                for j in range(len(self.phi)):
                    if j < len(psi):
                        next_psi += self.phi[j] * psi[-j - 1]
                for j in range(len(self.Phi)):
                    if (j + 1) * self.S <= len(psi):
                        next_psi += self.Phi[j] * psi[-(j + 1) * self.S]
                psi = np.append(psi, next_psi)

            # variance in differenced space
            forecasts_var_diff[t] = self.sigma2 * np.sum(psi[: t + 1] ** 2)

        forecasts_diff = np.array(forecasts_diff, dtype=float)
        forecast_var_diff = np.array(forecasts_var_diff, dtype=float)

        combined_diff = np.concatenate([diff_hist, forecasts_diff])
        combined_var_diff = np.concatenate([np.zeros(len(diff_hist), dtype=float), forecast_var_diff])

        if self.D > 0:
            combined_diff = self._integrate(combined_diff, lag=self.S, order=self.D)
            combined_var = self._integrate_var(combined_var_diff, lag=self.S, order=self.D)
        else:
            combined_var = combined_var_diff.copy()

        if self.d > 0:
            combined_diff = self._integrate(combined_diff, lag=1, order=self.d)
            combined_var = self._integrate_var(combined_var, lag=1, order=self.d)

        restored_forecasts = combined_diff[-n_ahead:]
        restored_var = combined_var[-n_ahead:]

        z = norm.ppf(1 - alpha / 2)
        ci_width = z * np.sqrt(restored_var)
        lower = restored_forecasts - ci_width
        upper = restored_forecasts + ci_width

        return torch.from_numpy(restored_forecasts), torch.from_numpy(lower), torch.from_numpy(upper)
