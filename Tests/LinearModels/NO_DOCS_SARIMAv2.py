import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from statsmodels.tsa.statespace.sarimax import SARIMAX
import torch

from DLL.MachineLearning.SupervisedLearning.LinearModels.TimeSeries import SARIMA


np.random.seed(1)
n = 200
phi_true = np.array([0.5, -0.3])
theta_true = np.array([0.4])
Phi_true = np.array([0.4])
Theta_true = np.array([0.3])
s = 12
sigma_true = 1.0

eps = np.random.normal(scale=np.sqrt(sigma_true), size=n + s * 7)
y = np.zeros(n + s * 7)
for t in range(max(7, s * 7), n + s * 7):
    ar_part = sum(phi_true[i] * y[t - i - 1] for i in range(len(phi_true)))
    ar_part += sum(Phi_true[i] * y[t - (i + 1) * s] for i in range(len(Phi_true)))
    ma_part = eps[t]
    ma_part += sum(theta_true[i] * eps[t - i - 1] for i in range(len(theta_true)))
    ma_part += sum(Theta_true[i] * eps[t - (i + 1) * s] for i in range(len(Theta_true)))
    y[t] = ar_part + ma_part
y = y[s * 7:]


p, d, q, P, D, Q = 2, 0, 1, 1, 0, 1
n_future = 24

res_sm = SARIMAX(y, order=(p, d, q), seasonal_order=(P, D, Q, s), trend="n").fit(disp=False)
fit_sm = res_sm.fittedvalues
forecast_sm = res_sm.get_forecast(steps=n_future)
y_forecast_sm = forecast_sm.predicted_mean
conf_int_sm = forecast_sm.conf_int()

print(res_sm.summary())


model = SARIMA(torch.from_numpy(y), (p, d, q), (P, D, Q, s))
model.fit()
y_forecast_dll, upper, lower = model.predict(n_future)

print(model.summary())


plt.figure(figsize=(12,6))
plt.plot(y, label="Observed", color="black", alpha=0.6)
plt.plot(model.fitted_values, label="DLL SARIMA (in-sample)", color="red", lw=2)
plt.plot(fit_sm, label="Statsmodels SARIMAX (in-sample)", color="blue", ls="--")
plt.plot(np.arange(len(y), len(y)+n_future), y_forecast_dll, color="red", lw=2, ls=":", label="DLL SARIMA Forecast")
plt.plot(np.arange(len(y), len(y)+n_future), y_forecast_sm, color="blue", lw=2, ls=":", label="Statsmodels Forecast")
plt.fill_between(np.arange(len(y), len(y)+n_future),
                 conf_int_sm[:, 0],
                 conf_int_sm[:, 1],
                 color="blue", alpha=0.2, label="Statsmodels 95% CI")
plt.fill_between(np.arange(len(y), len(y)+n_future),
                 lower,
                 upper,
                 color="red", alpha=0.2, label="DLL 95% CI")

plt.title(f"SARIMA(p,d,q)(P,D,Q)[s] Fit & Forecast ({n_future} steps ahead)")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.show()
