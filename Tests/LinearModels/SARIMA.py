import torch
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from src.DLL.MachineLearning.SupervisedLearning.LinearModels.TimeSeries import SARIMA


n = 1000
daily = torch.tensor([9181.231, 8966.770, 8803.231, 8635.086, 8588.066, 8624.506, 8805.777, 8849.321, 9044.870, 9126.868, 9295.427, 9440.119, 9483.297, 9526.187, 9552.341, 9763.771, 10143.724, 10098.868, 10184.532, 10175.405, 10060.573, 9783.887, 10079.360, 9959.197,])
hours_per_day = 24
n_days = 30
n_hours = n_days * hours_per_day
daily_pattern = torch.tile(daily / 10, (n_days,))
torch.manual_seed(1)
series = 2 * torch.arange(n_hours) + (0.3 + torch.arange(n_hours) / n) * daily_pattern + 10 * torch.randn(n_hours)

p = 0.7
split = int(n_hours * p)
indicies = torch.arange(n_hours)
series_train = series[:split]
indicies_train = indicies[:split]
series_test = series[split:]
indicies_test = indicies[split:]

order = (0, 1, 1)  # AR(0), since no peak at pacf in the start, SAR(1), since exponentially decreasing pacf
seasonal_order = (1, 1, 1, hours_per_day)  # MA(1), since peak at 1 in acf, SMA(1), since peak at 24 in acf.
model = SARIMA(series_train, order, seasonal_order)
model.fit()

plt.figure()
diff_series = model._differentiate(series, order=order[1])
diff_series = model._differentiate(diff_series, order=seasonal_order[1], lag=hours_per_day)
# orig_series = model._integrate(diff_series, order=order[1], lag=hours_per_day)
# orig_series = model._integrate(orig_series, order=seasonal_order[1])
plt.plot(diff_series, label="Differenced series")
# plt.plot(orig_series, label="Integrated series")
# plt.plot(series, alpha=0.5, label="Original series")
plt.legend()

fig, ax = plt.subplots(2, 1)
plot_acf(diff_series.numpy(), ax=ax[0], lags=100, title="Autocorrelation Function (ACF)")
plot_pacf(diff_series.numpy(), ax=ax[1], lags=100, method="ywm", title="Partial Autocorrelation Function (PACF)")

plt.figure()
plt.plot(indicies_train, series_train, label="train")
plt.plot(indicies_test, series_test, label="test")
preds = model.predict(steps=n_hours - split, fit_between_steps=False)
plt.plot(indicies_test, preds, label="preds")
print(f"RMSE: {torch.sqrt(torch.sum((preds - series_test) ** 2))}")
plt.legend()
plt.show()
