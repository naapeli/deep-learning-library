"""
Linear Regression with confidence intervals
==============================================

This script performs linear regression on a synthetic dataset using the 
LinearRegression class from the DLL library. It also computes and visualizes 
the 95% confidence interval for the regression line.
"""
import torch
import matplotlib.pyplot as plt
from scipy.stats import t

from DLL.MachineLearning.SupervisedLearning.LinearModels import LinearRegression


x = torch.linspace(0, 10, 100)
y = 2.5 * x + 3 + 5 * torch.randn(len(x))
x = x.unsqueeze(1)

model = LinearRegression()
model.fit(x, y)
y_pred = model.predict(x)
x = x.squeeze()

n = len(x)
dof = n - 2
s_err = torch.sqrt(torch.sum(model.residuals ** 2) / dof)

t_crit = t.ppf(0.975, dof)
x_mean = torch.mean(x)
sum_sq_x = torch.sum((x - x_mean)**2)
conf_margin = t_crit * s_err * torch.sqrt(1/n + (x - x_mean)**2 / sum_sq_x)

y_upper = y_pred + conf_margin
y_lower = y_pred - conf_margin

plt.figure(figsize=(8, 8))
plt.scatter(x.numpy(), y.numpy(), s=10, label="Data", alpha=0.6)
plt.plot(x.numpy(), y_pred.numpy(), color="red", label="Regression Line")
plt.fill_between(x.numpy(), y_lower.numpy(), y_upper.numpy(), color="red", alpha=0.2, label="95% CI")

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression with 95% Confidence Interval")
plt.legend()
plt.show()