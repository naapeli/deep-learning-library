"""
Support Vector Regression for 3D Surface Fitting
====================================================

This script demonstrates the use of Support Vector Regression (SVR) to model and predict a synthetic 3D surface. The objective is to train the model to approximate the surface defined by the equation:

    Z = 2 * X^2 - 5 * Y^2 + noise

The script performs the following steps:

1. Generates a synthetic 3D dataset with two input features (X, Y) and one output (Z).
2. Splits the dataset into training and test sets.
3. Trains SVR models with a quadratic kernel (product of two linear kernels) using the epsilon-insensitive and the squared epsilon-insensitive losses (:class:`~DLL.DeepLearning.Losses.EpsilonInsensitive`), and compares their predictions against a Scikit-learn SVR model with a radial basis function (RBF) kernel.
4. Visualizes the true values and the predictions from both models in a 3D scatter plot.

The comparison allows an evaluation of the model's performance in approximating the underlying surface.
"""
import torch
import matplotlib.pyplot as plt
from sklearn import svm

from DLL.MachineLearning.SupervisedLearning.SupportVectorMachines import SVR
from DLL.Data.Preprocessing import data_split
from DLL.MachineLearning.SupervisedLearning.Kernels import Linear
from DLL.Data.Metrics import mean_squared_error

# %%
# Data
# ----
#
# Noisy samples of :math:`z = 2x^2 - 5y^2` on a 20 x 20 grid.

torch.manual_seed(10)

x = torch.linspace(-2, 2, 20)
y = torch.linspace(-2, 2, 20)
XX, YY = torch.meshgrid(x, y, indexing="xy")
X = XX.flatten()
Y = YY.flatten()
X_input = torch.stack((X, Y), dim=1)
Z = 2 * X ** 2 - 5 * Y ** 2 + torch.normal(0, 1, size=X.size())
x_train, y_train, x_test, y_test, _, _ = data_split(X_input, Z, train_split=0.8, validation_split=0.2)

# %%
# Fitting the models
# ------------------
#
# The square of a linear kernel is a quadratic polynomial kernel, which matches the shape of the surface. Both
# models minimise the primal objective with Newton's method. The epsilon-insensitive loss ignores the errors
# smaller than epsilon, while the squared loss penalises the large errors more. Scikit-learn's SVR with an RBF
# kernel is fitted for comparison.

model = SVR(kernel=Linear() ** 2, epsilon=0.5)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(f"Epsilon-insensitive loss: {len(model.history['loss'])} Newton iterations, test MSE {mean_squared_error(y_pred, y_test):.3f}")

squared_model = SVR(kernel=Linear() ** 2, epsilon=0.5, loss="squared_epsilon_insensitive")
squared_model.fit(x_train, y_train)
y_pred_squared = squared_model.predict(x_test)
print(f"Squared epsilon-insensitive loss: {len(squared_model.history['loss'])} Newton iterations, test MSE {mean_squared_error(y_pred_squared, y_test):.3f}")

correct = svm.SVR(kernel="rbf", C=1, epsilon=0.5)
correct.fit(x_train, y_train)
y_pred_true = correct.predict(x_test)
print(f"Scikit-learn: test MSE {mean_squared_error(torch.from_numpy(y_pred_true), y_test):.3f}")

# %%
# Predictions on the test set
# ---------------------------

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_test[:, 0].numpy(), x_test[:, 1].numpy(), y_test.numpy(), label="True")
ax.scatter(x_test[:, 0].numpy(), x_test[:, 1].numpy(), y_pred.numpy(), label="Epsilon-insensitive")
ax.scatter(x_test[:, 0].numpy(), x_test[:, 1].numpy(), y_pred_squared.numpy(), label="Squared epsilon-insensitive")
ax.scatter(x_test[:, 0].numpy(), x_test[:, 1].numpy(), y_pred_true, label="sklearn")
ax.legend()
plt.show()
