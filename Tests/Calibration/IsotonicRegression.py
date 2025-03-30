"""
Isotonic regression
=======================

This script showcases the use of IsotonicRegression for predicting monotonic relationships.
"""
import torch
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression as sk_isotonic

from DLL.Data.Preprocessing import data_split
from DLL.MachineLearning.SupervisedLearning.Calibration import IsotonicRegression


torch.manual_seed(0)
X = torch.linspace(0, 1, 100)
y = X ** 2 + 1 + 0.1 * torch.randn_like(X)
Xtrain, ytrain, _, _, Xtest, ytest = data_split(X, y, train_split=0.8, validation_split=0)

increasing = True
if not increasing: ytrain, ytest = -ytrain, -ytest

model = IsotonicRegression()
model.fit(Xtrain, ytrain, increasing=increasing)
ypred = model.predict(Xtest)
print(ypred)

model = sk_isotonic(increasing=increasing)
model.fit(Xtrain, ytrain)
sk_ypred = model.predict(Xtest)
print(sk_ypred)

plt.scatter(Xtrain, ytrain, label="Train", alpha=0.5)
plt.scatter(Xtest, ytest, label="Test", alpha=0.5)
plt.scatter(Xtest, ypred, label="Pred", alpha=0.5)
plt.scatter(Xtest, sk_ypred, label="SKlearn pred", alpha=0.5)
plt.legend()
plt.show()
