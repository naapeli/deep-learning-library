"""
Outlier detection using isolation forest
=========================================

This script samples data points from a multivariate normal distribution, adds an outlier 
and tries to detect it using `DLL.MachineLearning.UnsupervisedLearning.OutlierDetection.IsolationForest`.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

from DLL.MachineLearning.UnsupervisedLearning.OutlierDetection import IsolationForest

# %%
# Data
# ----
#
# 2000 samples from a standard normal distribution, with the first point moved to (5, 5) as an outlier.

mean = [0, 0]
cov = [[1, 0], [0, 1]]
n = 2000
X1, X2 = np.random.multivariate_normal(mean, cov, n).T
X1[0] = 5
X2[0] = 5
X = torch.from_numpy(np.array([X1, X2]).T)

# %%
# Detecting outliers
# ------------------
#
# Points that are isolated with few random splits get a high anomaly score. Every one of the 100 trees is fitted to
# a random subsample of 256 points and grown to a depth of 8, as in the original paper, so the trees stay small
# however much data there is. The scores are close to 1 for outliers and below 0.5 for normal points. The five
# largest scores are printed, and the points more than 4.7 standard deviations above the mean score are labelled as
# outliers.

model = IsolationForest(threshold=4.7)
model.fit(X)
scores = model.predict(X, return_scores=True)
predictions = model.predict(X)
print("The largest anomaly scores:", [round(score, 3) for score in torch.sort(scores, descending=True).values[:5].tolist()])
print("Detected outliers:", X[predictions].tolist())

plt.scatter(X[:, 0][predictions], X[:, 1][predictions], label="Outliers")
plt.scatter(X[:, 0][~predictions], X[:, 1][~predictions], label="Inliers")
plt.legend()
plt.show()
