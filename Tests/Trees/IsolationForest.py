import torch
import numpy as np
import matplotlib.pyplot as plt

from src.DLL.MachineLearning.UnsupervisedLearning.OutlierDetection import IsolationForest


mean = [0, 0]
cov = [[1, 0], [0, 1]]
n = 2000
X1, X2 = np.random.multivariate_normal(mean, cov, n).T
X1[0] = 5
X2[0] = 5
X = torch.from_numpy(np.array([X1, X2]).T)


model = IsolationForest(n_trees=25, threshold=6)
predictions = model.fit_predict(X)
print(sorted([round(score, 2) for score in model.fit_predict(X, return_scores=True).tolist()]))

plt.scatter(X[:, 0][predictions], X[:, 1][predictions], label="Outliers")
plt.scatter(X[:, 0][~predictions], X[:, 1][~predictions], label="Inliers")
plt.legend()
plt.show()
