import torch
import matplotlib.pyplot as plt
from sklearn import datasets

from src.DLL.MachineLearning.UnsupervisedLearning.Clustering import KMeansClustering


X, y = datasets.make_blobs(n_samples=1000, n_features=2, cluster_std=1, centers=3, random_state=3)
X = torch.from_numpy(X)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

model = KMeansClustering(k=3, init="kmeans++", n_init=10, max_iters=100)
model.fit(X)
labels = model.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(model.centroids[:, 0], model.centroids[:, 1], c="red")
plt.show()
