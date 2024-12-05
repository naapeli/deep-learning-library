import torch
import matplotlib.pyplot as plt
from sklearn import datasets

from src.DLL.MachineLearning.UnsupervisedLearning.Clustering import KMeansClustering, GaussianMixture


# X, y = datasets.make_blobs(n_samples=1000, n_features=2, cluster_std=1, centers=3, random_state=3)
X, y = datasets.make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, n_classes=3, random_state=1)
X = torch.from_numpy(X)
fig, ax = plt.subplots(1, 3)
ax[0].scatter(X[:, 0], X[:, 1], c=y)
ax[0].set_title("True classes")

model = KMeansClustering(k=3, init="kmeans++", n_init=10, max_iters=100)
model.fit(X)
labels = model.predict(X)
ax[1].scatter(X[:, 0], X[:, 1], c=labels)
ax[1].scatter(model.centroids[:, 0], model.centroids[:, 1], c="red")
ax[1].set_title("K means clustering")

torch.manual_seed(1)
model = GaussianMixture(k=3, max_iters=100)
model.fit(X, verbose=True)
labels = model.predict(X)
ax[2].scatter(X[:, 0], X[:, 1], c=labels)
ax[2].scatter(model.mus[:, 0], model.mus[:, 1], c="red")
ax[2].set_title("Gaussian mixture clustering")
plt.show()
