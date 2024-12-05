import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import datasets
from sklearn.metrics import silhouette_score as sk_silhouette_score, silhouette_samples
import numpy as np

from src.DLL.MachineLearning.UnsupervisedLearning.Clustering import KMeansClustering, GaussianMixture
from src.DLL.Data.Metrics import silhouette_score


# X, y = datasets.make_blobs(n_samples=1000, n_features=2, cluster_std=1, centers=3, random_state=3)
X, y = datasets.make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, n_classes=3, random_state=1)
X = torch.from_numpy(X)
y = torch.from_numpy(y)
fig, ax = plt.subplots(1, 3)
ax[0].scatter(X[:, 0], X[:, 1], c=y)
ax[0].set_title("True classes")

model = KMeansClustering(k=3, init="kmeans++", n_init=10, max_iters=100)
model.fit(X)
k_mean_labels = model.predict(X)
ax[1].scatter(X[:, 0], X[:, 1], c=k_mean_labels)
ax[1].scatter(model.centroids[:, 0], model.centroids[:, 1], c="red")
ax[1].set_title("K means clustering")
k_means_silhouette_samples = silhouette_score(X, k_mean_labels, return_samples=True)

torch.manual_seed(1)
model = GaussianMixture(k=3, max_iters=100)
model.fit(X, verbose=True)
gaussian_mixture_labels = model.predict(X)
ax[2].scatter(X[:, 0], X[:, 1], c=gaussian_mixture_labels)
ax[2].scatter(model.mus[:, 0], model.mus[:, 1], c="red")
ax[2].set_title("Gaussian mixture clustering")
gaussian_mixture_silhouette_Sampels = silhouette_score(X, gaussian_mixture_labels, return_samples=True)

fig, ax = plt.subplots(1, 2)
y_lower_k_means = 10
y_lower_gaussian_mixture = 10
n_clusters = 3
for i in range(n_clusters):
    ith_cluster_k_means_silhouette_values = k_means_silhouette_samples[k_mean_labels == i]
    ith_cluster_gaussian_mixture_silhouette_values = gaussian_mixture_silhouette_Sampels[gaussian_mixture_labels == i]

    ith_cluster_k_means_silhouette_values, _ = ith_cluster_k_means_silhouette_values.sort()
    ith_cluster_gaussian_mixture_silhouette_values, _ = ith_cluster_gaussian_mixture_silhouette_values.sort()

    size_cluster_i = ith_cluster_k_means_silhouette_values.shape[0]
    y_upper = y_lower_k_means + size_cluster_i
    color = cm.nipy_spectral(float(i) / n_clusters)
    ax[0].fill_betweenx(np.arange(y_lower_k_means, y_upper), 0, ith_cluster_k_means_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7,)
    y_lower_k_means = y_upper + 10

    size_cluster_i = ith_cluster_gaussian_mixture_silhouette_values.shape[0]
    y_upper = y_lower_gaussian_mixture + size_cluster_i
    color = cm.nipy_spectral(float(i) / n_clusters)
    ax[1].fill_betweenx(np.arange(y_lower_gaussian_mixture, y_upper), 0, ith_cluster_gaussian_mixture_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7,)
    y_lower_gaussian_mixture = y_upper + 10

ax[0].axvline(x=silhouette_score(X, k_mean_labels), color="red", linestyle="--")
ax[1].axvline(x=silhouette_score(X, gaussian_mixture_labels), color="red", linestyle="--")

ax[0].set_xlabel("The silhouette coefficient values")
ax[1].set_xlabel("The silhouette coefficient values")

ax[0].set_ylabel("Clusters")
ax[1].set_ylabel("Clusters")

ax[0].set_title(f"K-means silhouette score: {round(silhouette_score(X, k_mean_labels), 3)}, sklearn score: {round(sk_silhouette_score(X.numpy(), k_mean_labels.numpy()), 3)}")
ax[1].set_title(f"Gaussian mixture silhouette score: {round(silhouette_score(X, gaussian_mixture_labels), 3)}, sklearn score: {round(sk_silhouette_score(X.numpy(), gaussian_mixture_labels.numpy()), 3)}")
plt.show()
