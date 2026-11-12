import torch

from ._centroid_utils import _nearest_centroid, _cluster_sums
from ....Exceptions import NotFittedError


class KMeansClustering:
    """
    KMeansClustering implements the K-Means clustering algorithm (Lloyd's algorithm), which partitions data points into `k` clusters.

    Args:
        k (int, optional): The number of clusters. Defaults to 3. Must be a positive integer.
        max_iters (int, optional): The maximum number of iterations for training the model. Defaults to 100. Must be a positive integer.
        init ({`kmeans++`, `random`}, optional): The method for initialising the centroids. `random` picks `k` distinct data points uniformly at random. Defaults to `kmeans++`.
        n_init (int, optional): The number of differently initialized centroids. The run with the lowest inertia is kept. Defaults to 10. Must be a positive integer.
        tol (float, optional): The iterations of a run are stopped once the Frobenius norm of the change of the centroids is below `tol`. Defaults to 1e-5. Must be non-negative.
    Attributes:
        centroids (torch.Tensor of shape (k, n_features)): The final chosen centroids.
        inertia (float): The total squared distance of the training points to their nearest centroid.
        labels (torch.Tensor of shape (n_samples,)): The cluster of each training point.
        n_iter (int): The number of iterations of the chosen run.

    Note:
        If a cluster becomes empty during training, its centroid is moved onto the training point that is farthest from its current centroid. This keeps all `k` clusters in use.
    """
    def __init__(self, k=3, max_iters=100, init="kmeans++", n_init=10, tol=1e-5):
        if not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer.")
        if not isinstance(max_iters, int) or max_iters < 1:
            raise ValueError("max_iters must be a positive integer.")
        if not isinstance(n_init, int) or n_init < 1:
            raise ValueError("n_init must be a positive integer.")
        if init not in ["random", "kmeans++"]:
            raise ValueError('init must be one of "random" or "kmeans++".')
        if not isinstance(tol, int | float) or tol < 0:
            raise ValueError("tol must be a non-negative real number.")
        self.k = k
        self.max_iters = max_iters
        self.init = init
        self.n_init = n_init
        self.tol = tol

    def fit(self, X):
        """
        Fits the KMeansClustering model to the input data by finding the best centroids.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data, where each row is a sample and each column is a feature. The number of samples must be atleast `k`.
        Returns:
            None
        Raises:
            TypeError: If the input matrix is not a PyTorch tensor.
            ValueError: If the input matrix is not the correct shape.
        """
        if not isinstance(X, torch.Tensor):
            raise TypeError("The input matrix must be a PyTorch tensor.")
        if X.ndim != 2 or X.shape[0] < self.k:
            raise ValueError("The input matrix must be a 2 dimensional tensor with atleast k samples.")
        if not torch.is_floating_point(X):
            X = X.to(torch.get_default_dtype())

        lowest_inertia = float("inf")
        for _ in range(self.n_init):
            centroids = self._initialize_centroids(X)
            centroids, labels, inertia, n_iter = self._lloyd(X, centroids)
            if inertia < lowest_inertia:
                lowest_inertia = inertia
                self.centroids, self.labels, self.n_iter = centroids, labels, n_iter
        self.inertia = lowest_inertia

    def predict(self, X):
        """
        Applies the fitted KMeansClustering model to the input data, partitioning it to `k` clusters.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data to be partitioned.
        Returns:
            labels (torch.Tensor of shape (n_samples,)): The cluster corresponding to each sample.
        Raises:
            NotFittedError: If the KMeansClustering model has not been fitted before predicting.
            TypeError: If the input matrix is not a PyTorch tensor.
            ValueError: If the input matrix is not the correct shape.
        """
        if not hasattr(self, "centroids"):
            raise NotFittedError("KMeansClustering.fit() must be called before predicting.")
        if not isinstance(X, torch.Tensor):
            raise TypeError("The input matrix must be a PyTorch tensor.")
        if X.ndim != 2 or X.shape[1] != self.centroids.shape[1]:
            raise ValueError("The input matrix must be a 2 dimensional tensor with the same number of features as the fitted tensor.")
        return _nearest_centroid(X.to(self.centroids.dtype), self.centroids)[0]

    def _lloyd(self, X, centroids):
        labels, squared_distances = _nearest_centroid(X, centroids)
        n_iter = 0
        for n_iter in range(1, self.max_iters + 1):
            old_centroids = centroids
            centroids = self._get_centroids(X, labels, squared_distances, old_centroids)
            labels, squared_distances = _nearest_centroid(X, centroids)
            if torch.linalg.norm(centroids - old_centroids) < self.tol:
                break
        # labels and distances always belong to the returned centroids
        return centroids, labels, squared_distances.sum().item(), n_iter

    def _get_centroids(self, X, labels, squared_distances, old_centroids):
        counts, sums = _cluster_sums(X, labels, self.k)
        centroids = old_centroids.clone()
        nonempty = counts > 0
        centroids[nonempty] = sums[nonempty] / counts[nonempty, None]
        empty = torch.nonzero(~nonempty).flatten()
        if len(empty) > 0:
            # Re-seed empty clusters with the points that are currently worst explained.
            farthest = torch.topk(squared_distances, len(empty)).indices
            centroids[empty] = X[farthest]
        return centroids

    def _initialize_centroids(self, X):
        n = len(X)
        if self.init == "random":
            return X[torch.randperm(n, device=X.device)[:self.k]].clone()
        # kmeans++: https://en.wikipedia.org/wiki/K-means%2B%2B
        # The squared distance to the closest chosen centroid is updated with only the newest centroid.
        centroids = torch.empty((self.k, X.shape[1]), dtype=X.dtype, device=X.device)
        centroids[0] = X[torch.randint(0, n, (1,), device=X.device)]
        closest = ((X - centroids[0]) ** 2).sum(dim=1)
        for i in range(1, self.k):
            total = closest.sum()
            if total > 0:
                index = torch.multinomial(closest / total, num_samples=1)
            else:  # every point coincides with a chosen centroid
                index = torch.randint(0, n, (1,), device=X.device)
            centroids[i] = X[index]
            closest = torch.minimum(closest, ((X - centroids[i]) ** 2).sum(dim=1))
        return centroids
