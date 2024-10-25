import torch

from ....Exceptions import NotFittedError


class KMeansClustering:
    """
    KMeansClustering implements the K-Means clustering algorithm, which partitions data points into `k` clusters.

    Args:
        k (int, optional): The number of clusters. Defaults to 3. Must be a positive integer.
        max_iters (int, optional): The maximum number of iterations for training the model. Defaults to 100. Must be a positive integer.
        init ({`kmeans++`, `random`}, optional): The method for initialising the centroids. Defaults to `kmeans++`.
        n_init (int, optional): The number of differently initialized centroids. Defaults to 10. Must be a positive integer.
    Attributes:
        centroids (torch.Tensor): The final chosen centroids.
        inertia (float): The total squared distance to the nearest centroid.
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
        centroids = self._initialize_centroids(X) # (n_init, k, m)
        best_centroid_index = -1
        lowest_inertia = float("inf")
        for i in range(self.n_init):
            for _ in range(self.max_iters):
                indicies = self._cluster(centroids[i], X)
                old_centroids = centroids[i].clone()
                centroids[i] = self._get_centroids(indicies, X)
                if torch.norm(centroids[i] - old_centroids) < self.tol:
                    break
            # select the centroids with the lowest inertia (total squared error)
            inertia = ((X - centroids[i][indicies]) ** 2).sum().item()
            if inertia < lowest_inertia:
                lowest_inertia = inertia
                best_centroid_index = i
            else:
                best_centroid_index = best_centroid_index
        self.inertia = lowest_inertia
        self.centroids = centroids[best_centroid_index]

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
        return self._cluster(self.centroids, X)
    
    def _initialize_centroids(self, X):
        if self.init == "random":
            initial_centroid_indicies = torch.randint(0, len(X), (self.n_init, self.k))
            centroids = torch.stack([X[initial_centroid_indicies[i]] for i in range(self.n_init)])
        elif self.init == "kmeans++":
            # https://en.wikipedia.org/wiki/K-means%2B%2B
            centroids = torch.zeros((self.n_init, self.k, X.shape[1]), dtype=X.dtype)
            centroids[:, 0] = X[torch.randint(0, len(X), (self.n_init,))]
            for i in range(self.n_init):
                for k in range(1, self.k):
                    # Calculate distance from points to the clostest of the chosen centroids
                    chosen_centroids = centroids[i, :k]
                    distances = torch.cdist(chosen_centroids, X, p=2).min(dim=0).values ** 2
                    probabilities = distances / distances.sum()
                    # Choose remaining points based on their distances
                    new_centroid_index = torch.multinomial(probabilities, num_samples=1)
                    centroids[i, k] = X[new_centroid_index]
        return centroids

    def _cluster(self, centroids, X):
        distances = ((centroids.unsqueeze(0) - X.unsqueeze(1)) ** 2).sum(dim=2).sqrt() # (n, k)
        indicies = torch.argmin(distances, dim=1)
        return indicies

    def _get_centroids(self, indicies, X):
        new_centroids = torch.zeros((self.k, X.shape[1]))
        for k in range(self.k):
            X_cls = X[indicies == k]
            new_centroids[k] = X_cls.mean(dim=0)
        return new_centroids
