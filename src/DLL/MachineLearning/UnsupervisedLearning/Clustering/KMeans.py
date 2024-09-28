import torch
from scipy.optimize import linear_sum_assignment


class KMeansClustering:
    def __init__(self, k=3, max_iters=100, init="kmeans++", n_init=10, tol=1e-5):
        self.k = k
        self.max_iters = max_iters
        self.init = init
        self.n_init = n_init
        self.tol = tol

    def fit(self, X):
        assert len(X.shape) == 2, "X has to have dimensions (n, m)."
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
            inertia = ((X - centroids[i][indicies]) ** 2).sum()
            if inertia < lowest_inertia:
                lowest_inertia = inertia
                best_centroid_index = i
            else:
                best_centroid_index = best_centroid_index
        self.centroids = centroids[best_centroid_index]

    def predict(self, X):
        assert hasattr(self, "centroids"), "KMeansClustering.fit() must be called before predicting."
        assert len(X.shape) == 2, "X has to have dimensions (n, m)."
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
        else:
            raise NotImplementedError(f'{self.init} initialization method is not implemented. Use "random" or "kmeans++" instead.')
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
