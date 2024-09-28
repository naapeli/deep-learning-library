import torch


class KNNRegressor:
    _metrics = {
        "euclidian": lambda X1, X2: ((X1 - X2) ** 2).sum(dim=2).sqrt(),
        "manhattan": lambda X1, X2: torch.abs(X1 - X2).sum(dim=2),
    }

    _weights = {
        "uniform": lambda distances: torch.ones_like(distances),
        "distance": lambda distances: 1 / (distances + 1e-10),
        "gaussian": lambda distances: torch.exp(-distances ** 2)
    }

    def __init__(self, k=3, metric="euclidian", weight="gaussian"):
        self.k = k
        self.metric = KNNRegressor._metrics[metric] if isinstance(metric, str) else metric
        self.weight = KNNRegressor._weights[weight] if isinstance(weight, str) else weight

    def fit(self, X, y):
        assert len(X.shape) == 2 and len(y.shape) == 1, "X has to have dimensions (n, m) and y (n,)."
        self.X = X
        self.y = y

    def predict(self, X):
        assert hasattr(self, "X"), "KNNRegressor.fit() must be called before predicting."
        distances: torch.Tensor = self.metric(self.X.unsqueeze(0), X.unsqueeze(1)) # (len(X), len(self.X))
        indicies = distances.topk(self.k, largest=False).indices
        k_values = self.y[indicies]
        k_distances = distances.gather(1, indicies)
        k_weights = self.weight(k_distances)
        return (k_values * k_weights).sum(dim=1) / (k_weights.sum(dim=1) + 1e-10)
