import torch


class KNNClassifier:
    _metrics = {
        "euclidian": lambda X1, X2: ((X1 - X2) ** 2).sum(dim=2).sqrt(),
        "manhattan": lambda X1, X2: torch.abs(X1 - X2).sum(dim=2),
    }

    def __init__(self, k=3, metric="euclidian"):
        self.k = k
        self.metric = KNNClassifier._metrics[metric] if isinstance(metric, str) else metric

    def fit(self, X, y):
        assert len(X.shape) == 2 and len(y.shape) == 1, "X has to have dimensions (n, m) and y (n,)."
        self.X = X
        self.y = y

    def predict(self, X):
        assert hasattr(self, "X"), "KNNClassifier.fit() must be called before predicting."
        distances: torch.Tensor = self.metric(self.X.unsqueeze(0), X.unsqueeze(1)) # (len(X), len(self.X))
        indicies = distances.topk(self.k, largest=False).indices
        k_labels = self.y[indicies]
        return torch.mode(k_labels, dim=1).values
