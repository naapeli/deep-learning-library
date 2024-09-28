import torch


class PCA:
    def __init__(self, n_components=2, epsilon=1e-10):
        self.n_components = n_components
        self.epsilon = epsilon

    def fit(self, X, normalize=True):
        self.normalize = normalize
        self.mean = X.mean(dim=0)
        self.standard_deviation = X.std(dim=0, unbiased=True)
        X = (X - self.mean)
        if self.normalize:
            X = X / (self.standard_deviation + self.epsilon)
        covariance = torch.cov(X.T)
        eig_vals, eig_vecs = torch.linalg.eig(covariance)
        indicies = torch.argsort(eig_vals.real, descending=True)
        self.components = eig_vecs.real.T[indicies][:self.n_components]
        self.explained_variance = eig_vals.real[indicies][:self.n_components]
    
    def transform(self, X):
        assert hasattr(self, "mean"), "PCA.fit() must be called before transforming."
        X = (X - self.mean)
        if self.normalize:
            X = X / (self.standard_deviation + self.epsilon)
        return X @ self.components.T
