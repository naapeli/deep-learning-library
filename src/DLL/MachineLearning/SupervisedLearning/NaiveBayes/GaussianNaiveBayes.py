import torch


class GaussianNaiveBayes:
    def fit(self, X, y):
        n, features = X.shape
        self.classes = torch.unique(y)
        self.means = torch.zeros((len(self.classes), features), dtype=X.dtype)
        self.vars = torch.zeros_like(self.means, dtype=X.dtype)
        self.priors = torch.zeros((len(self.classes),), dtype=X.dtype)

        for i, cls in enumerate(self.classes):
            X_cls = X[y == cls]
            self.means[i] = X_cls.mean(dim=0)
            self.vars[i] = X_cls.var(dim=0)
            self.priors[i] = len(X_cls) / n

    def predict(self, X):
        assert hasattr(self, "priors"), "GaussianNaiveBayes.fit() must be called before predicting"
        posteriors = torch.zeros((len(self.classes), len(X)), dtype=X.dtype)

        for i in range(len(self.classes)):
            prior = torch.log(self.priors[i])
            posterior = torch.log(self._pdf(X, self.means[i], self.vars[i])).sum(dim=1) + prior
            posteriors[i] = posterior
        return self.classes[torch.argmax(posteriors, dim=0)]

    def predict_proba(self, X):
        assert hasattr(self, "priors"), "GaussianNaiveBayes.fit() must be called before predicting"
        posteriors = torch.zeros((len(self.classes), len(X)), dtype=X.dtype)

        for i in range(len(self.classes)):
            prior = torch.log(self.priors[i])
            posterior = torch.log(self._pdf(X, self.means[i], self.vars[i])).sum(dim=1) + prior
            posteriors[i] = posterior
        prob_normalizers = torch.logsumexp(posteriors, dim=0)
        log_probs = posteriors - prob_normalizers
        probs = torch.exp(log_probs).T
        return self.classes, probs

    def _pdf(self, x, mean, var):
        return torch.exp(-(x - mean) ** 2 / (2 * var)) / torch.sqrt(2 * torch.pi * var)
