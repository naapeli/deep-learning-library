import torch


class BernoulliNaiveBayes:
    def fit(self, X, y, alpha=1):
        assert set(torch.unique(X).numpy()) == {0, 1}, "The features must ba labeled 0 and 1."
        self.classes = torch.unique(y)
        self.priors = torch.zeros((len(self.classes),), dtype=torch.float32)
        self.likelihoods = torch.zeros((len(self.classes), X.shape[1]), dtype=torch.float32)

        for i, cls in enumerate(self.classes):
            X_cls = X[y == cls]
            self.priors[i] = len(X_cls) / len(y)
            self.likelihoods[i] = (X_cls.sum(dim=0) + alpha) / (len(X_cls) + alpha * X.shape[1]) # laplace smoothing

    def predict(self, X):
        assert hasattr(self, "priors"), "BernoulliNaiveBayes.fit() must be called before predicting"
        assert set(torch.unique(X).numpy()) == {0, 1}, "The features must ba labeled 0 and 1."
        posteriors = torch.zeros((len(self.classes), len(X)), dtype=torch.float32)

        for i in range(len(self.classes)):
            prior = torch.log(self.priors[i])
            posterior = (X * torch.log(self.likelihoods[i]) + (1 - X) * torch.log(1 - self.likelihoods[i])).sum(dim=1) + prior
            posteriors[i] = posterior
        return self.classes[torch.argmax(posteriors, dim=0)]
