import torch
from math import floor

from ....DeepLearning.Losses import MSE
from ....Data.Metrics import calculate_metrics, _round_dictionary
from ....Data import DataReader
from ....DeepLearning.Optimisers import ADAM
from ....DeepLearning.Optimisers._BaseOptimiser import BaseOptimiser
from ....Exceptions import NotFittedError
from ....DeepLearning.Losses._BaseLoss import BaseLoss


class LASSORegression:
    """
    Implements a linear regression model with L1 regularization.
    
    Args:
        alpha (int | float, optional): The regularization parameter. Larger alpha will force the l1 norm of the weights to be lower. Must be a positive real number. Defaults to 1.
        loss (:ref:`losses_section_label`, optional): A loss function used for training the model. Defaults to the mean squared error.

    Attributes:
        n_features (int): The number of features. Available after fitting.
        weights (torch.Tensor of shape (n_features,)): The weights of the linear regression model. Available after fitting.
        bias (torch.Tensor of shape (1,)): The constant of the model. Available after fitting.
        residuals (torch.Tensor of shape (n_samples,)): The residuals of the fitted model. For a good fit, the residuals should be normally distributed with zero mean and constant variance. Available after fitting.
    """
    def __init__(self, alpha=1.0, loss=MSE()):
        if not isinstance(alpha, int | float) or alpha < 0:
            raise ValueError("alpha must be a non-negative real number.")
        if not isinstance(loss, BaseLoss):
            raise TypeError("loss must be an instance of DLL.DeepLearning.Losses.")

        self.alpha = alpha
        self.loss = loss

    def fit(self, X, y, sample_weight=None, val_data=None, epochs=100, callback_frequency=1, metrics=["loss"], verbose=False):
        """
        Fits the LASSORegression model to the input data by minimizing the chosen loss function using coordinate descent.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data, where each row is a sample and each column is a feature.
            y (torch.Tensor of shape (n_samples,)): The target values corresponding to each sample.
            val_data (tuple[X_val, y_val] | None, optional): Optional validation samples. If None, no validation data is used. Defaults to None.
            epochs (int, optional): The number of training iterations. Must be a positive integer. Defaults to 100.
            callback_frequency (int, optional): The number of iterations between printing info from training. Must be a positive integer. Defaults to 1, which means that every iteration, info is printed assuming verbose=True.
            metrics (list[str], optional): The metrics that will be tracked during training. Defaults to ["loss"].
            verbose (bool, optional): If True, prints info of the chosen metrics during training. Defaults to False.
        Returns:
            history (dict[str, torch.Tensor], each tensor is floor(epochs / callback_frequency) long.): A dictionary tracking the evolution of selected metrics at intervals defined by callback_frequency.
        Raises:
            TypeError: If the input matrix or the target matrix is not a PyTorch tensor or if other parameters are of wrong type.
            ValueError: If the input matrix or the target matrix is not the correct shape or if other parameters have incorrect values.
        """
        if not isinstance(X, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise TypeError("The input matrix and the target matrix must be a PyTorch tensor.")
        if X.ndim != 2:
            raise ValueError("The input matrix must be a 2 dimensional tensor.")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError("The targets must be 1 dimensional with the same number of samples as the input data")
        if not isinstance(val_data, tuple) and val_data is not None:
            raise TypeError("val_data must either be a tuple containing validation samples or None.")
        if isinstance(val_data, tuple) and len(val_data) != 2:
            raise ValueError("val_data must contain both X_val and y_val.")
        if isinstance(val_data, tuple) and len(val_data) == 2 and (val_data[0].ndim != 2 or val_data[1].ndim != 1 or val_data[0].shape[1] != X.shape[1] or len(val_data[0]) != len(val_data[1])):
            raise ValueError("X_val and y_val must be of correct shape.")
        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("epochs must be a positive integer.")
        if not isinstance(callback_frequency, int) or callback_frequency <= 0:
            raise ValueError("callback_frequency must be a positive integer.")
        if not isinstance(metrics, list | tuple):
            raise TypeError("metrics must be a list or a tuple containing the strings of wanted metrics.")
        if not isinstance(verbose, bool):
            raise TypeError("verbose must be a boolean.")
        if not isinstance(sample_weight, torch.Tensor) and sample_weight is not None:
            raise TypeError("sample_weight must be torch.Tensor or None.")
        if isinstance(sample_weight, torch.Tensor) and (sample_weight.ndim != 1 or len(X) != len(sample_weight)):
            raise ValueError("sample_weight must be of shape (n_samples,)")

        # self.n_features = X.shape[1]
        # self.metrics = metrics
        # history = {metric: torch.zeros(floor(epochs / callback_frequency)) for metric in metrics}
        # batch_size = len(X) if batch_size is None else batch_size
        # data_reader = DataReader(X, y, batch_size=batch_size, shuffle=shuffle_data, shuffle_every_epoch=shuffle_every_epoch)

        # self.weights = torch.randn((self.n_features,))
        # self.bias = torch.zeros((1,))
        # optimiser = ADAM() if optimiser is None else optimiser
        # optimiser.initialise_parameters([self.weights, self.bias])

        # for epoch in range(epochs):
        #     for x_batch, y_batch in data_reader.get_data():
        #         predictions = self.predict(x_batch)
        #         dCdy = self.loss.gradient(predictions, y_batch)
        #         dCdweights = (x_batch.T @ dCdy) + self.alpha * torch.sign(self.weights)
        #         dCdbias = dCdy.mean(dim=0, keepdim=True)
        #         self.weights.grad = dCdweights
        #         self.bias.grad = dCdbias
        #         optimiser.update_parameters()
        #     if epoch % callback_frequency == 0:
        #         values = calculate_metrics(data=(self.predict(X), y), metrics=self.metrics, loss=self.loss.loss)
        #         if val_data is not None:
        #             val_values = calculate_metrics(data=(self.predict(val_data[0]), val_data[1]), metrics=self.metrics, loss=self.loss.loss, validation=True)
        #             values |= val_values
        #         for metric, value in values.items():
        #             history[metric][int(epoch / callback_frequency)] = value
        #         if verbose: print(f"Epoch: {epoch + 1} - Metrics: {_round_dictionary(values)}")
        # self.residuals = y - self.predict(X)
        # return history

        n_samples, self.n_features = X.shape
        X = torch.cat((torch.ones(size=(X.shape[0], 1)), X), dim=1)
        
        history = {metric: torch.zeros(floor(epochs / callback_frequency)) for metric in metrics}
        self.weights = torch.randn((self.n_features + 1))

        if sample_weight is None:
            sample_weight = torch.ones(n_samples)

        norm_cols_X = torch.sum(sample_weight[:, None] * X**2, axis=0)

        for epoch in range(epochs):
            for j in range(self.n_features + 1):
                R_j = y - X @ self.weights + self.weights[j] * X[:, j]
                rho_j = torch.dot(X[:, j], sample_weight * R_j)
                z_j = norm_cols_X[j]
                if z_j == 0:
                    continue
                w_j = torch.sign(rho_j) * max(abs(rho_j) - self.alpha, 0) / z_j
                self.weights[j] = w_j

            if epoch % callback_frequency == 0:
                loss_function = lambda prediction, true_outputs: self.loss.loss(prediction, true_outputs) + self.alpha * torch.linalg.norm(self.weights, ord=1)
                values = calculate_metrics(data=(self.predict(X), y), metrics=metrics, loss=loss_function)
                if val_data is not None:
                    val_values = calculate_metrics(data=(self.predict(val_data[0]), val_data[1]), metrics=metrics, loss=loss_function, validation=True)
                    values |= val_values
                for metric, value in values.items():
                    history[metric][int(epoch / callback_frequency)] = value
                if verbose: print(f"Epoch: {epoch + 1} - Metrics: {_round_dictionary(values)}")
        
        self.residuals = y - self.predict(X)
        return history

    def predict(self, X):
        """
        Applies the fitted LASSORegression model to the input data, predicting the correct values.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data to be regressed.
        Returns:
            target values (torch.Tensor of shape (n_samples,)): The predicted values corresponding to each sample.
        Raises:
            NotFittedError: If the LASSORegression model has not been fitted before predicting.
            TypeError: If the input matrix is not a PyTorch tensor.
            ValueError: If the input matrix is not the correct shape.
        """
        if not hasattr(self, "weights"):
            raise NotFittedError("LASSORegression.fit() must be called before predicting.")
        if not isinstance(X, torch.Tensor):
            raise TypeError("The input matrix must be a PyTorch tensor.")
        if X.ndim != 2 or (X.shape[1] != self.n_features and (X.shape[1] != self.n_features + 1 or torch.any(X[:, 0] != torch.ones(len(X))))):
            raise ValueError("The input matrix must be a 2 dimensional tensor with the same number of features as the fitted tensor.")

        if X.shape[1] != self.n_features + 1: X = torch.cat((torch.ones(size=(X.shape[0], 1)), X), dim=1)
        return X @ self.weights
