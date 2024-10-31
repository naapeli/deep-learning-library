import torch
from math import floor

from ....DeepLearning.Losses import bce, cce
from ....DeepLearning.Layers.Activations import Sigmoid, SoftMax
from ....Data.Metrics import calculate_metrics, _round_dictionary, prob_to_pred
from ....Data.DataReader import DataReader
from ....DeepLearning.Optimisers import Adam
from ....DeepLearning.Optimisers.BaseOptimiser import BaseOptimiser
from ....Exceptions import NotFittedError
from ....Data.Preprocessing import OneHotEncoder


class LogisticRegression:
    """
    Implements a logistic regression model for binary and multi-class classification.
    
    Args:
        learning_rate (float, optional): The step size towards the negative gradient. Must be a positive real number. Defaults to 0.01.

    Attributes:
        n_features (int): The number of features. Available after fitting.
        weights (torch.Tensor of shape (n_features,)): The weights of the logistic regression model. Available after fitting.
        bias (torch.Tensor of shape (1,)): The constant of the model. Available after fitting.
    """
    def __init__(self, learning_rate=0.001):
        if not isinstance(learning_rate, float) or learning_rate <= 0:
            raise ValueError("learning_rate must be a positive real number.")

        self.learning_rate = learning_rate

    def fit(self, X, y, val_data=None, epochs=100, optimiser=None, callback_frequency=1, metrics=["loss"], batch_size=None, shuffle_every_epoch=True, shuffle_data=True, verbose=False):
        """
        Fits the LogisticRegression model to the input data by minimizing the cross entropy loss (logistic loss).

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data, where each row is a sample and each column is a feature.
            y (torch.Tensor of shape (n_samples,)): The labels corresponding to each sample. Every element must be in [0, ..., n_classes - 1].
            val_data (tuple[X_val, y_val] | None, optional): Optional validation samples. If None, no validation data is used. Defaults to None.
            epochs (int, optional): The number of training iterations. Must be a positive integer. Defaults to 100.
            optimiser (:ref:`optimisers_section_label` | None, optional): The optimiser used for training the model. If None, the Adam optimiser is used.
            callback_frequency (int, optional): The number of iterations between printing info from training. Must be a positive integer. Defaults to 1, which means that every iteration, info is printed assuming verbose=True.
            metrics (list[str], optional): The metrics that will be tracked during training. Defaults to ["loss"].
            batch_size (int | None, optional): The batch size used in training. Must be a positive integer. If None, every sample is used for every gradient calculation. Defaults to None.
            shuffle_every_epoch (bool, optional): If True, shuffles the order of the samples every epoch. Defaults to True.
            shuffle_data (bool, optional): If True, shuffles data before the training.
            verbose (bool, optional): If True, prints info of the chosen metrics during training. Defaults to False.
        Returns:
            history (dict[str, torch.Tensor], each tensor is floor(epochs / callback_frequency) long.): A dictionary tracking the evolution of selected metrics at intervals defined by callback_frequency epochs.
        Raises:
            TypeError: If the input matrix or the target matrix is not a PyTorch tensor or if other parameters are of wrong type.
            ValueError: If the input matrix or the target matrix is not the correct shape or if other parameters have incorrect values.
        """
        if not isinstance(X, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise TypeError("The input matrix and the target matrix must be a PyTorch tensor.")
        if X.ndim != 2:
            raise ValueError("The input matrix must be a 2 dimensional tensor.")
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("The targets must be 1 dimensional with the same number of samples as the input data")
        if not isinstance(val_data, list | tuple) and val_data is not None:
            raise TypeError("val_data must either be a tuple containing validation samples or None.")
        if isinstance(val_data, list | tuple) and len(val_data) != 2:
            raise ValueError("val_data must contain both X_val and y_val.")
        if isinstance(val_data, list | tuple) and len(val_data) == 2 and (val_data[0].ndim != 2 or val_data[1].ndim != 1 or val_data[0].shape[1] != X.shape[1] or len(val_data[0]) != len(val_data[1])):
            raise ValueError("X_val and y_val must be of correct shape.")
        if not isinstance(optimiser, BaseOptimiser) and optimiser is not None:
            raise TypeError("optimiser must be from DLL.DeepLearning.Optimisers")
        if not isinstance(callback_frequency, int) or callback_frequency <= 0:
            raise ValueError("callback_frequency must be a positive integer.")
        if not isinstance(metrics, list | tuple):
            raise TypeError("metrics must be a list or a tuple containing the strings of wanted metrics.")
        if (not isinstance(batch_size, int) or batch_size <= 0) and batch_size is not None:
            raise ValueError("batch_size must be a positive integer.")
        if not isinstance(shuffle_every_epoch, bool):
            raise TypeError("shuffle_every_epoch must be a boolean.")
        if not isinstance(shuffle_data, bool):
            raise TypeError("shuffle_data must be a boolean.")
        if not isinstance(verbose, bool):
            raise TypeError("verbose must be a boolean.")
        vals = torch.unique(y).numpy()
        if set(vals) != {*range(len(vals))}:
            raise ValueError("y must only contain the values in [0, ..., n_classes - 1].")

        n_samples, self.n_features = X.shape
        if len(vals) == 2:
            # normal logistic regression
            self.loss = bce()
            self.activation = Sigmoid()
            weight_shape = (self.n_features,)
            self.multiclass = False
        elif len(vals) > 2:
            # multiple logistic regression
            encoder = OneHotEncoder()
            encoder.fit(y)
            y = encoder.one_hot_encode(y)
            self.multiclass = True
            self.loss = cce()
            self.activation = SoftMax()
            weight_shape = (self.n_features, y.shape[1])

        history = {metric: torch.zeros(floor(epochs / callback_frequency)) for metric in metrics}
        batch_size = len(X) if batch_size is None else n_samples
        data_reader = DataReader(X, y, batch_size=batch_size, shuffle=shuffle_data, shuffle_every_epoch=shuffle_every_epoch)

        self.weights = torch.randn(weight_shape)
        self.bias = torch.zeros((1,)) if len(weight_shape) == 1 else torch.zeros(y.shape[1])
        optimiser = Adam(self.learning_rate) if optimiser is None else optimiser
        optimiser.initialise_parameters([self.weights, self.bias])

        for epoch in range(epochs):
            for x_batch, y_batch in data_reader.get_data():
                y_linear = x_batch @ self.weights + self.bias
                predictions = self.activation.forward(y_linear)
                bce_derivative = self.loss.gradient(predictions, y_batch)
                dCdy = self.activation.backward(bce_derivative)
                dCdweights = x_batch.T @ dCdy
                dCdbias = torch.mean(dCdy, dim=0, keepdim=(y.ndim == 1))
                self.weights.grad = dCdweights
                self.bias.grad = dCdbias
                optimiser.update_parameters()
            if epoch % callback_frequency == 0:
                values = calculate_metrics(data=(self.predict_proba(X), y), metrics=metrics, loss=self.loss.loss)
                if val_data is not None:
                    val_values = calculate_metrics(data=(self.predict_proba(val_data[0]), val_data[1]), metrics=metrics, loss=self.loss.loss, validation=True)
                    values |= val_values
                for metric, value in values.items():
                    history[metric][int(epoch / callback_frequency)] = value
                if verbose: print(f"Epoch: {epoch + 1} - Metrics: {_round_dictionary(values)}")
        return history

    def predict(self, X):
        """
        Applies the fitted LogisticRegression model to the input data, predicting the labels.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data to be classified.
        Returns:
            target values (torch.Tensor of shape (n_samples,)): The predicted values corresponding to each sample.
        Raises:
            NotFittedError: If the LogisticRegression model has not been fitted before predicting.
            TypeError: If the input matrix is not a PyTorch tensor.
            ValueError: If the input matrix is not the correct shape.
        """
        if not hasattr(self, "weights"):
            raise NotFittedError("LogisticRegression.fit() must be called before predicting.")
        if not isinstance(X, torch.Tensor):
            raise TypeError("The input matrix must be a PyTorch tensor.")
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError("The input matrix must be a 2 dimensional tensor with the same number of features as the fitted tensor.")

        probs = self.activation.forward(X @ self.weights + self.bias)
        return prob_to_pred(probs)
    
    def predict_proba(self, X):
        """
        Applies the fitted LogisticRegression model to the input data, predicting the probabilities of each class.

        Args:
            X (torch.Tensor of shape (n_samples, n_features)): The input data to be classified.
        Returns:
            target values (torch.Tensor of shape (n_samples,)): The predicted values corresponding to each sample.
        Raises:
            NotFittedError: If the LogisticRegression model has not been fitted before predicting.
            TypeError: If the input matrix is not a PyTorch tensor.
            ValueError: If the input matrix is not the correct shape.
        """
        if not hasattr(self, "weights"):
            raise NotFittedError("LogisticRegression.fit() must be called before predicting.")
        if not isinstance(X, torch.Tensor):
            raise TypeError("The input matrix must be a PyTorch tensor.")
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError("The input matrix must be a 2 dimensional tensor with the same number of features as the fitted tensor.")

        return self.activation.forward(X @ self.weights + self.bias)
