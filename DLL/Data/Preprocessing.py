import torch
from math import floor
from itertools import combinations_with_replacement

from ..Exceptions import NotCompiledError


def data_split(X, Y, train_split=0.8, validation_split=0.2):
    """
    Splits the data into train, validation and test sets.

    Args:
        X (torch.Tensor of shape (n_samples, ...)): The input values.
        Y (torch.Tensor of shape (n_samples, ...)): The target values.
        train_split (float, optional): The precentage of train data of the whole data. Must be a real number in range (0, 1]. Defaults to 0.8.
        validation_split (float, optional): The precentage of validation data of the whole data. Must be a real number in range [0, 1). Defaults to 0.2.
    
    Returns:
        x_train, y_train, x_val, y_val, x_test, y_test (tuple[torch.Tensor]): The original data shuffled and split according to train and validation splits.
    
    Note:
        The sum of train_split and validation_split must be less than or equal to 1. The remaining samples are returned as the test data.
    """
    if not isinstance(train_split, float | int) or train_split <= 0 or train_split > 1:
        raise ValueError("train_split must be a real number in range (0, 1].")
    if not isinstance(validation_split, float | int) or validation_split < 0 or validation_split >= 1:
        raise ValueError("validation_split must be a real number in range [0, 1).")
    if train_split + validation_split > 1:
        raise ValueError("The sum of train_split and validation_split must be less than or equal to 1.")

    data_length = X.size(0)
    perm = torch.randperm(data_length, requires_grad=False, device=X.device)
    x_data = X.index_select(0, perm)
    y_data = Y.index_select(0, perm)
    split_index1 = floor(data_length * train_split)
    split_index2 = floor(data_length * (train_split + validation_split))
    x_train, y_train = x_data[:split_index1], y_data[:split_index1]
    x_val, y_val = x_data[split_index1:split_index2], y_data[split_index1:split_index2]
    x_test, y_test = x_data[split_index2:], y_data[split_index2:]
    return x_train, y_train, x_val, y_val, x_test, y_test


class OneHotEncoder:
    """
    The one-hot encoder.
    """
    def fit(self, data):
        """
        Finds the classes in the data.

        Args:
            data (torch.Tensor of shape (n_samples,) or (n_samples, n_features)): the true labels of samples.
        """
        if not isinstance(data, torch.Tensor) or (data.ndim != 1 and data.ndim != 2):
            raise ValueError("data must be a 1 or 2 dimensional torch tensor.")
        
        if data.ndim == 1: data = data.unsqueeze(1)

        # unique_elements = [torch.unique(feature) for feature in data.T]
        # self.element_to_index = [{element.item(): i for i, element in enumerate(uniques)} for uniques in unique_elements]
        # self.index_to_element = [{i: element for element, i in table.items()} for table in self.element_to_index]
        # self.one_hot_lengths = [len(uniques) for uniques in unique_elements]
        range_elements = [range(torch.min(feature).int(), torch.max(feature).int() + 1) for feature in data.T]
        self.element_to_index = [{element: i for i, element in enumerate(uniques)} for uniques in range_elements]
        self.index_to_element = [{i: element for element, i in table.items()} for table in self.element_to_index]
        self.one_hot_lengths = [len(uniques) for uniques in range_elements]

    def encode(self, data):
        """
        One-hot encodes the data. OneHotEncoder.fit() must be called before encoding.

        Args:
            data (torch.Tensor of shape (n_samples,) or (n_samples, n_features)): the true labels of samples.

        Returns:
            torch.Tensor of shape (n_samples, n_classes_1 + ... + n_classes_n_features): A one-hot encoded tensor.
        """
        if not isinstance(data, torch.Tensor) or (data.ndim != 1 and data.ndim != 2):
            raise ValueError("data must be a 1 or 2 dimensional torch tensor.")
        if not hasattr(self, "one_hot_lengths"):
            raise NotCompiledError("OneHotEncoder.fit() must be called before encoding.")

        if data.ndim == 1: data = data.unsqueeze(1)
        encoded_features = []
        for i in range(data.shape[1]):
            label_to_distribution = torch.tensor([self._get_distribution(self.element_to_index[i][y.item()], self.one_hot_lengths[i]) for y in data[:, i]], device=data.device)
            encoded_features.append(label_to_distribution)
        return torch.cat(encoded_features, dim=1)
    
    def fit_encode(self, data):
        """
        First fits the encoder and then one-hot encodes the data.

        Args:
            data (torch.Tensor of shape (n_samples,) or (n_samples, n_features)): the true labels of samples.

        Returns:
            torch.Tensor of shape (n_samples, n_classes_1 + ... + n_classes_n_features): A one-hot encoded tensor.
        """
        if not isinstance(data, torch.Tensor) or (data.ndim != 1 and data.ndim != 2):
            raise ValueError("data must be a 1 or 2 dimensional torch tensor.")

        self.fit(data)
        return self.encode(data)
    
    def decode(self, data):
        """
        One-hot encodes the data. OneHotEncoder.fit() must be called before decoding.

        Args:
            data (torch.Tensor of shape (n_samples, n_classes_1 + ... + n_classes_n_features)): the predictions of samples.

        Returns:
            torch.Tensor of shape (n_samples,) or (n_samples, n_features): A decoded predictions transformed to the original classes.
        """
        if not isinstance(data, torch.Tensor) or data.ndim != 2:
            raise ValueError("data must be a 2 dimensional torch tensor.")
        if not hasattr(self, "one_hot_lengths"):
            raise NotCompiledError("OneHotEncoder.fit() must be called before decoding.")

        decoded = []
        i = 0
        j = 0
        while i < sum(self.one_hot_lengths):
            feature = data[:, i:(i + self.one_hot_lengths[j])]
            features_decoded = [self.index_to_element[j][torch.argmax(tensor, dim=0).item()] for tensor in feature]
            decoded.append(features_decoded)
            i += self.one_hot_lengths[j]
            j += 1
        decoded = torch.tensor(decoded, device=data.device).T
        if len(self.one_hot_lengths) == 1: decoded = decoded.squeeze(1)
        return decoded

    def _get_distribution(self, index, size):
        distribution = [0 if i != index else 1 for i in range(size)]
        return distribution


class CategoricalEncoder:
    """
    The categorical encoder.
    """
    def fit(self, data):
        """
        Finds the classes in the data.

        Args:
            data (torch.Tensor of shape (n_samples,)): the true labels of samples.
        """
        if not isinstance(data, torch.Tensor) or data.ndim != 1:
            raise ValueError("data must be a 1 dimensional torch tensor.")

        self.unique_elements = torch.unique(data)
        self.element_to_key = {element.item(): i for i, element in enumerate(self.unique_elements)}

    def encode(self, data):
        """
        Encodes the data to values [0, ..., n_classes - 1]. CategoricalEncoder.fit() must be called before encoding.

        Args:
            data (torch.Tensor of shape (n_samples,)): the true labels of samples.

        Returns:
            torch.Tensor of shape (n_samples,): An encoded tensor.
        """
        if not isinstance(data, torch.Tensor) or data.ndim != 1:
            raise ValueError("data must be a 1 dimensional torch tensor.")
        if not hasattr(self, "element_to_key"):
            raise NotCompiledError("CategoricalEncoder.fit() must be called before encoding.")

        label_to_distribution = torch.tensor([self.element_to_key[y.item()] for y in data], device=data.device)
        return label_to_distribution
    
    def fit_encode(self, data):
        """
        First fits the encoder and then encodes the data.

        Args:
            data (torch.Tensor of shape (n_samples,)): the true labels of samples.

        Returns:
            torch.Tensor of shape (n_samples,): An encoded tensor.
        """
        if not isinstance(data, torch.Tensor) or data.ndim != 1:
            raise ValueError("data must be a 1 dimensional torch tensor.")

        self.fit(data)
        return self.encode(data)
    
    def decode(self, data):
        """
        Decodes the data to the original classes. CategoricalEncoder.fit() must be called before decoding.

        Args:
            data (torch.Tensor of shape (n_samples,)): the predicted labels of samples.

        Returns:
            torch.Tensor of shape (n_samples,): A decoded predictions transformed to the original classes.
        """
        if not isinstance(data, torch.Tensor) or data.ndim != 1:
            raise ValueError("data must be a 1 dimensional torch tensor.")
        if not hasattr(self, "element_to_key"):
            raise NotCompiledError("CategoricalEncoder.fit() must be called before decoding.")

        return torch.tensor([self.unique_elements[label] for label in data], device=data.device)


class MinMaxScaler:
    """
    The min-max scaler.
    """
    def fit(self, data):
        """
        Finds the minimum and the maximum of the data.

        Args:
            data (torch.Tensor): the input samples.
        """
        if not isinstance(data, torch.Tensor) or data.ndim == 0:
            raise ValueError("data must be a torch.Tensor.")

        self.min = torch.min(data, dim=0).values
        self.max = torch.max(data, dim=0).values

        if torch.any(self.max - self.min == torch.tensor(0)):
            raise ZeroDivisionError("Some features do not change and result in division by zero.")

    def transform(self, data):
        """
        Normalises the data between 0 and 1.

        Args:
            data (torch.Tensor): the input samples.

        Returns:
            torch.Tensor: the transformed data.
        """
        if not isinstance(data, torch.Tensor) or data.ndim == 0:
            raise ValueError("data must be a torch.Tensor.")
        if not hasattr(self, "min"):
            raise NotCompiledError("MinMaxScaler.fit() must be fitted before transforming.")
        
        return (data - self.min) / (self.max - self.min)

    def fit_transform(self, data):
        """
        First fits the scaler and then transforms the data.

        Args:
            data (torch.Tensor): the input samples.

        Returns:
            torch.Tensor: the transformed data.
        """
        if not isinstance(data, torch.Tensor) or data.ndim == 0:
            raise ValueError("data must be a torch.Tensor.")

        self.fit(data)
        return self.transform(data)
    
    def inverse_transform(self, data):
        """
        Scales the data back to it's original space.

        Args:
            data (torch.Tensor): the input samples.

        Returns:
            torch.Tensor: the transformed data.
        """
        if not isinstance(data, torch.Tensor) or data.ndim == 0:
            raise ValueError("data must be a torch.Tensor.")
        if not hasattr(self, "min"):
            raise NotCompiledError("MinMaxScaler.fit() must be fitted before inverse transforming.")
        
        return data * (self.max - self.min) + self.min

class StandardScaler:
    """
    The standard scaler.
    """
    def fit(self, data):
        """
        Finds the mean and the variance of the data.

        Args:
            data (torch.Tensor): the input samples.
        """
        if not isinstance(data, torch.Tensor) or data.ndim == 0:
            raise ValueError("data must be torch.Tensor.")

        self.mean = torch.mean(data, dim=0)
        self.var = torch.var(data, dim=0)

    def transform(self, data):
        """
        Transforms the data to zero mean and one variance.

        Args:
            data (torch.Tensor): the input samples.

        Returns:
            torch.Tensor: the transformed data.
        """
        if not isinstance(data, torch.Tensor) or data.ndim == 0:
            raise ValueError("data must be a torch.Tensor.")
        if not hasattr(self, "mean"):
            raise NotCompiledError("StandardScaler.fit() must be fitted before transforming.")
        
        return (data - self.mean) / torch.sqrt(self.var)

    def fit_transform(self, data):
        """
        First fits the scaler and then encodes the data.

        Args:
            data (torch.Tensor): the input samples.

        Returns:
            torch.Tensor: the transformed data.
        """
        if not isinstance(data, torch.Tensor):
            raise ValueError("data must be a torch.Tensor.")
        
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data):
        """
        Scales the data back to it's original space.

        Args:
            data (torch.Tensor): the input samples.

        Returns:
            torch.Tensor: the transformed data.
        """
        if not isinstance(data, torch.Tensor) or data.ndim == 0:
            raise ValueError("data must be a torch.Tensor.")
        if not hasattr(self, "mean"):
            raise NotCompiledError("StandardScaler.fit() must be fitted before inverse transforming.")
        
        return data * torch.sqrt(self.var) + self.mean


class PolynomialFeatures:
    """
    Polynomial features.

    Args:
        degree (int, optional): The degree of the polynomial. Must be a positive integer. Defaults to 2.
        include_bias (bool): If true, a column of ones is included. Must be a boolean. Defaults to True.
    """
    def __init__(self, degree=2, include_bias=True):
        if not isinstance(degree, int) or degree <= 0:
            raise ValueError("degree must be a positive integer.")
        if not isinstance(include_bias, bool):
            raise TypeError("include_bias must be a boolean.")
        
        self.degree = degree
        self.include_bias = include_bias

    def transform(self, data):
        """
        Creates a matrix of data containing every possible combination of the given set of features.

        Args:
            data (torch.Tensor of shape (n_samples, n_features)): the input samples.

        Returns:
            torch.Tensor of shape (n_samples, sum([nCr(n_features + deg - 1, deg) for deg in range(1, degree + 1)]) + 1): A tensor of the new features.
        """
        if not isinstance(data, torch.Tensor) or data.ndim != 2:
            raise ValueError("data must be a 2 dimensional torch tensor.")

        n_samples, n_features = data.shape
        features = [torch.ones(n_samples, device=data.device, dtype=data.dtype)] if self.include_bias else []

        for deg in range(1, self.degree + 1):
            for items in combinations_with_replacement(range(n_features), deg):
                new_feature = torch.prod(torch.stack([data[:, i] for i in items]), axis=0)
                features.append(new_feature)

        return torch.vstack(features).T
