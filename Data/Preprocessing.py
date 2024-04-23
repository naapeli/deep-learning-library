import torch
from math import floor


"""
Splits the data into train, validation and test sets

X.shape = (data_length, input_shape)
Y.shape = (data_length, output_shape)
train_split = precentage of train data
validation_split = precentage of validation data
1 - train_split - validation_split = precentage of test data
"""
def data_split(X, Y, train_split=0.8, validation_split=0.2):
    assert train_split + validation_split <= 1 and validation_split >= 0 and train_split >= 0, "Splits must be between 0 and 1 and their sum less than or equal to 1."
    axis = 0
    data_length = X.size(axis)
    perm = torch.randperm(data_length, requires_grad=False)
    x_data = X.index_select(axis, perm)
    y_data = Y.index_select(axis, perm)
    split_index1 = floor(data_length * train_split)
    split_index2 = floor(data_length * (train_split + validation_split))
    x_train, y_train = x_data[:split_index1], y_data[:split_index1]
    x_val, y_val = x_data[split_index1:split_index2], y_data[split_index1:split_index2]
    x_test, y_test = x_data[split_index2:], y_data[split_index2:]
    return x_train, y_train, x_val, y_val, x_test, y_test

"""
one-hot encodes the given categorical Y labels

Y.shape = (data_length, 1)
"""
class OneHotEncoder:
    def one_hot_encode(self, data):
        assert (len(data.shape) == 2 and data.shape[1] == 1) or len(data.shape) == 1, "Y-labels must be of shape (data_length, 1)"
        unique_elements = torch.unique(data)
        self.element_to_index = {element.item(): i for i, element in enumerate(unique_elements)}
        self.index_to_element = {i: element for element, i in self.element_to_index.items()}
        one_hot_length = tuple(unique_elements.size())[0]
        label_to_distribution = torch.tensor([self._get_distribution(self.element_to_index[y.item()], one_hot_length) for y in data], requires_grad=False)
        return label_to_distribution
    
    def one_hot_decode(self, data):
        assert len(data.shape) == 2, "Input must be of shape (data_length, number_of_categories)"
        return torch.tensor([self.index_to_element[torch.argmax(tensor, dim=0).item()] for tensor in data], requires_grad=False)

    def _get_distribution(self, index, size):
        distribution = [0 if i != index else 1 for i in range(size)]
        return distribution

"""
Normalises the data between 0 and 1

data.shape = (data_length, input_shape)
"""
class MinMaxScaler:
    def fit(self, data):
        self.min = torch.min(data, dim=0).values
        self.max = torch.max(data, dim=0).values

    def transform(self, data):
        assert hasattr(self, "min"), "scaler.fit(data) must be called before transforming data"
        return (data - self.min) / (self.max - self.min)
    
    def inverse_transform(self, data):
        assert hasattr(self, "min"), "scaler.fit(data) must be called before transforming data"
        return data * (self.max - self.min) + self.min

"""
Standardises the data to 0 mean and 1 variance

data.shape = (data_length, input_shape)
"""
class StandardScaler:
    def fit(self, data):
        self.mean = torch.mean(data, dim=0)
        self.var = torch.var(data, dim=0)

    def transform(self, data):
        assert hasattr(self, "min"), "scaler.fit(data) must be called before transforming data"
        return (data - self.mean) / self.var

    def inverse_transform(self, data):
        assert hasattr(self, "min"), "scaler.fit(data) must be called before transforming data"
        return data * self.var + self.mean
