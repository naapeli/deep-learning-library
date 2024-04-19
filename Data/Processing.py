import torch
from math import floor


"""
Splits the data into train and validation sets

X.shape = (data_length, input_shape)
Y.shape = (data_length, output_shape)
"""
def validation_split(X, Y, split=0.2):
    axis = 0
    data_length = X.size(axis)
    perm = torch.randperm(data_length, requires_grad=False)
    x_data = X.index_select(axis, perm)
    y_data = Y.index_select(axis, perm)
    split_index = floor(data_length * split)
    x_train, y_train = x_data[split_index:], y_data[split_index:]
    x_val, y_val = x_data[:split_index], y_data[:split_index]
    return x_train, y_train, x_val, y_val

"""
one-hot encodes the given categorical Y labels

Y.shape = (data_length, 1)
"""
def one_hot_encode(Y):
    assert (len(Y.shape) == 2 and Y.shape[1] == 1) or len(Y.shape) == 1, "Y-labels must be of shape (data_length, 1)"
    unique_elements = torch.unique(Y)
    element_to_index = {element.item(): i for i, element in enumerate(unique_elements)}
    one_hot_length = tuple(unique_elements.size())[0]
    label_to_distribution = torch.tensor([_get_distribution(element_to_index[y.item()], one_hot_length) for y in Y], requires_grad=False)
    return label_to_distribution

def _get_distribution(index, size):
    distribution = [0 if i != index else 1 for i in range(size)]
    return distribution
